"""視覚バックエンドの差し替え機構（F1-3b）— 語彙学習(Lexicon)が使う「見ていたもの」の
作り方を、差し替え式の登録機構にする。

【なぜ要るか】F1-3のスモーク走行で、語↔物の連合の分離（assoc_sep）が発話20回で
0.013へ潰れた。原因の仮説は、連合が経由する視覚表現（taro.fusion.vision）が
太郎自身の未訓練・ランダム初期化のCNNのままだったこと（何を見ても似た出力しか
返さない）。この機構は、その視覚表現の「作り方」を差し替え可能にし、最初の
差し替え先として学習済みDINOv2を装着する（F/docs/仕様_F1-3b_視覚バックエンドの
差し替え機構.md 技術付録「新設1」）。

【インターフェース】各バックエンドは以下を持つ：
    encode(img_left, img_right) -> np.ndarray(dim,)
    dim      : 出力ベクトルの次元数
    name     : バックエンド名（ログ表示用）

【登録式（辞書）】BACKENDS に名前→クラスを登録する。get_backend(spec, ...) が
実験ファイルの taro.lexicon_vision の値からインスタンスを組み立てる。
    spec=None                                    → "custom"（後方互換の既定）
    spec={"backend": "dinov2_vits14"}            → DINOv2 ViT-S/14
    spec={"backend": "dinov2_vits14", "fovea_px": 64} → フォビア切り出し幅を指定

【フォビア切り出し（中心窩の模倣）】太郎の目のカメラは「視線の先＝画像の中央」なので、
画像の中央部を切り出してからバックエンドへ通す＝「見ているものが表現の主役になる」
ための実在の仕組みの再現。根拠ラベル：【人間模倣】中心窩（fovea）。MIMoのacuity
ぼかし（環境側・既適用、月齢依存の解像度低下）とは役割が別
（ぼかし＝解像度の年齢依存／フォビア＝注視対象の選択的表現）。
"""
import os

import numpy as np
import torch


# ============================================================================
# フォビア切り出し（中心窩）
# ============================================================================
def fovea_crop(img, fovea_px=None):
    """画像の中央 fovea_px × fovea_px を切り出す。

    fovea_px が None のときは画像の辺（高さ）の1/2を使う
    （128px画像なら中央64×64、という仕様書の既定に対応）。

    img: (H, W, 3) の配列（numpy or 同等）。
    戻り値: (fovea_px, fovea_px, 3)（画像がそれより小さければ画像そのもの）。
    """
    arr = np.asarray(img)
    h, w = arr.shape[0], arr.shape[1]
    if fovea_px is None:
        fovea_px = max(1, h // 2)
    fovea_px = int(fovea_px)
    fh = min(fovea_px, h)
    fw = min(fovea_px, w)
    top = (h - fh) // 2
    left = (w - fw) // 2
    return arr[top:top + fh, left:left + fw]


# ============================================================================
# 登録式レジストリ
# ============================================================================
BACKENDS = {}


def register(name):
    def deco(cls):
        BACKENDS[name] = cls
        return cls
    return deco


def get_backend(spec, *, vision_encoder=None):
    """taro.lexicon_vision の値からバックエンドを組み立てる。

    spec: None（既定＝custom）／文字列（バックエンド名）／
          辞書（{"backend": 名前, ...そのバックエンド固有の引数}）。
    vision_encoder: "custom"バックエンドが包む既存のVisionEncoderインスタンス
        （taro.fusion.vision）。custom以外では無視される。
    """
    if spec is None:
        spec = {"backend": "custom"}
    elif isinstance(spec, str):
        spec = {"backend": spec}
    else:
        spec = dict(spec)
    name = spec.pop("backend", None)
    if name not in BACKENDS:
        raise ValueError(
            f"未知の視覚バックエンド: {name!r}。使えるもの: {sorted(BACKENDS)}")
    return BACKENDS[name](vision_encoder=vision_encoder, **spec)


# ============================================================================
# "custom"：既存のVisionEncoder（fusion.vision）をそのまま包む（後方互換の既定）
# ============================================================================
@register("custom")
class CustomVisionBackend:
    """後方互換の既定。taro.fusion.vision（未訓練・ランダム初期化のCNN、64次元）を
    そのまま呼ぶだけで、フォビア切り出し・正規化などの追加処理は一切しない
    （既定lexicon_vision=nullで挙動が1ビットも変わらないことの根拠）。
    """
    name = "custom"

    def __init__(self, vision_encoder=None, **_ignored):
        if vision_encoder is None:
            raise ValueError(
                "custom バックエンドには taro.fusion.vision のインスタンスが要る"
                "（vision_encoder引数）。")
        self._enc = vision_encoder

    @property
    def dim(self):
        # VisionEncoder.fuse は (embedding_dim*2 -> embedding_dim) の最終層。
        # embedding_dim を決め打ちせず層の形から取る（手打ち数字の根絶）。
        return int(self._enc.fuse.out_features)

    def encode(self, img_left, img_right):
        vec = self._enc(img_left, img_right)
        return vec.detach().cpu().numpy()


# ============================================================================
# "dinov2_vits14"：torch.hub経由のDINOv2 ViT-S/14（出力384次元）
# ============================================================================
# 【出典】DINOv2（Caron et al. / Oquab et al., Meta AI）。Apache 2.0ライセンス。
#   自己教師あり（ラベルなし訓練、自己蒸留）。訓練分布はLVD-142M（ネット画像の
#   キュレーション集合）であり、太郎自身の視覚経験でも乳児の視覚経験でもない
#   （doc/人間模倣からの逸脱リスト.md その23として登録）。

# 重みキャッシュ先を <repo>/.models_cache/ に固定（初回のみダウンロード、
#   以後オフライン動作）。.gitignoreへ追加済み。
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir))
MODELS_CACHE_DIR = os.path.join(_REPO_ROOT, ".models_cache")

# DINOv2の前処理（公式実装のImageNet正規化）。
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
# ViT-S/14のパッチサイズは14。入力解像度は14の倍数であればよいが、
#   事前学習時の解像度(224)に合わせて中央フォビアを224×224へ拡大してから通す
#   （ViTの位置埋め込みが訓練時解像度に最も忠実に対応するため）。
_MODEL_INPUT_PX = 224


@register("dinov2_vits14")
class DINOv2VisionBackend:
    """torch.hub経由のDINOv2 ViT-S/14。両目それぞれフォビア切り出し→エンコード
    →平均→L2正規化した384次元ベクトルを返す。
    """
    name = "dinov2_vits14"

    def __init__(self, fovea_px=None, vision_encoder=None, **_ignored):
        # vision_encoder（taro.fusion.vision）はcustomバックエンド専用の引数だが、
        #   get_backend()が全バックエンド共通で渡すため、ここでは無視するだけ。
        self.fovea_px = fovea_px
        self._dim = 384
        os.makedirs(MODELS_CACHE_DIR, exist_ok=True)
        torch.hub.set_dir(MODELS_CACHE_DIR)
        try:
            self._model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
        except Exception as e:  # noqa: BLE001 — 取得失敗時は分かるエラーで止める
            raise RuntimeError(
                "DINOv2(dinov2_vits14)の重み取得に失敗した。初回はネット接続が要る"
                f"（キャッシュ先: {MODELS_CACHE_DIR}）。元のエラー: "
                f"{type(e).__name__}: {e}") from e
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

    @property
    def dim(self):
        return self._dim

    def _encode_one_eye(self, img):
        cropped = fovea_crop(img, self.fovea_px)
        x = torch.as_tensor(np.asarray(cropped).copy(), dtype=torch.float32)
        x = x / 255.0
        x = x.permute(2, 0, 1)                       # (3, h, w)
        x = x.unsqueeze(0)                            # (1, 3, h, w)
        x = torch.nn.functional.interpolate(
            x, size=(_MODEL_INPUT_PX, _MODEL_INPUT_PX),
            mode="bilinear", align_corners=False)
        x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
        with torch.no_grad():
            feat = self._model(x)                     # (1, 384) CLSトークン特徴
        return feat.squeeze(0)

    def encode(self, img_left, img_right):
        left_vec = self._encode_one_eye(img_left)
        right_vec = self._encode_one_eye(img_right)
        avg = (left_vec + right_vec) / 2.0
        norm = torch.linalg.norm(avg)
        if norm > 1e-12:
            avg = avg / norm
        return avg.cpu().numpy()
