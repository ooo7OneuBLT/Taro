"""自己接触（太郎の体どうし）と外界接触（柵・おもちゃなど）を区別するための共有部品。

【なぜこのファイルか、2026-08-05】太郎自身の体と外界を区別する実装は、既に3か所に
別々に存在していた（`D/scripts/d0_contact_truth.py`・`d1_contact_test.py`の名前
ヒューリスティック2種、`E/scripts/e_toy_touch.py`のbody木を辿る方式1種）。今回
また別のやり方を作ると4つ目の重複になる（設計
作業記録（非公開）
1節）。ここでは`e_toy_touch.py`の`_geoms_of_body`と同じロジックを、目標フォルダに
閉じずに`run/plugins/common/`へ切り出し、`contact_reward.py`（今回）・将来の
目標D（他者理解、2体目のエージェントを扱う）でも同じ関数をそのまま呼べるようにする。

設計：作業記録（非公開）
仕様：作業記録（非公開）

【geom名とbody名の両方を前方一致で見る理由（実装担当が事前確認済み、仕様2節）】
設計は「world_targetsはbody名の前方一致」と書いているが、実際のシーンでは
柵の柱（`E/scripts/e_toy_env.py`、`spec.worldbody.add_geom()` で作られ、
`g.name = f"fence_post_{i}"` という**geom自身の名前**しか持たない。柵の柱に
対応する独立したbodyは無く、全てworldbody（body_id==0）に直付けされている）と、
おもちゃ（`test_object1`という独立したbodyを持つ）の両方を、同じ関数で拾える
必要がある。body名だけで前方一致させると、設計4-4節の具体例
`world_targets: ["fence_post"]` そのものが空振りする。そのため
`geoms_by_name_prefix` は、各geomについて「geom自身の名前」と「そのgeomが
属すbodyの名前」の**両方**を前方一致（startswith）の対象にし、どちらかが
一致すれば採用する。
"""


def geoms_of_body_id(model, root_bid, include_children=True):
    """root_bid のbodyに属するgeom idの集合を返す。

    include_children=Trueなら、root_bidを起点にkinematic tree上の子孫bodyも
    含める（`e_toy_touch.py`の`_geoms_of_body`と同じロジック）。

    Args:
        model: mujoco model
        root_bid: int  起点のbody id
        include_children: bool  子孫bodyのgeomも含めるか（既定True）

    Returns:
        set[int]  geom idの集合
    """
    ids = set()
    targets = {int(root_bid)}
    if include_children:
        for b in range(model.nbody):
            p = b
            while p != 0:
                if p == root_bid:
                    targets.add(b)
                    break
                p = model.body_parentid[p]
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) in targets:
            ids.add(g)
    return ids


def geoms_of_body(model, root_body_name, include_children=True):
    """root_body_name（文字列）からbody idを引いて geoms_of_body_id を呼ぶだけの薄いラッパー。

    Args:
        model: mujoco model
        root_body_name: str  例 "mimo_location"・"right_hand"
        include_children: bool  既定True

    Returns:
        set[int]  geom idの集合

    Raises:
        ValueError: body名がモデルに存在しないとき（分かりやすいメッセージを出す）。
    """
    try:
        bid = int(model.body(root_body_name).id)
    except Exception as exc:
        raise ValueError(
            f"geoms_of_body: body '{root_body_name}' がモデルに存在しない: {exc}") from exc
    return geoms_of_body_id(model, bid, include_children=include_children)


def geoms_by_name_prefix(model, prefixes, include_children=True):
    """名前の前方一致でgeomを集める。geom自身の名前とbody名の両方を見る。

    【なぜ両方見るか】ファイル冒頭のdocstring参照。柵の柱はgeom自身の名前しか
    持たず、おもちゃは独立したbodyを持つ。片方だけだと`world_targets:
    ["fence_post"]`という設計の具体例そのものが空振りする。

    Args:
        model: mujoco model
        prefixes: list[str]  前方一致させたい名前の接頭辞のリスト
        include_children: bool  body名で一致した場合、そのbodyの子孫geomも
            含めるか（既定True）。geom名だけで一致した場合は、そのgeomの
            属すbodyに子を持たない構造（柵の柱等）では効果が無いだけで、
            問題は起きない。

    Returns:
        dict[int, str]  {geom_id: 一致した名前}
            一致した名前は、geom名で一致した場合はgeom名、body名で一致した
            場合はbody名（どちらで一致したかが分かる形。報告(report)で
            「どの外界物体に何回触れたか」の内訳に使う）。
    """
    out = {}
    matched_body_ids = set()     # body名の前方一致で拾ったbodyのid（子孫展開用）
    for g in range(model.ngeom):
        gname = model.geom(g).name or ""
        bid = int(model.geom_bodyid[g])
        bname = model.body(bid).name or ""
        matched_name = None
        for pfx in prefixes:
            if gname.startswith(pfx):
                matched_name = gname
                break
            if bname.startswith(pfx):
                matched_name = bname
                break
        if matched_name is not None:
            out[g] = matched_name
            if matched_name == bname:
                matched_body_ids.add(bid)

    if include_children:
        # body名で一致したbodyについては、その子孫のgeomも同じ名前で拾う。
        #   （柵の柱のようにgeom名一致・かつ子を持たないケースでは、
        #   matched_body_idsに入らないので何も起きない＝問題なし）
        for bid in matched_body_ids:
            bname = model.body(bid).name or ""
            for g2 in geoms_of_body_id(model, bid, include_children=True):
                if g2 not in out:
                    out[g2] = bname
    return out


def contact_pairs_between(data, geoms_a, geoms_b):
    """このtickで geoms_a側 と geoms_b側 が実際に接触しているペアの一覧を返す。

    MuJoCoの`data.contact.geom1`/`geom2`をncon件ループし、片方がgeoms_aに、
    もう片方がgeoms_bに入っているペアを集める
    （`E/scripts/e_toy_touch.py`の`ToyTouchProbe.update`の接触判定ループと
    同じロジック）。

    Args:
        data: mujoco data
        geoms_a: set[int]  例：toucher（探索でよく動く1部位）側のgeom集合
        geoms_b: set[int]  例：外界（柵・おもちゃ等）側のgeom集合

    Returns:
        list[tuple[int, int]]  (geoms_a側のgeom_id, geoms_b側のgeom_id) の
        ペアのリスト。1件も無ければ空リスト（`len(...)>0`で「接触あり」を
        判定できる。各要素の2つ目が「相手のgeom_id」）。
    """
    pairs = []
    for c in range(int(data.ncon)):
        g1 = int(data.contact.geom1[c])
        g2 = int(data.contact.geom2[c])
        if g1 in geoms_a and g2 in geoms_b:
            pairs.append((g1, g2))
        elif g2 in geoms_a and g1 in geoms_b:
            pairs.append((g2, g1))
    return pairs
