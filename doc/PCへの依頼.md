# PCへの依頼（クラウドの段階7スレッドから）

PCの1時間ごとの予約で読む（ユーザー承認 2026-10-17 13:28「予約のときに、クラウドが置く依頼メモ（doc/PCへの依頼.md）も読んで、太郎の実験なら回していい」）。
範囲は太郎の実験を回すことだけ。終わった依頼は状態を「済」にし、結果の置き場所を書いて、このファイルをコミット・push する（.md は .gitignore 対象なので `git add -f`）。
ブランチ：claude/project-thread-1ua42q

## 書き方（PC側の取り決め）

依頼ごとに：番号／状態（未・済）／回すコマンド（そのまま実行できる形）／何を返してほしいか。
PCの予約は commit・push をしない。結果は PC の G/logs/夜間_2026-10-17/クラウドへの返事.md に書き、朝まとめて届く。
実験以外の依頼（削除・設定・push など）は保留される。

## 依頼

### 依頼1　段階7：体ありの太郎が、見ている物の名前を言うか（3人）

状態：済（2026-10-18 PC 08:09〜08:27。名づけが11回とも取り違えで意味の辞書は不良、say_names はボール当たり6〜7／声7〜9。結果 PC G/logs/段階7_体あり_24か月_声seed0・seed1）
ブランチの最新で回す（`git pull`）。2026-10-17 夕方に24か月の声へ差し替えた。

**何をするか**：クラウドの体なしの試験場で24か月まで育てた声（種0。比べるために種1も）を体ありの太郎に読み、段階2bの3つの物の場面で
親は黙ったまま自由に見させる。太郎が声を出したら、そのとき見ていた物と一緒に数える。脳の種0・1・2を3人として回す。

**声の状態はコミット済み**：`core_v2/data/段階7_声_seed0_24か月.json`・`段階7_声_seed1_24か月.json`（各約50〜60KB）。G/ はクラウドの機械の中だけにあるので、
声はここに入れた。耳と意味の辞書は体の目の荷札で覚える必要があるので、PC で作り直す（下の1・2。数分〜十数分）。

**回すコマンド**（リポジトリの一番上で、順に）：

```
python -m core_v2.tools.hear_parent 1000 --seed 1 --out G/models/段階7_体あり_耳.pt --log-dir 段階7_体あり_耳
python -m core_v2.tools.learn_names 5 --ear G/models/段階7_体あり_耳.pt --out G/models/段階7_体あり_semantic_N5.pt --log-dir 段階7_体あり_名づけ
python -m core_v2.tools.say_names --voice core_v2/data/段階7_声_seed0_24か月.json --semantic G/models/段階7_体あり_semantic_N5.pt --taros 0,1,2 --trials 6 --trial-s 30 --log-dir 段階7_体あり_24か月_声seed0
python -m core_v2.tools.say_names --voice core_v2/data/段階7_声_seed1_24か月.json --semantic G/models/段階7_体あり_semantic_N5.pt --taros 0,1,2 --trials 6 --trial-s 30 --log-dir 段階7_体あり_24か月_声seed1
```

（3つめ・4つめは それぞれ 3人×6試行×30秒。声の出口へ渡す月齢は JSON の 24.0か月。）

**返してほしいもの**：
1. 3つめ・4つめの最後に出る1行（太郎ごとの ball・cup の looks〔1秒以上見に行った回〕・said〔見ながら声を出した回〕・hit〔当たり〕・examples〔言った声〕）
2. `G/logs/段階7_体あり_24か月_声seed0/まとめ.json` と `…_声seed1/まとめ.json` の中身
3. 2つめの最後の行（「できた：…」の meta。覚えた回数と取り違えの数）
4. 途中でエラーが出たら、その全文

**当たりの決まり**：言った声に一番近い名前（親の語の中で）が見ていた物の名前で、似かたが0.5を超える。
クラウドの予想（体なしの試験場の結果から）：ボールを見たら「ぼおる」と言う。コップを見たら、種0の声は「こく」と言う
（一番近い名前が「くつ」なので外れ）、種1の声は「こか」と言う（当たり）。体ありでは、物を1秒以上見続けて意味の辞書が
語を思い出せた回だけ声が出るので、声の回数は見に行った回より少ないはず。


### 依頼2　段階7：名づけを直して、体ありでもう一度（どこでも「ぼおる」の原因さがし）

状態：済（2026-10-18 PC 約12分半。名づけ 取り違え0・ぼおる4こっぷ7で時間切れ。say_names は依頼1と同じ数。ボールを見て1語だけの声は「ぼおる」18回／23回、箱や何もない所の「ぼおる」は喃語の連なりの中。記録 PC G/logs/段階7_体あり_24か月_v2）
ブランチの最新で回す（`git pull`。ced86e3 以降）。

**なぜ**：依頼1の名づけが11回とも取り違えだったのは、learn_names の数え方の不具合だった（意味の辞書は語を約5秒後に覚えるのに、
3回前までしか語を探していなかった）。辞書そのものは正しく覚えていたはず。ced86e3 で直した。依頼1の say_names で
箱や何もない所を見ているときにも「ぼおる」と言った原因は、まだわからない。

**回すコマンド**（耳は依頼1で作ったものを使う）：

```
python -m core_v2.tools.learn_names 5 --ear G/models/段階7_体あり_耳.pt --out G/models/段階7_体あり_semantic_N5_v2.pt --log-dir 段階7_体あり_名づけ_v2
python -m core_v2.tools.say_names --voice core_v2/data/段階7_声_seed0_24か月.json --semantic G/models/段階7_体あり_semantic_N5_v2.pt --taros 0,1,2 --trials 6 --trial-s 30 --log-dir 段階7_体あり_24か月_v2
```

**返してほしいもの**：
1. 1つめの最後の行（meta：counts・n_mixup。予想：ぼおる5・こっぷ5、取り違え0、時間切れなし）
2. 2つめの最後の1行
3. `G/logs/段階7_体あり_24か月_v2/taro0.json` の、声を出した回の記録をそのまま（声・そのとき見ていた物・時刻。長ければ最初の30回）
4. `G/logs/段階7_体あり_名づけ_v2/` の中の名づけの記録（csv があれば最初の30行）
5. 途中でエラーが出たら、その全文

**クラウドの予想**：名づけは時間切れにならず通る。say_names の数は依頼1とほぼ同じ（辞書の中身は変わらないため）で、
箱や何もない所での「ぼおる」は残る。原因の候補は、言う語の係が思い出しの弱い時もいちばん近い語に合わせてしまうこと、
物の名簿の「さっき見ていた物」の行が目を離しても残ること。3の記録で見分ける。

### 依頼3　段階8a：体ありの太郎で、驚きが「消えた」ときだけ立つか（合格の線 A の本当の試し）

状態：済（2026-10-18 PC 27分45秒、3adbada。消した30回のうち驚いた3回・強さの平均0.008、対照30試行のうち驚きが立ったのは12試行・16回。
記録 PC G/logs/段階8_体あり_驚き。クラウドの診断：**物が1度も消えていなかった**。場面の物は親が手に持っていて（toy_env の hold）毎歩
定位置へ置き直されるので、`hide` で場所を動かしても次の歩で戻っていた（消える試行と対照の試行で、立った回数が種ごとに同じだったのはこのため）。
3回の「驚いた」は対照と同じ偽の驚き。もう1つ、驚きの係が「いま見ている物」を目の真ん中から3マス以内で決めていたので、目が物の少し横に
止まると覚えが育たなかった。どちらも直して依頼4で回し直す）
ブランチの最新で回す（`git pull`）。

**なぜ**：体なしの試験場では、驚きの係（仮部品：消えた×覚え×中心の重み）は作りから A を通る（消える20回中20回、対照0回）。
本当の試しは体ありで、目の切り出しが物を拾い損ねる回や、太郎が自分で目をそらす回が、物が消えていない場面で偽の驚きを出さないか。

**何をするか**：段階2bの3つの物の場面で親は黙ったまま自由に見させる。消える試行では、5秒たったあと太郎が物を2秒続けて
見ていたら、その物を急に遠くへ移して見えなくする（動くところは見えない）。対照の試行では何もしない。意味の辞書・声は使わない。
3人×10試行×（消える・対照）×30秒。耳や意味の辞書の準備は要らない。

**回すコマンド**（リポジトリの一番上で）：

```
python -m core_v2.tools.nai_body --taros 0,1,2 --trials 10 --trial-s 30 --log-dir 段階8_体あり_驚き
```

**返してほしいもの**：
1. 最後に出る1行（まとめ：消した回数・そのうち3秒以内に驚いた回数・驚きの平均の強さ・対照で驚きが立った試行の数と回数・
   消える試行で消す前に立った回数）
2. 途中の各行（「太郎0 試行0 消える：消した ball 驚き True 立った回数 1」の形）をそのまま
3. 途中でエラーが出たら、その全文（物を遠くへ移す所 `nai_body.hide` は体なしで試せていない）

**クラウドの予想**：消した回のほとんどで驚く（3秒以内）。対照では、目を物に向けたまま切り出しが3回続けて物を拾い損ねると
偽の驚きが立つ。段階4の名簿の確かめで、切り出しの取りこぼしは「3回に1回の切り出しの間に物を探す」で減っているので、
対照30試行のうち偽の驚きが立つのは数試行（2割以下）と予想。太郎が目をそらしたときは、物が視野の外へ出るか、周辺に残って
目立ちが急に下がらないので、驚かない。

### 依頼4　段階8a：依頼3のやり直し（物を本当に消す・いま見ている物を名簿の行で決める）

状態：済（2026-10-18 PC 約25分、b53eece。消した30回のうち驚いた19回（ボール・コップ 17/21、妨害物 2/9）、強さの平均0.21。対照は30試行中12試行・16回で変わらず。
記録 PC G/logs/段階8_体あり_驚き_v2、1回ごとの記録 G/logs/夜間_2026-10-17/依頼4_trace.log。クラウドの診断：偽の驚きは、見始めて1秒たたない「かけら1つの行」が
切り出しから消えたもの（覚え0.12〜0.13）→ 1秒以上見た行だけを「あるはずの物」にした。妨害物は点滅が毎歩色を書き直して透明が戻っていた → 揺れと点滅を止めてから消す。
消す前後の左目の画像を保存するようにした。次の体ありの確かめは、言葉の段階を目だけの世界へ移すユーザーの決定があるので、置くかどうか統括と相談）
ブランチの最新で回す（`git pull`）。

**直したこと**：(1) `nai_body.hide` は場所を動かさず、前の太郎の親の消し方（parent.py の _hide_all）と同じく、物の面を透明にして材質の参照を外す
（物は親の手の中にあるが目に映らない）。(2) 驚きの係の「いま見ている物」を、名簿の今見ている行（目を向ける反射が撃った先の物）にした
（仕様書の「いま見ている物は1」どおり）。(3) 1回ごとの記録（`--trace`）を足した。

**回すコマンド**（リポジトリの一番上で）：

```
python -m core_v2.tools.nai_body --taros 0,1,2 --trials 10 --trial-s 30 --trace 2 --log-dir 段階8_体あり_驚き_v2
python - <<'PY'
import glob, json
for p in sorted(glob.glob("G/logs/段階8_体あり_驚き_v2/trace_taro0_*.jsonl")):
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    keep = [r for i, r in enumerate(rows) if r["u"] > 0 or (i + 1 < len(rows) and rows[i + 1]["u"] > 0 and rows[i]["u"] == 0)]
    print("==", p, len(rows))
    for r in keep[:40]:
        print(json.dumps(r, ensure_ascii=False))
PY
```

**返してほしいもの**：
1. 1つめの最後の1行（まとめ）と、途中の各行（「太郎0 試行0 消える：…」）をそのまま
2. 2つめの出力をそのまま（太郎0の最初の2試行の、驚きが立った回とその直前の回の記録。名簿の行・席の状態・目の入口の物の数）
3. 途中でエラーが出たら、その全文（透明にする所は体なしで試せていない）

**クラウドの予想**（`2026-10-18_段階8_途中経過.md` にも書いた）：消した30回のうち8割以上（24回以上）で3秒以内に驚き、強さは0.3〜0.7。
対照では、いま見ている物を目の切り出しが3回続けて拾い損ねたときと、名簿が同じ物を別の行に書き直したときに偽の驚きが立つ。
今回は見ている物の覚えが育つので、偽の驚きも強くなり、立つ試行の数は依頼3と同じか多い（30試行のうち12〜20）。そうなら A は体ありで通らない。
2の記録で、偽の驚きが「拾い損ね」か「書き直し」かを見分ける。

### 依頼5　段階9：見た目の素の表を作る（20語×4個体×24の向き、DINOv2）

状態：未
ブランチの最新で回す（`git pull`）。ユーザー決定（2026-10-18）：物は全部自作の形、DINOv2 の見た目の素は PC で1回作って表をクラウドへ持ってくる。

**なぜ**：段階9（名詞を6語→20語に）の体なしの試験場（`core_v2/tools/noun_bed.py`）は、今は「代わりの表」（DINOv2 ではない作り物）で回している。
本物の見た目（目の入口と同じ DINOv2＋床から浮いたまとまりの切り出し）で、同じ種類の別の個体がどれだけ近く、色がどれだけ強く出るかを測り、
試験場をその表で回し直す。表を作るのは1回だけ（決まった種、描画は乱数なし）。

**回すコマンド**（リポジトリの一番上で、順に。1つめと2つめは目で確かめるため。2つめで拾えた数が 24 のうち 18 未満なら、3つめは回さずに止めて返す）：

```
python -m core_v2.tools.make_appearance_bank --preview
python -m core_v2.tools.make_appearance_bank --check
python -m core_v2.tools.make_appearance_bank
python -m core_v2.tools.noun_bed --bank core_v2/data/段階9_見た目の素.pt --conds full,kind_off --log-dir 段階9_名詞_本物
```

1つめ（DINOv2 なし、約15秒）：`G/logs/段階9_見た目の素_確かめ/個体_食事.png・個体_お風呂.png・個体_遊び.png`（語×4個体、部屋ごと）と
`向き_dog.png・向き_cup.png`（24の向き）を描く。
2つめ（DINOv2 あり、数十秒）：8個体×3つの向きを切り出しにかけ、拾った物の場所（緑の点）と表に使う物（赤い丸）を描いた
`G/logs/段階9_見た目の素_確かめ/切り出しの確かめ.png` を作り、「確かめ：拾えた n/24」を出す。
3つめ（約2〜5分の見込み）：`core_v2/data/段階9_見た目の素.pt`（約2MB）を作る。最後に「できた：{…}」の1行（拾えた割合・
同じ種類の別の個体／別の種類／別の種類で同じ色の似かた）。
4つめ（時間があれば。太郎5人×2条件、1人約2分）：本物の表で試験場を回す。最後に条件ごとの1行。

**返してほしいもの**：
1. 2つめの「確かめ：拾えた n/24」の行と、`切り出しの確かめ.png`・`個体_遊び.png` の画像（見て、物が真ん中に描けているか・赤い丸が物の上か、を一言）
2. 3つめの各語の行（「わんわん 拾えた 96/96」の形）と、最後の「できた：…」の1行をそのまま
3. `core_v2/data/段階9_見た目の素.pt` そのもの（push はしない。クラウドへの運び方はユーザー・統括が決める）と、その大きさと sha256
   （`python -c "import hashlib;print(hashlib.sha256(open('core_v2/data/段階9_見た目の素.pt','rb').read()).hexdigest())"`）
4. 4つめを回したら、条件ごとの最後の1行と `G/logs/段階9_名詞_本物/まとめ.json`
5. 途中でエラーが出たら、その全文

**クラウドの予想**：拾えた割合は9割以上（物は1つずつ、画面の真ん中、壁と床の前）。拾えないのは、細いすぷうん・平たいおさら・たおるを
真横や上から見た向き。似かた（荷札64）：同じ種類の別の個体 0.55〜0.7、別の種類 0.35〜0.5、別の種類で同じ色 0.45〜0.6
（単色の単純な形なので、色が形と同じくらい強く出る＝仕様書の心配どおり）。

### 依頼10　段階9：目だけの場面で、段3（名前）を本物の物と本物の目で1回だけ育て直し、段4・段5もその上で

状態：未
ブランチの最新で回す（`git pull`）。ユーザー承認（2026-10-18）：段3から1回だけ育て直す（まばらな意味の辞書・本物の物）。
本物の物のファイル（G/assets/real/）と DINOv2 の重みは PC にある前提。物の作り直しが要るなら先に `python -m core_v2.tools.make_real_objects`。

**何をするか**：目だけの場面に、部屋3つと20語×4個体を出せるようにした（`world_eye/noun_scene.py`）。体なしの試験場（noun_bed）と同じ親で、
3つ並べて12秒、太郎が見始めて0.5秒で音（半分）、1.5秒で名前（半分）、見ている間は1秒ごとに回す。太郎は自分で目を動かす（41係の脳そのまま）。
段1（耳、20語の話しかけ）→段2（喃語、段階7の声を読む）→段3（名前）の写しを育ち帳に残し、段3の写しで段階9の A・B4・B2・C を体ありで試す。
クラウドでは「代わりの目」（DINOv2 の表を引く）と自作の物で、3場面だけ回して道具が最後まで動くことを確かめた（下の予想の欄）。

**回すコマンド**（リポジトリの一番上で、順に）：

```
python -m core_v2.tools.grow_early 1 --seed 0
python -m core_v2.tools.grow_early 2 --seed 0
python -m core_v2.tools.growth list
python -m core_v2.tools.grow_names --check --objects real --eye dinov2 --seed 0
```
ここで一度止まって、4つめの画像を見る：`G/logs/段階9_段3_目_real_dinov2_種0_確かめ/<部屋>/ep001_seed0/` の `*_show_*_after.png`（並べた3つの物が写っているか）と
`*_say.png`（名前を言った回。そのとき画面の真ん中にある物が、events.csv の say の行の object と同じ物か）。
3つとも写っていて、名前の物が真ん中なら、続ける（写っていなければ止めて、画像と events.csv を返す）：

```
python -m core_v2.tools.grow_names --objects real --eye dinov2 --seed 0 --scenes 300
python -m core_v2.tools.noun_body --from auto --seed 0 --eye dinov2 --words わんわん,こっぷ --trials 1 --image-trials 8 --log-dir 段階9_体あり_確かめ
python -m core_v2.tools.noun_body --from auto --seed 0 --eye dinov2 --trials 10
python -m core_v2.tools.say_names --from auto --objects real --eye dinov2 --growth-seed 0 --taros 0,1,2 --trials 6 --trial-s 30 --log-dir 段階9_段4_名前を言う
python -m core_v2.tools.nai_body --from auto --growth-seed 0 --taros 0,1,2 --trials 10 --trial-s 30 --log-dir 段階9_段5_ない
python -m core_v2.tools.growth list
```

（6つめは試しの短い確かめ：試しの絵 `G/logs/段階9_体あり_確かめ_real_dinov2_種0/<部屋>/ep001_seed0/*_show_*_after.png` に4つ・2つの物が写っているか。
**変更（2026-10-18 夜）**：say_names に `--objects real --eye dinov2` を足した（段階2bの丸と箱は20語の太郎が見たことのない形なので、段階9の部屋で、覚えた ぼおる・こっぷ をほかの2語と並べて数える）。もう前の行で回していたら、この行でもう1回だけ回す。結果は `G/logs/段階9_段4_名前を言う_real_dinov2_種0/まとめ.json`。
say_names と nai_body は `--from auto` で段3（段4）の写しを読み、終わりに段4・段5の写しを残す。この2つは確かめだけで覚えないので、写しの中身は元と同じ。）

**かかる時間（見込み。クラウドの実測からの推定）**：クラウド（CPU で描く、代わりの目）で段3は1場面 約5〜6.5秒（1回 約40ミリ秒）、試しは1回 約1.6秒。
PC は描くのが速く、そのかわり DINOv2 と切り出しが1回 約30ミリ秒かかるので、1回 約40〜60ミリ秒と見込む。
段1・段2：数秒ずつ。段3（300場面）：約30〜45分。試し（20語×3×10回＋音7語×10回）：約20〜30分。say_names：約10分。nai_body：約15〜20分。全部で約1.5〜2時間。

**返してほしいもの**：
1. `growth list` の2回の出力（段1〜段5の写しの行）
2. 4つめ（--check）の画像のうち、show の後の絵1枚と say の絵1枚、それを見ての一言（物が写っているか、名前の物が真ん中か）
3. 5つめの10場面ごとの行（名づけ・音・覚えた・耳が知っている語）と最後の1行、`G/logs/段階9_段3_目_real_dinov2_種0/まとめ.json`
4. 7つめの最後の1行と `G/logs/段階9_体あり_real_dinov2_種0/まとめ.json`（A・B4・B2・C と語ごと）
5. say_names・nai_body の最後の行と、それぞれの `まとめ.json`（`G/logs/段階9_段4_名前を言う_real_dinov2_種0/`・`G/logs/段階9_段5_ない_目だけ/`）
6. 途中でエラーが出たら、その全文

**クラウドの予想**：段3の300場面で名づけは約150〜350回（クラウドの10場面で 4回、3場面で 2回。太郎の見方は1.2秒前後の短いものと5〜12秒の長いものに分かれ、
1.5秒に届かない見方が多い）。耳が知っている語は最後に20/20。体ありの試し（本物の物・本物の目）は、試験場（A 98%・B4 96%・B2 98%・C 93%）より下がる：
A 70〜90%（16語以上が70%以上に届くかは半々）、B4 65〜85%、B2 70〜90%、C 80〜95%。名づけの数が試験場（約600回）より少ないのと、
本物の目が物を拾い損ねる回があるため。say_names は段階7（ボール当たり6〜7）と同じくらい、nai_body は段階8の体ありと同じくらい。

### 依頼12　段階9：見る時間・自分の声・「ばす」の3つの直しのあと、段1から本番を回し直す
状態：未。依頼11（と依頼10の残り）をこちらに置き換える（依頼11は回さない）。
**まず `git pull`**（ブランチ claude/project-thread-1ua42q の最新。直しが3つ入っている）。それから、リポジトリの一番上で順に:

直したこと（クラウド、2026-10-18）：
* 目が動いた分を、目を動かす指令の写しで先に知る（eyein。画像のずれは小さな直しだけ）。「今見ている物」の印が物から外れなくなり、
  見る時間が 1.2秒（真ん中）→ 4.6秒（クラウドの代わりの目・40場面。目の動きの見積もりが3マス以内：8〜12マスの跳び 36%→100%、
  12〜16マス 4%→100%。名づけ 37→56回、覚えた 29→55回。コミット 4282274・c124c01・d3ca7c6 の説明）
* 耳は太郎が話している間も聞く。自分の声の写し（口の帳面で予想した音）に合う音だけを自分の声とし、親の名前は捨てない
* 段1の親の話しかけで、20語どれも「語だけ」で4回以上聞く（「ばす」が1回だけだった）。**段1が変わったので段1から育て直す**
  （古い 段1_耳_種0・段2_喃語_種0・段3 の写しは「古い」と判定され、--from auto では使われない）

1. `python -m core_v2.tools.grow_early 1 --seed 0`（数秒。育ち帳の known_words に20語ぜんぶ、ばす が単独で入っていること。「ばすだよ」などの文が入っていないこと。クラウドでは27語＝20語＋「いいこだね」などの文7つ）
2. `python -m core_v2.tools.grow_early 2 --seed 0`（数秒）
3. `python -m core_v2.tools.growth list`（新しい 段1_耳_種0_2・段2_喃語_種0_2 のような行ができていること）
4. `python -m core_v2.tools.grow_names --check --objects real --eye dinov2 --seed 0`（1場面。場面の始めに目が (0,-20) で、物が写っていることを絵で確かめる）
5. `python -m core_v2.tools.grow_names --objects real --eye dinov2 --seed 0 --scenes 1000`（見込み 約1時間45分）。まとめ.json の learn_min が5未満なら `--from <できた段3の写しの名前>` で続けて回す。
6. `python -m core_v2.tools.noun_body --from auto --seed 0 --eye dinov2 --words わんわん,こっぷ --trials 1 --image-trials 8 --log-dir 段階9_体あり_確かめ` → 良ければ `python -m core_v2.tools.noun_body --from auto --seed 0 --eye dinov2 --trials 10`
7. `python -m core_v2.tools.say_names --from auto --objects real --eye dinov2 --growth-seed 0 --taros 0,1,2 --trials 6 --trial-s 30 --log-dir 段階9_段4_名前を言う`
8. `python -m core_v2.tools.nai_body --from auto --growth-seed 0 --taros 0,1,2 --trials 10 --trial-s 30 --log-dir 段階9_段5_ない`
9. `python -m core_v2.tools.smith_yu --world eye --seeds 0`
10. `python -m core_v2.tools.growth list`
（任意・約3分）PCの本物の目でも見る時間を測るなら：`python -m core_v2.tools.look_diag --scenes 40 --objects real --eye dinov2 --from <1〜2でできた段2の写しの名前>` と、同じものに `--ego image`（直し前の決まり）を足したもの。

返してほしいもの: 1の known_words、3と10の growth list、4の絵を見た一言、5の10場面ごとの行と learn_by_word・learn_min（名づけの数がクラウドの見込みどおり増えたか）、
6〜9の最後の行とまとめ.json、（回したなら）look_diag の2つのまとめ.json、エラー全文。写しと記録の json はコミット・push してよい（G/assets は入れない）。

### 依頼13　家の場面の下見：AI2-THOR が Windows で動くか、赤ちゃんの目の高さ（約60cm）の絵が出るか

状態：済（2026-10-19 Windows用ビルドが配られておらず動かず。依頼14へ）（2026-10-19 ユーザー承認「いいよ」03:25。依頼12の鎖が終わってから。GPUを取り合わないように）

**ねらい**：太郎の世界を「手を抜かずに作られた家」に移す下見。調べた結果は `/mnt/project-files/太郎/文献調査/2026-10-19_家の場面_AI2-THORとVirtualHome.md`。
公式に配られているのは Mac・Linux 用だけ（ai2thor 5.0.0、2022-12）。Windows で動くかを、軽い順に試す。

**太郎の .venv には入れない。** 別の箱を作る：`py -3.10 -m venv G\envs\ai2thor`（3.10 が無ければ手元の 3.x）→ `G\envs\ai2thor\Scripts\python -m pip install ai2thor pillow`

**試す順**（どれかで絵が出たら、そこで止めてよい）：
1. そのまま：`from ai2thor.controller import Controller; c = Controller(scene="FloorPlan1", width=640, height=480)`。ビルドの自動ダウンロードが Windows で通るか。
2. Windows 用ビルド指定：`from ai2thor.platform import StandaloneWindows64` があれば `Controller(platform=StandaloneWindows64, scene="FloorPlan1")`（無ければ `pip install --upgrade git+https://github.com/allenai/ai2thor.git` の最新で再挑戦）。
3. WSL2 の Ubuntu が入っていれば、その中で `pip install ai2thor` → `Controller(platform=CloudRendering, scene="FloorPlan1")`。
4. どれも駄目なら、ここで止めて、各段のエラーの最後の20行を返す（Unity Editor 2020.3.25f1 を入れて中身から動かす案は、ユーザーに聞いてから）。

**絵が出たら**（下の短いスクリプトで。G/scripts/ai2thor_shita_mi.py として置いてよい）：
- 台所 FloorPlan1・居間 FloorPlan201・寝室 FloorPlan301・お風呂 FloorPlan401 の4つで、ふつうの目線の絵を1枚ずつ `c.last_event.frame` から PNG に保存。
- 同じ4部屋で、目を床から約0.6mに置いた絵：`c.step(action="AddThirdPartyCamera", position=dict(x=<エージェントのx>, y=0.6, z=<z>), rotation=dict(x=10, y=<向き>, z=0), fieldOfView=90)` → `c.last_event.third_party_camera_frames[0]` を保存。うまくいかなければ `c.step(action="Crouch")` の絵でもよい。
- 速さ：`RotateRight` を200回して、1秒あたりの回数。
- 物を1つ拾えるか：`c.step(action="PickupObject", objectId=<見えている拾える物のid>)` の成否。

**返してほしいもの**：どの手で動いたか（または全部のエラー）、1秒あたりの回数、GPUのメモリ使用量、絵8枚（前回の動画と同じやり方で共有フォルダへ。置けなければ置いた場所）。

### 依頼14　WSL2 を入れて、AI2-THOR を Ubuntu の中で動かす（依頼13の続き）

状態：済（2026-10-19 04:26 窓あり WSLg で成功・GPU描画、CloudRendering は Vulkan が無く不可。毎秒17.9回）（2026-10-19 04:08 ユーザー「僕から見えないっていうのが気になるけど、一回やってみよう」）

**WSL2 を入れる**（管理者の PowerShell。再起動が要るので、上野さんに声をかけてから）：
1. `wsl --install -d Ubuntu-22.04` → 再起動 → Ubuntu が開いたら利用者名とパスワードを作る（上野さんに入力してもらう）。
2. Ubuntu の中で `nvidia-smi` が GPU を表示するか（Windows の NVIDIA ドライバが新しければ出る）。出なければ、その表示を返して止める。

**AI2-THOR**（Ubuntu の中。リポジトリは Windows 側の C:\claude\AI\Taro を /mnt/c/claude/AI/Taro から使ってよい）：
3. `sudo apt update && sudo apt install -y python3-venv python3-pip` → `python3 -m venv ~/ai2thor && ~/ai2thor/bin/pip install ai2thor pillow`
4. 依頼13 のスクリプト G/scripts/ai2thor_shita_mi.py を、まず `platform=CloudRendering`（画面なし）で回す：4部屋×ふつうの目線と0.6mの目線の絵8枚・1秒あたりの回数・物を拾えるか・GPUのメモリ。
5. **見えるかの確かめ**：同じスクリプトを platform 指定なし（Linux64、窓あり）で回し、Windows の画面に AI2-THOR の窓が出るか（WSLg。Windows 11 なら出るはず）。出たら画面の写真を1枚。

**返してほしいもの**：各段の成否、絵8枚と窓の写真（共有フォルダへ）、1秒あたりの回数、GPUメモリ。どこかで詰まったらエラーの最後の20行を返して止める。

### 依頼15　AI2-THOR：赤ちゃん目線から体を消す＋「見えない親の手」が物を見せる短い動画

状態：未（2026-10-19 依頼14の続き。WSL の ~/ai2thor、窓あり＝Linux64・server_timeout=900）

1. `Controller(..., makeAgentsVisible=False)` で、60cm の絵にエージェントの体が映らないか。映るなら AddThirdPartyCamera を体の前 0.3m に出す。4部屋の60cmの絵を撮り直す。
2. 居間 FloorPlan201 で、60cm の目の前に「見えない親の手」が物を見せる20秒：
   - 部屋にある拾える物を1つ選ぶ（なければ `CreateObject` 系は使わず、近くの拾える物でよい）。
   - その物を目の前 0.4〜0.5m へ置き直し（`PlaceObjectAtPoint` か `TeleportObject`）、1秒に2回くらい左右に少し動かす（振る）。0.1秒ごとに絵を1枚。
   - 5秒ごとに別の物へ替える（計4つ）。
   - 絵を mp4（ffmpeg が無ければ gif）にする。
3. 0.1秒ごと1歩で、20秒（200歩）が実時間で何秒かかったか。

**返してほしいもの**：体が消えたか、60cmの絵4枚、動画、使えた物の名前、かかった秒数。詰まったらエラーの最後の20行。

### 依頼16　家の場面①：太郎の脳を AI2-THOR の家につなぎ、20秒の動画で見る

状態：未（2026-10-19 計画①、ユーザー承認済み。依頼15のあとに。WSL の Ubuntu 22.04、窓あり＝Linux64）

**ねらい**：目だけの場面（MuJoCo の机）の代わりに、AI2-THOR の台所に太郎の目（床から0.6m、画角60度）を置き、41係の脳・目の入口（DINOv2）を
そのままつなぐ。見えない親の手（follow-in）：太郎が物を1秒見続けたら、その物を目の前0.3mへ持ってきて ±3cm・2回/秒で振り、名前を言い
（見続けていれば2秒ごとに最大2回言い直す）、5秒で元へ戻す。名前を言える物は りんご・ぱん・こっぷ・おさら・すぷうん・ほん・りもこん・まくら・くま・ぼおる。
コードは `core_v2/world_thor/`（説明は __init__.py）、道具は `core_v2/tools/thor_watch.py`（`watch_scene --world thor` でも同じ）。
クラウドでは偽物の家（world_thor/fake.py）と代わりの目（seg）でしか確かめていない。本物の AI2-THOR・本物の目（dinov2）はここが初めて。

**WSL に太郎の箱を作る**（依頼14の ~/ai2thor とは別。AI2-THOR のビルドは ~/.ai2thor に落ちているので、取り直しにならない）：
1. リポジトリは WSL の中へ clone する（/mnt/c 越しは読み書きが遅い）：
   `cd ~ && git clone https://github.com/ooo7OneuBLT/Taro.git taro_repo && cd taro_repo && git checkout claude/project-thread-1ua42q && git pull`
   （2回目からは `git pull` だけ。育ち帳の写し core_v2/data/育ち/*.pt と 育ち帳.json はリポジトリに入っているので、そのまま使える）
2. `python3 -m venv ~/taro && source ~/taro/bin/activate`
3. torch は CUDA 版（requirements.txt の①と同じ版）：`pip install torch==2.14.0+cu130 torchvision==0.29.0+cu130 --index-url https://download.pytorch.org/whl/cu130`
   （WSL の nvidia-smi が示す CUDA が 13 より古くて入らなければ、cu126 か cu121 の索引で同じ系の版を入れ、入れた版を返す）
4. `pip install -r requirements.txt`（mobile_sam は git から。通らなければ飛ばしてよい：家の場面の dinov2 の目は MobileSAM を使わない）→ `pip install ai2thor imageio-ffmpeg`
5. DINOv2 を取り直さないよう、Windows の torch.hub の置き場を写す：
   `mkdir -p ~/.cache/torch && cp -r /mnt/c/Users/<利用者名>/.cache/torch/hub ~/.cache/torch/`
   （中に facebookresearch_dinov2_main と checkpoints/dinov2_vits14_pretrain.pth があること）
6. 確かめ：`python -m core_v2.world_eye.eye_backend` が「作れる」、`python -c "import torch; print(torch.cuda.is_available())"` が True
7. 日本語の字体（動画の字）：`sudo apt install -y fonts-ipafont-gothic`（上野さんにパスワードを頼む。無くても動くが字が四角になる）

**回すコマンド**（~/taro_repo で、`source ~/taro/bin/activate` のあと。負荷の決まり core_v2/perf.py はそのまま効く：torch のスレッド1本。
変えるときだけ TARO_THREADS。AI2-THOR の窓が Windows の画面に出る。閉じないこと）：

```
python -m core_v2.tools.thor_watch --seconds 20 --scene FloorPlan1 --eye dinov2 --stills 3
```

* 始める太郎は、目だけの場面で育った写しのいちばん上の段（段5_目 があればそれ）。表示の「始める太郎」を返す
* 最初の起動は遅い（server_timeout=900）。20秒（200回）が実時間で何秒かかったかも表示される
* 通ったら、居間でも1回：`--scene FloorPlan201`（りもこん・まくら・ほん が出るか）

**返してほしいもの**：
- 最後の3行（[家の動画] 動画…／親：持ってきた・名づけ・語ごと／見た時間：回数・平均・真ん中・最長・1秒以上・語ごと）と、始めの行（名前を言える物の一覧）
- 動画 G/logs/動画/家の場面_FloorPlan1_dinov2_種0.mp4 と静止画3枚（_1〜_3.png）を、前回と同じやり方で共有フォルダへ。_まとめ.json と events_*.csv は返事に添える
- 本物の目の様子のひとこと：黄色い四角（太郎が見ていると思っている所）が物に乗っているか、家具や壁ばかりを物として拾っていないか
  （目の入口の切り出しは「床から浮いたまとまり」を拾う決まりで、MuJoCo の無地の床に合わせてある。家の床と家具でどうなるかが、いちばん知りたいこと）
- `thor_fail` が0でなければ、events_*.csv の teleport_fail・camera_fail の行。詰まったらエラーの最後の20行
- かかった時間、GPU のメモリ（nvidia-smi）

### 依頼17　家の場面②：座る場所を選ぶ・親がひとりで物を見せる（bring-in）を入れて、依頼16 と同じ20秒の動画をもう一度

状態：未（2026-10-19 依頼16 の結果を受けて。依頼16 では FloorPlan1 で冷蔵庫の扉の目の前、FloorPlan201 でソファの後ろの壁ぎわに座り、
名前を言える物がほとんど目に入らず、持ってきた 0・名づけ 0 だった）

**変えたこと**（core_v2/world_thor。クラウドでは偽物の家でテスト済み）：
1. 座る場所（`--place auto`、既定）：reset のとき GetReachablePositions の場所から最大60か所 × 向き 0/90/180/270度を、メタデータの物の場所から
   幾何で数え（描かない）、「名前を言える拾える物が 0.5〜2m にあって目（下へ10度・画角60度）の絵に入る数」がいちばん多い所へ Teleport する。
   真正面 0.6m 以内に名前の無い大きな物（冷蔵庫・ソファ…）があると減点、0.5m 先の床が歩けない（壁ぎわ）と減点。閉じた冷蔵庫・戸棚の中の物は数えない。
   `--place start` で今までどおり（エージェントの始めの場所）
2. 親がひとりで見せる（bring-in、旧太郎の solo_presentation）：太郎が名前の物を5秒見ず、親の手が空いていれば、近い物（3m 以内、見せた回数の少ない順）を
   目の前 0.3m・視線の先へ持ってきて振る（follow-in と同じ）。名前は太郎がその物を見てから言う（見はじめて 0.5秒、見続ければ2秒ごとに言い直し）。
   見たら5秒で戻す、一度も見なければ8秒で言わずに戻す。`--bring-in 0` で切れる
3. 60度の画角・指令の写し・見た時間の記録（LookLog）は変えていない

**準備**（WSL の Ubuntu、~/taro_repo。依頼16 と同じ箱 ~/taro）：
PC の取り決めどおり G 側（Windows の作業場所）が正。Windows の作業場所で `git pull` したあと、WSL へ写す（依頼16 のときと同じやり方。
<Windows の作業場所> は依頼16 で使った /mnt/c/… の場所）：

```
source ~/taro/bin/activate
rsync -a <Windows の作業場所>/core_v2/ ~/taro_repo/core_v2/
rsync -a <Windows の作業場所>/run/world/ ~/taro_repo/run/world/
cd ~/taro_repo
python -m pytest core_v2/tests/test_thor_world.py -q      # 15 passed になること（偽物の家。AI2-THOR は起こさない）
```

**回すコマンド**（依頼16 と同じ2本。始める太郎は 段5_目_種0_2 にそろえる）：

```
python -m core_v2.tools.thor_watch --seconds 20 --scene FloorPlan1 --eye dinov2 --from 段5_目_種0_2 --stills 3 --out G/logs/動画/家の場面2_FloorPlan1_dinov2_種0.mp4
python -m core_v2.tools.thor_watch --seconds 20 --scene FloorPlan201 --eye dinov2 --from 段5_目_種0_2 --stills 3 --out G/logs/動画/家の場面2_FloorPlan201_dinov2_種0.mp4
```

**返してほしいもの**（部屋ごとに）：
- 始めの行「[家の動画] 座る場所（auto）：…」（選んだ場所・向き・目に入る名前の物の数と語・点・始めの場所なら何個だったか・候補の数）
- 最後の3行（動画…／親：持ってきた（follow・bringin、bringin を見た・見なかった）・名づけ（follow・bringin）・語ごと／見た時間：回数・平均・真ん中・最長・1秒以上・語ごと）
- 動画 家の場面2_<部屋>_dinov2_種0.mp4 と静止画3枚（_1〜_3.png）を、依頼16 と同じ共有フォルダへ。_まとめ.json（"placement" と "parent" の欄）と
  events_<部屋>_seed0.csv（place・bring・look_held・say の行）は返事に添える
- 動画を見たひとこと：選んだ場所から物が見えているか（冷蔵庫・壁の目の前でないか）、親が持ってきた物を太郎が見たか
- `thor_fail` が0でなければ events_*.csv の teleport_fail・camera_fail の行。Teleport が通らなかったときは「座る場所」の行が start のままになるので、
  エラーの文（_まとめ.json の placement.error と events の place の行に出る）と、端末の最後の20行を返す
