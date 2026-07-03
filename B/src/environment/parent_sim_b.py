"""
親シミュレータ（目標B用）— イベント駆動で太郎を育てる

モデルAの親：毎ターン話しかけるだけ。
モデルBの親：在/不在があり、泣いたら来て、世話をしながら言葉を添える。

スケジュールは config/schedule.yaml で実験ごとに変更可能。
"""

import sys
import os
import random
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.core_b import TaroEnvironmentB


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 【人間模倣・B2-14】発達に合わせて授乳の量・回数・時間割の度合いを変える。
# 人間：0-6ヶ月はほぼ需要ベース（空腹で飲む・少量・頻回）、6ヶ月から離乳で時間割が
# 増え、18ヶ月で3食＋間食2回へ。量は増え回数は減る（CDC/AAP/Solid Starts, 参考文献§9）。
def _dev_months(sim_seconds):
    return sim_seconds / 2592000.0   # 1ヶ月=30日

def _age_feed_amount(sim_seconds):
    """1回の授乳量。新生児は胃が小さく少量、成長で増える（0.30→0.70で頭打ち）。
    少量だと血糖が満タンまで届かず早く空腹に戻る＝頻回、多量だと持つ＝少回、と
    回数も年齢で自然に変わる。"""
    m = _dev_months(sim_seconds)
    return 0.30 + min(1.0, m / 12.0) * 0.40


class ParentSchedule:
    """
    親の在/不在スケジュール。schedule.yamlから読み込む。
    """

    def __init__(self, schedule_path=None):
        if schedule_path is None:
            schedule_path = os.path.join(_project_root(), "config", "schedule.yaml")

        with open(schedule_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        p = cfg.get("presence", {})
        self.presence_prob = p.get("prob", 0.7)
        self.check_interval = p.get("check_interval", 100)

        r = cfg.get("respond", {})
        self.respond_prob = r.get("prob", 0.9)
        self.respond_prob_absent = r.get("prob_absent", 0.1)
        self.respond_delay_min = r.get("delay_min", 1)
        self.respond_delay_max = r.get("delay_max", 30)
        self.respond_delay_max_absent = r.get("delay_max_absent", 90)

        fd = cfg.get("feeding", {})
        self.feed_amount = fd.get("amount", 0.6)

        bb = cfg.get("babble", {})
        self.babble_interval = bb.get("interval", 45)
        self.babble_arousal_threshold = bb.get("arousal_threshold", 0.3)

        br = cfg.get("babble_response", {})
        self.babble_response_prob = br.get("prob", 0.4)

        ml = cfg.get("meals", {})
        self.meals_enabled = ml.get("enabled", False)
        # B2-14：食事時刻は月齢から動的に決める（年齢graded）。日中の時間帯だけ使う。
        self.meal_day_start = ml.get("day_start", 25200)   # 7:00
        self.meal_day_end = ml.get("day_end", 75600)       # 21:00
        self._last_meal_served = -1                        # 直近に出した時間割授乳の絶対秒

        sp = cfg.get("speech", {})
        self.speak_with_care = sp.get("enabled", True)
        self.words = sp.get("words", {"feed": "まんま", "sleep": "ねんね", "comfort": "だっこ"})
        # B2-17：世話ごとの「枠つき」言い回し。芯の語を含みつつ枠が変わる（繰り返し＋変化）。
        self.templates = sp.get("templates", {})

        sim = cfg.get("simulation", {})
        self.max_seconds = sim.get("max_seconds", 3600)
        self.log_interval = sim.get("log_interval", 300)

        self.present = True
        self.last_feed_time = 0
        self._pending_respond = None

    def update_presence(self, sim_seconds):
        if sim_seconds % self.check_interval == 0:
            self.present = random.random() < self.presence_prob

    def _schedule_response(self, sim_seconds):
        """
        親が何らかの合図（泣き・要求語）に気づいて反応するかを判定する。

        B2-1：泣きだけでなく「空腹時にまんまに近い発声をした」場合にも
        同じ経路で親に気づかせる（Skinnerのmand理論：欠乏状態でのみ
        要求語が結果と結びつく。合図の種類が泣きか発声かは親の気づき方
        としては同じ確率過程でよい）。
        """
        if not self.present:
            if random.random() < self.respond_prob_absent:
                delay = random.randint(self.respond_delay_max, self.respond_delay_max_absent)
                self._pending_respond = sim_seconds + delay
                self.present = True
            return False

        if random.random() < self.respond_prob:
            delay = random.randint(self.respond_delay_min, self.respond_delay_max)
            self._pending_respond = sim_seconds + delay
            return True
        return False

    def on_cry(self, sim_seconds):
        return self._schedule_response(sim_seconds)

    def on_word_request(self, sim_seconds):
        return self._schedule_response(sim_seconds)

    def should_respond_now(self, sim_seconds):
        if self._pending_respond is not None and sim_seconds >= self._pending_respond:
            self._pending_respond = None
            return True
        return False

    def choose_word(self, care_type):
        if not self.speak_with_care:
            return None
        return self.words.get(care_type, None)

    def choose_template(self, care_type, months=None):
        """B2-17：親が実際に言う言い回しを1つ選ぶ。テンプレがあれば抽選、無ければ芯の語。
        B2-18（cを軽く）：月齢で"解禁数"が増える。新生児は短く単純な言い回しだけ→成長で
        長く多彩に。※新しい語は増えない（視覚が無く物の名前を接地できないため）。"""
        if not self.speak_with_care:
            return None
        tmpls = self.templates.get(care_type)
        if tmpls:
            ordered = sorted(tmpls, key=len)                 # 短い（単純）ものを先に
            if months is None:
                k = len(ordered)
            else:
                k = min(len(ordered), 1 + int(months // 3))  # 3か月ごとに1つ解禁
            return random.choice(ordered[:k])
        return self.words.get(care_type, None)

    def meals_for_age(self, months):
        """月齢に応じた「1日の時間割授乳の時刻」（秒, 日中に均等配置）。
        0-5ヶ月＝0回（ほぼ完全な需要ベース）、6ヶ月から離乳で増え、18ヶ月で5回
        （3食＋間食2回）へ。人間の需要→時間割の移行を模倣。"""
        if months < 5:    n = 0
        elif months < 7:  n = 1
        elif months < 9:  n = 2
        elif months < 12: n = 3
        elif months < 18: n = 4
        else:             n = 5
        if n <= 0:
            return []
        s, e = self.meal_day_start, self.meal_day_end
        if n == 1:
            return [(s + e) // 2]
        step = (e - s) // (n - 1)
        return [s + i * step for i in range(n)]

    def meal_due(self, sim_seconds):
        """時間割授乳（B2-14・年齢graded）：今日の食事時刻を過ぎたら1回だけTrue。
        月齢が上がると食事時刻が増える。寝て逃した食事は食べない（最新の1回のみ）。"""
        if not self.meals_enabled:
            return False
        times = self.meals_for_age(_dev_months(sim_seconds))
        if not times:
            return False
        day, tod = divmod(sim_seconds, 86400)
        passed = [t for t in times if t <= tod]
        if not passed:
            return False
        latest = day * 86400 + max(passed)
        if latest > self._last_meal_served:
            self._last_meal_served = latest
            return True
        return False


# replayViewer用：イベント種別ごとに「発火する部品」と「情報の流れ」を対応づける。
# （抽象表示。太郎の内部を1本1本計測するのではなく、イベントの意味に基づく模式的な発火）
TRACE_MAP = {
    "babble":        {"modules": ["locus", "cortex", "vocal"],
                      "flows": [["locus", "cortex"], ["cortex", "vocal"]]},
    "babble_response": {"modules": ["cortex", "vocal"],
                      "flows": [["cortex", "vocal"]]},
    "word_request":  {"modules": ["stomach", "insula", "cortex", "vocal"],
                      "flows": [["stomach", "insula"], ["insula", "cortex"], ["cortex", "vocal"]]},
    "feed":          {"modules": ["stomach", "insula", "cortex", "critic"],
                      "flows": [["cortex", "insula"], ["insula", "critic"], ["stomach", "insula"]]},
    "comfort":       {"modules": ["cortex", "insula"], "flows": [["cortex", "insula"]]},
    "cry":           {"modules": ["stomach", "insula", "cortex", "lungs"],
                      "flows": [["stomach", "insula"], ["insula", "cortex"], ["cortex", "lungs"]]},
    "sleep":         {"modules": ["hippocampus", "cortex"], "flows": [["hippocampus", "cortex"]]},
    "sleep_word":    {"modules": ["cortex", "insula"], "flows": [["cortex", "insula"]]},
    "excrete":       {"modules": ["stomach", "insula"], "flows": [["stomach", "insula"]]},
}


def run_simulation_b(max_sim_seconds=None, verbose=True, run_name=None,
                     schedule_path=None, trace_path=None):
    """
    B用シミュレーション。イベント駆動で太郎を育てる。

    trace_path: 指定すると replayViewer 用の trace.jsonl を書き出す（オプトイン）。
    """
    rn = run_name or "B_sim"
    env = TaroEnvironmentB(run_name=rn)
    schedule = ParentSchedule(schedule_path=schedule_path)

    # 【2026-07-02】既定でreplayViewer用ログを replayViewer/data/<run名>/ に出力する。
    # trace_path=False を渡せば無効化できる。
    if trace_path is None:
        _data = os.path.join(_project_root(), "..", "replayViewer", "data", rn)
        os.makedirs(_data, exist_ok=True)
        trace_path = os.path.join(_data, "trace.jsonl")

    trace = None
    if trace_path:
        from trace_logger import TraceLogger
        trace = TraceLogger(trace_path)
        env.set_trace(trace)

    if max_sim_seconds is None:
        max_sim_seconds = schedule.max_seconds

    # 長い走行でトレースが巨大化しないよう、頻出イベント（喃語・喃語反応・泣き・あやし）は
    # 期間に応じて間引いて記録する。授乳・要求語・睡眠など重要イベントは常に全部記録。
    _days = max(1.0, max_sim_seconds / 86400.0)
    sample_n = max(1, round(_days / 2))
    _tr_ct = {}
    _FREQUENT = {"babble", "babble_response", "cry", "comfort"}

    def tr(kind, utter="", say=""):
        if trace is None:
            return
        m = TRACE_MAP.get(kind)
        if not m:
            return
        if kind in _FREQUENT:
            _tr_ct[kind] = _tr_ct.get(kind, 0) + 1
            if _tr_ct[kind] % sample_n != 0:
                return
        env.trace_event(sim_seconds, kind, m["modules"], m["flows"], utter, say=say)

    cry_count = 0
    feed_count = 0
    meal_count = 0          # 時間割授乳の回数（B2-12）
    meal_low_hunger = 0     # そのうち満腹寄り(hunger<0.5)で行われた回数＝語と空腹の脱相関
    speak_count = 0
    babble_count = 0
    request_count = 0
    consolidate_count = 0
    sim_seconds = 0
    last_babble_time = -schedule.babble_interval
    prev_sleeping = False

    # 理解の配線メーター（replayViewer用）：月イチで「聞いた語→食べ物予期」を測って記録する。
    # 満腹(0.1)で測るのは、空腹だと語によらず予期が上がるため（語の寄与＝理解は満腹時に出る）。
    # B2-20：3語（まんま/ねんね/だっこ）を測り、「まんまだけが食べ物予期を上げ、ねんね・だっこは
    # 上げない」＝3語が別物として区別されているか（混線してないか＝意味）を可視化する。
    _COMP_WORDS = ["まんま", "ねんね", "だっこ"]
    _comp_interval = 2592000     # 30日ごと
    _last_comp = -_comp_interval

    if verbose:
        print(f"=== 目標B シミュレーション開始 ===")
        print(f"最大: {max_sim_seconds}秒 ({max_sim_seconds//60}分)")
        print(f"親在確率: {schedule.presence_prob} 反応確率: {schedule.respond_prob}")
        print(f"食事量: {schedule.feed_amount}（泣いて空腹のときのみ授乳）")
        print(f"喃語間隔: {schedule.babble_interval}秒 つらさ閾値: {schedule.babble_arousal_threshold}")
        print()

    sleep_count = 0
    was_crying = False
    was_drowsy = False

    def fmt_time(s):
        """秒数を 時:分:秒 に変換する。"""
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:02d}"

    while sim_seconds < max_sim_seconds:
        currently_sleeping = env.internal_state.is_sleeping()

        # 睡眠移行：海馬リプレイ → 大脳皮質へ定着
        if currently_sleeping and not prev_sleeping:
            consol = env.consolidate()
            consolidate_count += consol["consolidated"]
            env.logger.log_event(sim_seconds, "sleep_start",
                                  consolidated=consol["consolidated"],
                                  p_loss=round(consol["p_loss"], 4))
            tr("sleep")
            if verbose and consol["consolidated"] > 0:
                print(f"  t={sim_seconds:5d}s ({fmt_time(sim_seconds)}) | "
                      f"海馬→皮質: {consol['consolidated']}件定着 "
                      f"loss={consol['p_loss']:.4f}")

        prev_sleeping = currently_sleeping

        # 寝ている間は時間を飛ばす
        if currently_sleeping:
            skip = env.internal_state._sleep_remaining
            env.tick_body(elapsed_seconds=skip, sim_seconds=sim_seconds + skip)
            sim_seconds += skip
            sleep_count += 1
            env.logger.log_event(sim_seconds, "sleep_end", duration=skip,
                                  hunger=round(env.internal_state.hunger, 4))
            if verbose:
                print(f"  t={sim_seconds:5d}s ({fmt_time(sim_seconds)}) | "
                      f"起きた（{skip//60}分寝た）| "
                      f"hunger={env.internal_state.hunger:.2f}")
            continue

        # うとうと中は時間を飛ばす
        if env.internal_state.is_drowsy():
            # B2-16：入眠時に親が「ねんね」と言う（合図が先→その後に眠りが来る）。眠りは
            # 太郎自身がやることなので親は"寝かしつける＝声をかける"だけ。満腹予期は付けない。
            sleep_say = schedule.choose_template("sleep", _dev_months(sim_seconds))  # B2-17/18
            if sleep_say and schedule.present:
                sr = env.step(sleep_say, r_social=0.5, satiety_target=0.0)
                speak_count += 1
                env.logger.log_event(sim_seconds, "sleep_word", say=sleep_say,
                                     sleepiness=round(env.internal_state.sleepiness, 4))
                tr("sleep_word", sr["taro"], say=sleep_say)
            if not was_drowsy and verbose:
                print(f"  t={sim_seconds:5d}s ({fmt_time(sim_seconds)}) | うとうと...")
            was_drowsy = True
            skip = env.internal_state._drowsy_remaining
            env.tick_body(elapsed_seconds=skip, sim_seconds=sim_seconds + skip)
            sim_seconds += skip
            was_drowsy = False
            continue

        env.tick_body(elapsed_seconds=1, sim_seconds=sim_seconds + 1)
        sim_seconds += 1
        schedule.update_presence(sim_seconds)

        # B2-19：排泄が起きたら記録（おむつが汚れて不快が上がる原因イベント）
        if env.internal_state._just_excreted:
            env.internal_state._just_excreted = False
            env.logger.log_event(sim_seconds, "excrete",
                                  discomfort=round(env.internal_state.discomfort, 4))
            tr("excrete")

        # 泣きの検出（泣き始めたときだけ通知）
        crying_now, intensity = env.check_cry()
        if crying_now and not was_crying:
            cry_count += 1
            env.logger.log_event(sim_seconds, "cry_start",
                                  intensity=round(intensity, 4),
                                  hunger=round(env.internal_state.hunger, 4),
                                  arousal=round(env.internal_state.get_arousal(), 4),
                                  parent_present=schedule.present)
            tr("cry")
            if verbose and (cry_count <= 20 or cry_count % 50 == 0):
                print(f"  t={sim_seconds:5d}s ({fmt_time(sim_seconds)}) | "
                      f"泣き始めた（強さ{intensity:.2f}）| "
                      f"hunger={env.internal_state.hunger:.2f} "
                      f"つらさ={env.internal_state.get_arousal():.2f} "
                      f"親{'在' if schedule.present else '不在'}")
            schedule.on_cry(sim_seconds)
        elif was_crying and not crying_now:
            env.logger.log_event(sim_seconds, "cry_end",
                                  hunger=round(env.internal_state.hunger, 4))
        was_crying = crying_now

        # 時間割授乳（B2-14・年齢graded）：月齢に応じた食事時刻に、空腹に関わらず「まんま」と
        # 言って授乳。0-5ヶ月は0回（ほぼ需要ベース）で、成長とともに増える（離乳の移行）。
        # 満腹寄りでもまんまを聞く機会になり語と空腹の相関を緩める。授乳量も年齢で変わる。
        # 理解の配線メーター：月イチで「聞いた語→食べ物予期」を測ってトレースに記録。
        # 測定が学習の乱数列を乱さないよう、プローブ前後でRNGを保存/復元する（非侵襲測定）。
        if trace is not None and sim_seconds - _last_comp >= _comp_interval:
            _last_comp = sim_seconds
            _rt, _rp = torch.get_rng_state(), random.getstate()
            cvals = {}
            for _w in _COMP_WORDS:
                _s = env.comprehension_probe(_w, 0.1, n_samples=1).get("satiety")
                cvals[_w] = round(_s, 4) if _s is not None else None
            torch.set_rng_state(_rt); random.setstate(_rp)
            trace.write_event({"type": "comprehension", "t": sim_seconds,
                               "mama": cvals["まんま"], "nenne": cvals["ねんね"],
                               "dakko": cvals["だっこ"]})

        # オンデマンド授乳（下の should_respond）が現実同様の主経路として残る。
        fa = _age_feed_amount(sim_seconds)
        if (schedule.meal_due(sim_seconds) and schedule.present
                and not env.internal_state.sleeping and not env.internal_state.drowsy):
            env.comfort("feed")
            meal_say = schedule.choose_template("feed", _dev_months(sim_seconds))  # B2-17/18：枠つき・月齢連動
            if meal_say:
                m = env.step(meal_say, r_social=0.5, satiety_target=1.0)
                env.feed(fa)
                feed_count += 1
                speak_count += 1
                meal_count += 1
                if m["hunger"] < 0.5:
                    meal_low_hunger += 1
                env.logger.log_turn(
                    m["turn"], sim_seconds, meal_say, m["taro"],
                    m["r_imit"], m["r_pred"], m["r_social"],
                    m["R"], m["delta"], m["p_loss"], m["a_loss"],
                    env.brain.temperature, context="meal", hunger=m["hunger"])
                env.logger.log_event(sim_seconds, "feed", amount=round(fa, 3),
                                     hunger_before=round(m["hunger"], 4), scheduled=True)
                tr("feed", m["taro"], say=meal_say)
                schedule.last_feed_time = sim_seconds

        if schedule.should_respond_now(sim_seconds):
            care_type = "feed" if env.internal_state.hunger > 0.5 else "comfort"
            env.comfort(care_type)

            say_text = schedule.choose_template(care_type, _dev_months(sim_seconds))  # B2-17/18
            if say_text:
                # B2-10：この発話の後に授乳が来るか（feedなら1.0）を満腹予期の教師にする。
                # まんまは授乳時・だっこは慰め時に聞くので「まんま→ごはん」を学べる。
                result = env.step(say_text, r_social=0.5,
                                  satiety_target=(1.0 if care_type == "feed" else 0.0))
            if care_type == "feed":
                env.feed(fa)                            # 発話の後に授乳（量は年齢graded）
                feed_count += 1
                speak_count += 1
                env.logger.log_turn(
                    result["turn"], sim_seconds, say_text, result["taro"],
                    result["r_imit"], result["r_pred"], result["r_social"],
                    result["R"], result["delta"], result["p_loss"], result["a_loss"],
                    env.brain.temperature,
                    context=care_type, hunger=result["hunger"],
                )
                env.logger.log_event(sim_seconds, "feed",
                                      amount=round(fa, 3),
                                      hunger_before=round(result["hunger"], 4))
                tr("feed", result["taro"], say=say_text)
                if verbose and (speak_count <= 30 or speak_count % 50 == 0):
                    print(f"  t={sim_seconds:5d}s ({fmt_time(sim_seconds)}) | "
                          f"親「{say_text}」→ 太郎「{result['taro']}」| "
                          f"模倣={result['r_imit']:.2f} hunger={result['hunger']:.2f}")
            else:
                env.logger.log_event(sim_seconds, "comfort",
                                      hunger=round(env.internal_state.hunger, 4))
                tr("comfort", say=say_text)

        # 自発的な喃語（穏やかな時間の練習）
        if (sim_seconds - last_babble_time >= schedule.babble_interval
                and env.internal_state.can_babble()):
            result = env.self_babble()
            babble_count += 1
            last_babble_time = sim_seconds
            tr("babble", result["taro"])
            if verbose and babble_count <= 5:
                print(f"  t={sim_seconds:5d}s ({fmt_time(sim_seconds)}) | "
                      f"喃語「{result['taro']}」| "
                      f"arousal={env.internal_state.get_arousal():.2f}")

            # 喃語への2つの反応経路を判定する。
            #   経路1（社会的）：親が言葉らしい発声に気づいて反応（Goldstein &
            #     Schwade）。空腹とは無関係だが、**言葉らしい発声ほど気づかれやすい**。
            #   経路2（要求語=mand, B2-1）：空腹時にまんま様発声をすると、泣いた
            #     ときと同じ経路で親に気づかせ実際に授乳させる。類似度そのものを
            #     気づかれる確率として連続的に使う（B2-3で閾値撤廃）。
            # B2-9修正：学習の更新は respond_to_babble の1か所に統一する（同じ
            # log_probsに2度backwardするとエラーのため）。どちらかが発火したら
            # まとめて1回だけ更新し、mand発火時は hunger に比例した「ホッとする
            # 報酬」を加算して「空腹→まんま」を配線する。物理的な授乳トリガー
            # （on_word_request）は従来どおり別に呼ぶ。
            hunger_now = env.internal_state.hunger
            # B2-16：欲求ごとに言い分ける。いま最も強い欲求を選び、その専用語に似た声を
            # 出したら（＝mand）その欲求を解消する。空腹だけでなく眠い・不快にも一般化。
            st = env.internal_state
            drives = [
                ("feed",    schedule.words.get("feed", "まんま"),    st.hunger),
                ("sleep",   schedule.words.get("sleep", "ねんね"),   st.sleepiness),
                ("comfort", schedule.words.get("comfort", "だっこ"), st.discomfort),
            ]
            dom_key, dom_word, dom_level = max(drives, key=lambda d: d[2])

            # B5-2修正：Goldstein & Schwade「言葉らしい発声ほど気づかれやすい」。
            # 従来は発声内容と無関係の固定確率で「気づくか」を決めていた（＝docstringの
            # 主旨と実装がズレていた）。最も近い語への音韻的類似度 word_sim（言葉らしさ、
            # まんま/ねんね/だっこ いずれかへの近さ＝母語の語らしさ）で気づく確率を高める。
            # ⚠️定数0.5＝語らしさに依らない基礎反応の割合（未検証・除去テスト可）。
            word_sim = 0.0
            if schedule.words:
                word_sim = max(env.word_similarity(result["tokens"], w)
                               for w in schedule.words.values())
            p_notice = schedule.babble_response_prob * (0.5 + word_sim)
            social_fired = (random.random() < p_notice) and bool(schedule.words)
            mand_fired = False
            similarity = 0.0
            if dom_level > 0.5 and dom_word:
                similarity = env.word_similarity(result["tokens"], dom_word)
                mand_fired = random.random() < similarity

            env.logger.log_babble(
                sim_seconds,
                result["taro"],
                env.internal_state.hunger,
                env.internal_state.get_arousal(),
                result["R"],
                result["r_pred"],
                result["r_home"],
                word_sim=word_sim,
                responded=(social_fired or mand_fired),
            )

            resp = None
            if (social_fired or mand_fired) and schedule.words:
                candidate_words = list(schedule.words.values())
                resp = env.respond_to_babble(
                    result["tokens"], result["log_probs"], candidate_words,
                    r_habit=result["r_habit"], hunger=dom_level,   # 報酬は解消される欲求の強さに比例
                    social=social_fired, mand=mand_fired,
                )

            if social_fired and resp:
                env.logger.log_event(
                    sim_seconds, "babble_response",
                    taro=result["taro"], target=resp["recognized_word"],
                    r_imit=round(resp["r_imit"], 4), R=round(resp["R"], 4),
                    delta=round(resp["delta"], 4),
                    hunger=round(hunger_now, 4),
                )
                tr("babble_response", result["taro"], say=resp["recognized_word"])

            if mand_fired:
                request_count += 1
                env.logger.log_event(sim_seconds, "word_request",
                                      taro=result["taro"], drive=dom_key, word=dom_word,
                                      similarity=round(similarity, 4),
                                      level=round(dom_level, 4),
                                      r_mand=round(resp["r_mand"], 4) if resp else 0.0)
                tr("word_request", result["taro"])
                # 欲求ごとに応える（親がその語を言ってから解消する）
                if dom_key == "feed":
                    schedule.on_word_request(sim_seconds)          # 既存：授乳を予約（feed時にまんま）
                elif dom_key == "sleep":
                    env.internal_state.drowsy = True               # 寝かしつけ→入眠へ（ねんね発話は入眠処理で）
                    env.internal_state._drowsy_remaining = random.randint(300, 600)
                elif dom_key == "comfort":
                    if schedule.present:
                        c_say = schedule.choose_template("comfort", _dev_months(sim_seconds)) or dom_word
                        cr = env.step(c_say, r_social=0.5, satiety_target=0.0)
                        tr("comfort", cr["taro"], say=c_say)
                    env.comfort("comfort")                          # 不快を下げる

        if verbose and sim_seconds % schedule.log_interval == 0:
            state = "寝" if env.internal_state.is_sleeping() else \
                    "うとうと" if env.internal_state.is_drowsy() else \
                    "泣き" if env.internal_state.is_crying() else "起きてる"
            feeding = " 授乳中" if env.stomach.is_feeding() else ""
            bg = env.blood_vessel.get_blood_glucose()
            print(f"  --- {fmt_time(sim_seconds)} | "
                  f"泣き{cry_count} 食事{feed_count} 発話{speak_count} 喃語{babble_count} 睡眠{sleep_count} | "
                  f"hunger={env.internal_state.hunger:.2f} 血糖={bg:.2f} "
                  f"つらさ={env.internal_state.get_arousal():.2f} | "
                  f"{state}{feeding}")

    if verbose:
        print(f"\n=== シミュレーション完了 ===")
        print(f"時間: {sim_seconds}秒 ({sim_seconds//60}分 = {sim_seconds/3600:.1f}時間)")
        print(f"泣き: {cry_count}回  食事: {feed_count}回  発話: {speak_count}回  "
              f"喃語: {babble_count}回  要求語: {request_count}回  睡眠: {sleep_count}回  定着: {consolidate_count}件")

    env.logger.close()
    trace_dir = None
    if trace:
        trace.close()
        trace_dir = os.path.dirname(trace_path)
        # 粒度別の概観＋マイルストーンをその場で作る（replayViewerがそのまま読める）
        try:
            _r = _project_root()
            if _r not in sys.path:
                sys.path.insert(0, _r)
            import build_trace_index
            build_trace_index.build(trace_path)
        except Exception as _e:
            if verbose:
                print(f"[trace] 概観の集約をスキップ: {_e}")
        if verbose:
            print(f"[trace] replayViewer用ログ: {trace_dir}")

    return {
        "sim_seconds": sim_seconds,
        "cry_count": cry_count,
        "feed_count": feed_count,
        "meal_count": meal_count,
        "meal_low_hunger": meal_low_hunger,
        "speak_count": speak_count,
        "babble_count": babble_count,
        "request_count": request_count,
        "sleep_count": sleep_count,
        "consolidate_count": consolidate_count,
        "trace_dir": trace_dir,
        "env": env,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="目標B シミュレーション")
    parser.add_argument("--seconds", type=int, default=None)
    parser.add_argument("--schedule", type=str, default=None, help="schedule.yamlのパス")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_simulation_b(
        max_sim_seconds=args.seconds,
        verbose=not args.quiet,
        schedule_path=args.schedule,
    )
