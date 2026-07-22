"""皮質脊髄路（corticospinal tract）相当の薄い配線層。

【解剖学的位置づけ、2026-07-23】皮質脊髄路は運動野が計算した指令を脊髄へ"運ぶ"軸索の束で、
それ自体は計算をしない（配線）。太郎では、運動野（taro_brain_motor.pyのmotor_head等が計算する
mean）と脊髄CPG（spinal_cord/cpg.pyのCPGが生成するgen_out）を、重みw_meanで混合する処理が
これに対応する。新生児期は皮質脊髄路が未髄鞘化で精密制御の証拠がないため w_mean は小さく
（＝CPG優位）、生後2-5ヶ月にかけて皮質脊髄路が成熟するにつれ w_mean を上げていく
（そわそわ運動への移行、E/docs/研究日誌.md 続き16）想定。今は固定値（`doc/やることリスト.md`
項目9の発達スケジュール化は将来課題）。
"""
import torch


def project(mean, gen_out, w_mean):
    """運動野の出力(mean)と脊髄CPGの出力(gen_out)を、w_meanで混合する。
    w_mean=1.0で従来（CPGを完全に無視、運動野のみ）＝1バイト差なし。
    w_mean=0.0でCPGのみ（皮質脊髄路が未成熟＝新生児のwrithing GM相当）。"""
    return (1.0 - w_mean) * gen_out + w_mean * mean
