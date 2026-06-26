"""
報酬の合成 — 各本能からの報酬を統合する
"""


def compute_total_reward(r_imit, r_pred, r_social, r_habit, weights):
    """
    R = w_imit * r_imit + w_pred * r_pred + w_social * r_social + r_habit
    """
    R = (weights["w_imit"] * r_imit
         + weights["w_pred"] * r_pred
         + weights["w_social"] * r_social
         + r_habit)
    return max(0.0, R)
