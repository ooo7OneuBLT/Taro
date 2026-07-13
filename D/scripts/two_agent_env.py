"""
④a：2体MIMo環境（最小）。A（観測者）とB（相手）を1つのシムに置き、
両者を動かし、Aの観測＝Aの固有感覚(qpos/qvel)＋「Bへの触覚」(Aの部位ごとにBから受ける力)を返す。
MjSpec attach で2体化（prefix b_）。触覚系はMIMoの重い系でなく、接触力を部位に集約した最小版。
"""
import os
import sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "C"))
import paths
_SCENE = paths.SCENE


class TwoAgentMIMo:
    def __init__(self, sep=0.18):
        spec_a = mujoco.MjSpec.from_file(_SCENE)
        spec_b = mujoco.MjSpec.from_file(_SCENE)
        fr = spec_a.worldbody.add_frame(); fr.pos = [sep, 0, 0.4]
        fr.attach_body(spec_b.body('mimo_location'), 'b_', '')
        self.model = spec_a.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        # actuator を A/B に分ける（名前 b_ で判定）
        self.aid, self.bid = [], []
        for i in range(m.nu):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ''
            (self.bid if nm.startswith('b_') else self.aid).append(i)
        self.na = len(self.aid); self.nb = len(self.bid)
        # A の「体の部位」body id（触覚の集約先）。world/test_object/b_ を除く。
        self.a_bodies = []
        for b in range(m.nbody):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ''
            if nm and not nm.startswith('b_') and nm != 'world' and not nm.startswith('test_object'):
                self.a_bodies.append(b)
        self.a_body_index = {b: k for k, b in enumerate(self.a_bodies)}
        self.n_touch = len(self.a_bodies)
        # A の関節（qpos/qvel）を固有感覚に。A所属のjoint id。
        self.a_joints = []
        for j in range(m.njnt):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
            if not nm.startswith('b_'):
                self.a_joints.append(j)

    def _clsA(self, gi):
        nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[gi]) or ''
        if nm.startswith('b_'):
            return 'B'
        if nm == 'world' or nm.startswith('test_object'):
            return 'O'
        return 'A'

    def touch_of_B(self):
        """Aの各部位がBから受ける接触力の大きさ（n_touch次元）。"""
        m, d = self.model, self.data
        vec = np.zeros(self.n_touch)
        f6 = np.zeros(6)
        for c in range(d.ncon):
            con = d.contact[c]; c1, c2 = self._clsA(con.geom1), self._clsA(con.geom2)
            if {c1, c2} == {'A', 'B'}:
                mujoco.mj_contactForce(m, d, c, f6)
                ag = con.geom1 if c1 == 'A' else con.geom2
                bid = m.geom_bodyid[ag]
                if bid in self.a_body_index:
                    vec[self.a_body_index[bid]] += float(np.linalg.norm(f6[:3]))
        return vec

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
        return self.obs_A()

    def obs_A(self):
        m, d = self.model, self.data
        qpos = np.array([d.qpos[m.jnt_qposadr[j]] for j in self.a_joints if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE])
        return {"proprio_qpos": qpos, "touch_of_B": self.touch_of_B()}

    def step(self, a_action, b_action, K=1):
        m, d = self.model, self.data
        lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
        for i, ai in enumerate(self.aid):
            d.ctrl[ai] = np.clip(a_action[i], lo[ai], hi[ai])
        for i, bi in enumerate(self.bid):
            d.ctrl[bi] = np.clip(b_action[i], lo[bi], hi[bi])
        for _ in range(K):
            mujoco.mj_step(m, d)
        return self.obs_A()


if __name__ == "__main__":
    env = TwoAgentMIMo(sep=0.16)
    print("na=%d nb=%d  touch_dim=%d  proprio_dim(sample)=%d"
          % (env.na, env.nb, env.n_touch, len(env.reset()["proprio_qpos"])))
    m = env.model
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    # 検証：Bをランダムに大きく動かし、Aは静止気味。Aの触覚が立つ瞬間を探す。
    np.random.seed(0)
    best = None
    for t in range(400):
        a_act = np.zeros(env.na)
        b_full = lo + np.random.rand(m.nu) * (hi - lo)
        b_act = np.array([b_full[bi] for bi in env.bid])
        obs = env.step(a_act, b_act, K=5)
        tot = obs["touch_of_B"].sum()
        if tot > 0 and (best is None or tot > best[0]):
            idx = np.argsort(-obs["touch_of_B"])[:4]
            names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, env.a_bodies[i]) for i in idx if obs["touch_of_B"][i] > 0]
            best = (tot, names)
    if best:
        print("Aの触覚が立った最大の瞬間: 総和=%.1f  部位=%s" % (best[0], best[1]))
        print("=> Aは『Bにどの部位を触られたか』を観測として取れる（④a成立）")
    else:
        print("接触が起きなかった（配置/動きの調整が必要）")
