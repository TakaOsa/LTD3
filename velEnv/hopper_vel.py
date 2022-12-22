import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HopperVelEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, screenshot=False):
        mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)
        self.screenshot = screenshot

    def _step(self, a):
        posbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]
        alive_bonus = 1.0
        # reward = (posafter - posbefore) / self.dt
        reward = np.minimum((posafter - posbefore) / self.dt, 1)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            np.clip(self.model.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.distance = self.model.stat.extent * 0.9
        self.viewer.cam.elevation = -20

        if self.screenshot:
            self.viewer.cam.trackbodyid = -1
            self.viewer.cam.lookat[2] = .8
            self.viewer.cam.lookat[0] = 2.
            self.viewer.cam.elevation = -20
        # self.viewer.cam.elevation = 0
