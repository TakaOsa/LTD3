import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class Walker2dShortHighEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, screenshot=True):
        mujoco_env.MujocoEnv.__init__(self, "walker2d_shorthigh.xml", 4)
        utils.EzPickle.__init__(self)
        self.screenshot = screenshot

    def _step(self, a):
        posbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]
        alive_bonus = 1.0
        # reward = - (4 - ((posafter - posbefore) / self.dt))**2 + 16
        reward = np.minimum((posafter - posbefore) / self.dt, 2)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.3 and
                    ang > -1.2 and ang < 1.2)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.lookat[2] += .8

        self.viewer.cam.distance = self.model.stat.extent * 0.5

        if self.screenshot:
            self.viewer.cam.trackbodyid = -1
            self.viewer.cam.lookat[2] = .8
            self.viewer.cam.lookat[0] = 2 #3

        self.viewer.cam.elevation = -20
