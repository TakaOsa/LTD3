from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym.envs.registration import register

register(
    id='Walker2dVel-v0',
    max_episode_steps=1000,
    entry_point='velEnv.walker2d_vel:Walker2dVelEnv'
)

register(
    id='Walker2dShort-v0',
    max_episode_steps=1000,
    entry_point='velEnv.walker2d_short:Walker2dShortEnv'
)

register(
    id='Walker2dShortOrange-v0',
    max_episode_steps=1000,
    entry_point='velEnv.walker2d_short_orange:Walker2dShortOrangeEnv'
)

register(
    id='Walker2dLong-v0',
    max_episode_steps=1000,
    entry_point='velEnv.walker2d_long:Walker2dLongEnv'
)

register(
    id='Walker2dLongOrange-v0',
    max_episode_steps=1000,
    entry_point='velEnv.walker2d_long_orange:Walker2dLongOrangeEnv'
)

register(
    id='Walker2dLowKnee-v0',
    max_episode_steps=1000,
    entry_point='velEnv.walker2d_lowknee:Walker2dLowKneeEnv'
)





register(
    id='HopperVel-v0',
    max_episode_steps=1000,
    entry_point='velEnv.hopper_vel:HopperVelEnv'
)

register(
    id='HopperShort-v0',
    max_episode_steps=1000,
    entry_point='velEnv.hopper_short:HopperShortEnv'
)

register(
    id='HopperShortShort-v0',
    max_episode_steps=1000,
    entry_point='velEnv.hopper_shortshort:HopperShortShortEnv'
)

register(
    id='HopperHighKnee-v0',
    max_episode_steps=1000,
    entry_point='velEnv.hopper_highknee:HopperHighkneeEnv'
)

register(
    id='HopperLowKnee-v0',
    max_episode_steps=1000,
    entry_point='velEnv.hopper_lowknee:HopperLowkneeEnv'
)

register(
    id='HopperLongHead-v0',
    max_episode_steps=1000,
    entry_point='velEnv.hopper_longhead:HopperLongHeadEnv'
)

register(
    id='HumanoidVel-v0',
    max_episode_steps=1000,
    entry_point='velEnv.humanoid_vel:HumanoidVelEnv'
)