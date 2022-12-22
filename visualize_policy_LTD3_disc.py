import numpy as np
import gym
import torch

import argparse
import pprint as pp

import time
import copy

from PIL import Image
import PIL

from LTD3_agent import LTD3

import velEnv


def to_one_hot(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_one_hot = np.zeros((y.shape[0], num_columns))
    y_one_hot[range(y.shape[0]), y] = 1.0

    return y_one_hot

def evaluate_greedy(env_test, agent, args, state_dim, latent_cont_dim, latent_disc_dim, test_num = 5 ):

    latent_dim = latent_cont_dim + latent_disc_dim
    z_set = None
    if latent_cont_dim == 0:
        z_disc = np.arange(test_num)
        z_set = to_one_hot(z_disc, latent_disc_dim)
    else:
        z_set = np.zeros((int(test_num), 2))

    for i in range(test_num):
        z = np.zeros((1, latent_dim))
        z[0, :] = z_set[i, :]

        state_test = env_test.reset()
        return_epi_test = 0

        for t_test in range(int(args['max_episode_len'])):
            env_test.render()
            action_test = agent.select_action(np.reshape(state_test, (1, state_dim)),
                                              np.reshape(z, (1, latent_dim)))
            state_test2, reward_test, terminal_test, info_test = env_test.step(action_test)
            terminal_bool = float(terminal_test) if t_test < int(args['max_episode_len']) else 0

            state_test = state_test2
            return_epi_test = return_epi_test + reward_test
            if terminal_bool:
                break

        print('epi_len', t_test)
        print('z', np.around(z, decimals=2), end=' ')
        print('test {:d}, return: {:d}'.format(int(i), int(return_epi_test)))
        time.sleep(1)

    env_test.close()

def main(args):

    seed = args['random_seed']
    video_folder = args['monitor_dir'] + '_' + args['env'] + \
                   '_cont' + str(args['latent_cont_dim']) + '_disc' + str(args['latent_disc_dim']) + '_seed' + str(seed)

    try:
        import pathlib
        pathlib.Path(video_folder).mkdir(parents=True, exist_ok=True)

    except:
        print("A result directory does not exist and cannot be created. The trial results are not saved")

    env = gym.make(args['env'])
    env.seed(int(args['trial_idx']))


    env_test = gym.make(args['env'])
    env_test.seed(int(args['trial_idx']))
    torch.manual_seed(int(args['trial_idx']))

    print('action_space.shape', env_test.action_space.shape)
    print('observation_space.shape', env_test.observation_space.shape)
    action_bound = float(env_test.action_space.high[0])

    assert (env_test.action_space.high[0] == -env_test.action_space.low[0])

    state_dim = env_test.observation_space.shape[0]
    action_dim = env_test.action_space.shape[0]
    latent_cont_dim = int(args['latent_cont_dim'])
    latent_disc_dim = int(args['latent_disc_dim'])


    agent = LTD3(state_dim=state_dim, action_dim=action_dim, latent_cont_dim=latent_cont_dim,
                 latent_disc_dim=latent_disc_dim,
                 max_action=action_bound, iw=args['IW'], info_freq=args['info_freq'])

    if args['save_video']:
        env = gym.wrappers.Monitor(env, video_folder,
                                    video_callable=lambda episode_id: episode_id % 1 == 0,force=True)

    agent.load_model(iter=61, seed=seed, env_name=args['env'], args = args)

    evaluate_greedy(env, agent, args, state_dim, latent_cont_dim, latent_disc_dim, test_num=latent_disc_dim)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments')

    # run parameters
    parser.add_argument('--env')
    parser.add_argument('--env-id', type=int, default=1)
    parser.add_argument('--random-seed', help='select from 1 to 5', default=2)

    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=1001) #50000
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)

    parser.add_argument('--result-file', help='file name for storing results from multiple trials',
                        default='./results/trials/ltd3/trials_ltd3_')
    parser.add_argument('--model-folder', help='folder name for storing models',
                        default='./models/ltd3_joint')
    parser.add_argument('--monitor-dir', help='directory for recording', default='video/ltd3')
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--trial-idx', help='index of trials', default=0)

    parser.add_argument("--optim-name", default='Adam')  # Frequency of delayed policy updates
    parser.add_argument("--latent-cont-dim", default=0, type=int)  # dimension of the continuous latent variable
    parser.add_argument("--latent-disc-dim", default=5, type=int)  # dimension of the discrete latent variable
    parser.add_argument("--IW", default='IW')  # whether to use co-teaching for info-max or not
    parser.add_argument("--info-freq", default=4, type=int)
    parser.add_argument("--iw-clip-ratio", default=0.3, type=float)
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--hidden", default=(256, 256))

    parser.add_argument("--save-screenshot", default=False, type=bool)
    parser.add_argument("--save-video", default=True, type=bool)

    parser.set_defaults(use_gym_monitor=False)
    parser.set_defaults(render_env=False)

    args_tmp = parser.parse_args()

    if args_tmp.env is None:
        env_dict = {1: 'Walker2dVel-v0',
                    2: 'HumanoidVel-v0',
                    3: 'HopperVel-v0'
        }
        args_tmp.env = env_dict[args_tmp.env_id]
    args = vars(args_tmp)

    pp.pprint(args)

    main(args)
