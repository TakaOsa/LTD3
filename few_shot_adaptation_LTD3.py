import numpy as np
import gym
import torch

import argparse
import pprint as pp

import time
import copy

from LTD3_agent import LTD3

import velEnv


def to_one_hot(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_one_hot = np.zeros((y.shape[0], num_columns))
    y_one_hot[range(y.shape[0]), y] = 1.0

    return y_one_hot

def evaluate_greedy(env_test, agent, args, state_dim, latent_cont_dim, latent_disc_dim, budget =25):
    latent_dim = latent_cont_dim + latent_disc_dim
    return_set = []
    z_set = None
    if latent_cont_dim == 0:
        z_disc = np.arange(budget)
        z_set = to_one_hot(z_disc, latent_disc_dim)
    else:
        z_set = np.zeros((int(budget), 2))

    z_max = z_set[0]
    ret_max = 0
    for i in range(budget):
        z = np.zeros((1, latent_dim))

        z_cont = None
        if not latent_cont_dim == 0:
            if i < 15:
                z_cont = np.random.uniform(-0.8, 0.8, size=(1, latent_cont_dim))
            else:
                z_cont = z_max[0, :latent_cont_dim] + np.random.uniform(-0.2, 0.2, size=(1, latent_cont_dim))
            if latent_disc_dim == 0:
                z = z_cont
            else:
                z_disc = np.random.randint(0, latent_disc_dim, 1)
                z_disc = to_one_hot(z_disc, latent_disc_dim)
                z = np.hstack((z_cont, z_disc))
        else:
            z[0, :] = z_set[i, :]


        state_test = env_test.reset()
        return_epi_test = 0

        for t_test in range(int(args['max_episode_len'])):
            if args['render']:
                env_test.render()
            action_test = agent.select_action(np.reshape(state_test, (1, state_dim)),
                                              np.reshape(z, (1, latent_dim)))
            state_test2, reward_test, terminal_test, info_test = env_test.step(action_test)
            terminal_bool = float(terminal_test) if t_test < int(args['max_episode_len']) else 0

            state_test = state_test2
            return_epi_test = return_epi_test + reward_test
            if terminal_bool:
                break
        return_set.append(return_epi_test)
        if return_epi_test > ret_max:
            ret_max = return_epi_test
            z_max = z

        print('epi_len', t_test)
        print('z', np.around(z, decimals=2), end=' ')
        print('test {:d}, return: {:d}'.format(int(i), int(return_epi_test)))
        time.sleep(1)

    return_set = []
    print('z_max', z_max)
    for i in range(5):
        state_test = env_test.reset()
        return_epi_test = 0
        for t_test in range(int(args['max_episode_len'])):
            if args['render']:
                env_test.render()
            action_test = agent.select_action(np.reshape(state_test, (1, state_dim)),
                                              np.reshape(z_max, (1, latent_dim)))
            state_test2, reward_test, terminal_test, info_test = env_test.step(action_test)
            terminal_bool = float(terminal_test) if t_test < int(args['max_episode_len']) else 0

            state_test = state_test2
            return_epi_test = return_epi_test + reward_test
            if terminal_bool:
                break
        return_set.append(return_epi_test)

    print('np.asarray(return_set)', np.asarray(return_set))
    # env_test.close()

    return np.asarray(return_set), z_set

def main(args):
    seed = args['random_seed']

    env = gym.make(args['env'])
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    action_bound = float(env.action_space.high[0])

    assert (env.action_space.high[0] == - env.action_space.low[0])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    latent_cont_dim = int(args['latent_cont_dim'])
    latent_disc_dim = int(args['latent_disc_dim'])

    train_env = None
    if args['env'] in [ 'Walker2dShort-v0' , 'Walker2dShortOrange-v0','Walker2dLowKnee-v0',
                        'Walker2dBigFootRed-v0','Walker2dBigFootOrange-v0', 'Walker2dBigFoot-v0',
                        'Walker2dLong-v0', 'Walker2dLongOrange-v0', 'Walker2dSmallFoot-v0']:
        train_env = 'Walker2dVel-v0'
    if args['env'] in ['HopperShort-v0', 'HopperShortShort-v0','HopperSmallFoot-v0','HopperHighKnee-v0', 'HopperLowKnee-v0',
                       'HopperLongHead-v0']:
        train_env = 'HopperVel-v0'

    agent = LTD3(state_dim=state_dim, action_dim=action_dim, latent_cont_dim=latent_cont_dim,
                 latent_disc_dim=latent_disc_dim,
                 max_action=action_bound, iw=args['IW'], info_freq=args['info_freq'])

    video_folder = args['monitor_dir'] + '/ltd3_cont' + str(int(args['latent_cont_dim'])) + '_disc' + str(
        int(args['latent_disc_dim'])) + \
                   'adaptation_from_' + train_env + '2' + args['env'] + '_k=' + str(args['budget'])

    if args['save_video']:
        try:
            import pathlib
            pathlib.Path(video_folder).mkdir(parents=True, exist_ok=True)

        except:
            print("A result directory does not exist and cannot be created. The trial results are not saved")

        env = gym.wrappers.Monitor(env, video_folder,
                                    video_callable=lambda episode_id: episode_id % 1 == 0,force=True)

    test_return = np.zeros((5,5))
    for seed in [1,2,3,4,5]:
        agent.load_model(iter=61, seed=seed, env_name=train_env, args=args)
        ret, z_epi = evaluate_greedy(env, agent, args, state_dim, latent_cont_dim, latent_disc_dim, budget=args['budget'])
        test_return[seed-1, :] = ret.flatten()
    print('mean_return', np.mean(test_return, axis=1))
    print('total mean return', np.mean(test_return))

    result_path = "./results/trials/few_shot"
    filename = result_path + '/ltd3_cont' + str(int(args['latent_cont_dim'])) + '_disc' + str(int(args['latent_disc_dim'])) +\
                        'adaptation_from_' +  train_env + '2' + args['env'] + '_k='+ str(args['budget']) +'.txt'

    try:
        import pathlib
        pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)
        np.savetxt(filename, test_return)
    except:
        print("A result directory does not exist and cannot be created. The trial results are not saved")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments')

    # run parameters
    parser.add_argument('--env')
    parser.add_argument('--env-id', type=int, default=1)

    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=1001) #50000
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1)

    parser.add_argument('--result-file', help='file name for storing results from multiple trials',
                        default='./results/trials/ltd3/trials_ltd3_')
    parser.add_argument('--model-folder', help='folder name for storing models',
                        default='./models/ltd3_joint')
    parser.add_argument('--monitor-dir', help='directory for recording', default='video/few-shot_ltd3')
    parser.add_argument('--render', help='render the gym env', default=True)

    parser.add_argument("--optim-name", default='Adam')  # Frequency of delayed policy updates
    parser.add_argument("--latent-cont-dim", default=0, type=int)  # dimension of the continuous latent variable
    parser.add_argument("--latent-disc-dim", default=5, type=int)  # dimension of the discrete latent variable
    parser.add_argument("--IW", default='IW')  # whether to use co-teaching for info-max or not
    parser.add_argument("--info-freq", default=4, type=int)
    parser.add_argument("--iw-clip-ratio", default=0.3, type=float)
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--hidden", default=(256, 256))

    parser.add_argument("--save-video", default=True, type=bool)
    parser.add_argument("--budget", default=25, type=int)

    parser.set_defaults(use_gym_monitor=False)
    parser.set_defaults(render_env=False)

    args_tmp = parser.parse_args()

    if args_tmp.env is None:
        if args_tmp.env is None:
            env_dict = {
                0: 'Walker2dShort-v0',
                1: 'Walker2dShortOrange-v0',
                2: 'Walker2dLowShort-v0',
                3: 'Walker2dShortHigh-v0',
            }
        args_tmp.env = env_dict[args_tmp.env_id]
    args = vars(args_tmp)

    pp.pprint(args)

    main(args)
