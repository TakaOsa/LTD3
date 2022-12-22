import numpy as np
import gym

import argparse
import pprint as pp
from scipy import stats

from LTD3_agent import LTD3
import torch
from replay_buffer_latent import ReplayBufferLatent

# ===========================
#   Agent Training
# ===========================

def evaluate_greedy(env_test, agent, args, test_iter, test_n, state_dim, latent_cont_dim, latent_disc_dim):

    state_test = env_test.reset()
    return_epi_test = 0

    latent_dim = latent_cont_dim + latent_disc_dim
    z = None
    z_cont = None
    if not latent_cont_dim == 0:
        z_cont = np.random.uniform(-1, 1, size=(1, latent_cont_dim))
        if latent_disc_dim == 0:
            z = z_cont
    if not latent_disc_dim == 0:
        z_disc = np.random.randint(0, latent_disc_dim, 1)
        z_disc = to_one_hot(z_disc, latent_disc_dim)
        if latent_cont_dim == 0:
            z = z_disc
        else:
            z = np.hstack((z_cont, z_disc))

    for t_test in range(int(args['max_episode_len'])):
        action_test = agent.select_action(np.reshape(state_test, (1, state_dim)), np.reshape(z, (1, latent_dim)))
        state_test2, reward_test, terminal_test, info_test = env_test.step(action_test)
        state_test = state_test2
        return_epi_test = return_epi_test + reward_test
        if terminal_test:
            break

    print('z', np.around(z, decimals=2), end=' ')
    print('test_iter:{:d}, nn:{:d}, return_epi_test: {:d}'.format(int(test_iter), int(test_n),
                                                                      int(return_epi_test)))

    return return_epi_test

def to_one_hot(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_one_hot = np.zeros((y.shape[0], num_columns))
    y_one_hot[range(y.shape[0]), y] = 1.0

    return y_one_hot

def train(env, env_test, agent, args ):

    # Initialize replay memory
    total_step_cnt = 0
    epi_cnt = 0
    test_iter = 0
    return_test = np.zeros((np.ceil(int(args['total_step_num']) / int(args['eval_step_freq'])).astype('int') + 1))

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    latent_cont_dim = int(args['latent_cont_dim'])
    latent_disc_dim = int(args['latent_disc_dim'])
    latent_dim = latent_cont_dim + latent_disc_dim

    replay_buffer = ReplayBufferLatent(state_dim, action_dim, latent_dim)

    while total_step_cnt in range( int(args['total_step_num']) ):

        state = env.reset()
        ep_reward = 0
        T_end = False

        z = None
        z_cont = None
        if not latent_cont_dim==0:
            z_cont = np.random.uniform(-1, 1, size=(1, latent_cont_dim))
            if latent_disc_dim==0:
                z = z_cont
        if not latent_disc_dim==0:
            z_disc = np.random.randint(0, latent_disc_dim, 1)
            z_disc = to_one_hot(z_disc, latent_disc_dim)
            if latent_cont_dim == 0:
                z = z_disc
            else:
                z = np.hstack((z_cont, z_disc))

        for t in range(int(args['max_episode_len'])):

            # Select action randomly or according to policy
            if total_step_cnt < int(args['start_timesteps']):
                action = env.action_space.sample()
            else:
                action = (
                        agent.select_action(np.array(state), np.array(z))
                        + np.random.normal(0, max_action * float(args['expl_noise']), size=action_dim)
                        ).clip(-max_action, max_action)

            state2, reward, terminal, info = env.step(action)
            terminal_bool = float(terminal) if t < int(args['max_episode_len']) else 0

            # Store data in replay buffer
            replay_buffer.add(state, action, state2, reward, terminal_bool, z)

            # Train agent after collecting sufficient data
            if total_step_cnt >= int(args['start_timesteps']):
                for i in range(int(args['update_freq'])):
                    agent.train(replay_buffer, int(args['batch_size']))

            if t == int(args['max_episode_len']) - 1:
                T_end = True

            state = state2
            ep_reward += reward
            total_step_cnt += 1

            # Evaluate the deterministic policy
            if total_step_cnt >= test_iter * int(args['eval_step_freq']) or total_step_cnt == 1:
                print('total_step_cnt', total_step_cnt)
                print('evaluating the deterministic policy...')
                for test_n in range(int(args['test_num'])):
                    return_epi_test = evaluate_greedy(env_test, agent, args, test_iter, test_n, state_dim, latent_cont_dim, latent_disc_dim)

                    # Store the average of returns over the test episodes
                    return_test[test_iter] = return_test[test_iter] + return_epi_test / float(args['test_num'])

                print('return_test[{:d}] {:d}'.format(int(test_iter), int(return_test[test_iter])))
                test_iter += 1

            if total_step_cnt % int(args['model_save_freq'])==0:
                    agent.save_model(iter=test_iter, seed=int(args['trial_idx']), env_name=args['env'], args=args)


            if terminal or T_end:
                epi_cnt += 1
                print('z', np.around(z,decimals=2), end=' ')
                print('| Reward: {:d}| Episode: {:d} | Total step num: {:d} |'.format(int(ep_reward), epi_cnt, total_step_cnt ))

                break

    return return_test

def main(args):
    for ite in range(int(args['trial_num'])):
        print('Trial Number:', ite)
        np.random.seed(int(args['trial_idx']))

        env = gym.make(args['env'])
        env.seed(int(args['trial_idx']))

        env_test = gym.make(args['env'])
        env_test.seed(int(args['trial_idx']))
        torch.manual_seed(int(args['trial_idx']))

        print('action_space.shape', env.action_space.shape)
        print('observation_space.shape', env.observation_space.shape)
        action_bound = float(env.action_space.high[0])

        assert (env.action_space.high[0] == -env.action_space.low[0])

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        latent_cont_dim = int(args['latent_cont_dim'])
        latent_disc_dim = int(args['latent_disc_dim'])

        agent = LTD3(state_dim=state_dim, action_dim=action_dim, latent_cont_dim=latent_cont_dim, latent_disc_dim=latent_disc_dim,
                     max_action=action_bound,  iw=args['IW'], info_freq=args['info_freq'],
                     clip_ratio=args['iw_clip_ratio'], hidden=args['hidden'])

        step_R_i = train(env, env_test, agent, args)

        IW = None
        if args['IW'] == 'IW':
            IW = 'IW_' + str(args['iw_clip_ratio'])
        else:
            IW = args['IW']

        result_path = "./results/trials/ltd3_joint"
        result_filename = result_path + args['result_file'] + '_' + args['env'] +  args['optim_name'] + '_update_freq_' + str(int(args['update_freq'])) \
                          + '_cont' + str(int(args['latent_cont_dim'])) + '_disc' + str(int(args['latent_disc_dim']))  + '_' + IW \
                          + '_info-freq_' + str(int(args['info_freq']))  \
                          +  '_trial_idx_' + str(int(args['trial_idx'])) + '.txt'

        try:
            import pathlib
            pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)
            np.savetxt(result_filename, np.asarray(step_R_i))
        except:
            print("A result directory does not exist and cannot be created. The trial results are not saved")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments')

    # run parameters
    parser.add_argument('--env')
    parser.add_argument('--env-id', type=int, default=1)
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=1001) #50000
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--trial-num', help='number of trials', default=1)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--total-step-num', help='total number of time steps', default=1000000)
    parser.add_argument('--eval-step-freq', help='frequency of evaluating the policy', default=50000)
    parser.add_argument('--test-num', help='number of test episodes', default=10)
    parser.add_argument('--model-save-freq', help='frequency of evaluating the policy', default=1000000)

    parser.add_argument('--result-file', help='file name for storing results from multiple trials',
                        default='/trials_ltd3_')
    parser.add_argument('--model-folder', help='folder name for storing models',
                        default='./models/ltd3')
    parser.add_argument('--trial-idx', help='index of trials', default=0)
    parser.add_argument('--monitor-dir', help='directory for recording', default='results/trials/ltd3')

    parser.add_argument("--start_timesteps", default=1e4, type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--expl_noise", default=0.2, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--update_freq", default=1, type=int)  # Number of policy updates
    parser.add_argument("--optim-name", default='Adam')  # Frequency of delayed policy updates
    parser.add_argument("--latent-cont-dim", default=2, type=int)  # dimension of the continuous latent variable
    parser.add_argument("--latent-disc-dim", default=0, type=int)  # dimension of the discrete latent variable
    parser.add_argument("--info-freq", default=4, type=int)
    parser.add_argument("--IW", default='IW')
    parser.add_argument("--iw-clip-ratio", default=0.3, type=float)
    parser.add_argument("--hidden", default=(256, 256))

    parser.set_defaults(render_env=True)
    parser.set_defaults(use_gym_monitor=False)
    parser.set_defaults(change_seed=True)
    parser.set_defaults(overwrite_result=True)

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
