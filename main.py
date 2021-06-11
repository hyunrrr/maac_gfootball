import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from make_env import make_soccer_env
from buffer import ReplayBuffer
from env_wrappers import SubprocVecEnv, DummyVecEnv
from AttentionSAC import AttentionSAC



def make_parallel_env(num_agents, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = make_soccer_env(num_agents)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(run_num)
    np.random.seed(run_num)

    # make parallel env_wrapper
    env = make_parallel_env(config.num_agents, config.n_rollout_threads, run_num)


    # make AttentionSAC algorithm
    maac = AttentionSAC.init_from_env(env,
                                        tau=config.tau,
                                        pi_lr=config.pi_lr,
                                        q_lr=config.q_lr,
                                        gamma=config.gamma,
                                        pol_hidden_dim=config.pol_hidden_dim,
                                        critic_hidden_dim=config.critic_hidden_dim,
                                        attend_heads=config.attend_heads,
                                        reward_scale=config.reward_scale)


    obsp_vec = [env.observation_space.shape[1] for _ in range(env.observation_space.shape[0])]
    acsp_vec = env.action_space.nvec


    # make replay buffer
    replay_buffer = ReplayBuffer(config.buffer_length, maac.num_agents,
                                 obsp_vec,
                                 acsp_vec)

    t = 0
    for ep_i in range(0, config.num_episodes, config.n_rollout_threads):
        print('Episodes %i-%i of %i' % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.num_episodes))
        
        obs = env.reset()
        maac.prep_rollouts(device='cpu')
        
        

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch variable
            torch_obs = [Variable(torch.Tensor(obs[:, i]), requires_grad=False)
                         for i in range(maac.num_agents)]
            
            
            torch_agent_actions = maac.step(torch_obs, explore=True)

            
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[np.argmax(ac[i]) for ac in agent_actions] for i in range(config.n_rollout_threads)]


            n_obs, rews, done, info = env.step(actions)

            dones = np.array([[done[i] for _ in range(config.num_agents)] for i in range(config.n_rollout_threads)])

            replay_buffer.push(obs, agent_actions, rews, n_obs, dones)


            if (len(replay_buffer) >= config.batch_size and
                    (t % config.steps_per_update) < config.n_rollout_threads):
                if config.use_gpu:
                    maac.prep_training(device='gpu')
                else:
                    maac.prep_training(device='cpu')
                for u_i in range(config.num_updates):
                    sample = replay_buffer.sample(config.batch_size,
                                                  to_gpu=config.use_gpu)

                    maac.update_critic(sample, logger=logger)
                    maac.update_policies(sample, logger=logger)
                    maac.update_all_targets()
                maac.prep_rollouts(device='cpu')

        ep_rews = replay_buffer.get_average_rewards(
                config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i,
                              a_ep_rew * config.episode_length, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            maac.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maac.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maac.save(run_dir / 'model.pt')

    maac.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close() 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument('--buffer_length', default=int(1e6), type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--n_rollout_threads', default=16, type=int)
    parser.add_argument('--num_episodes', default=25000, type=int)
    parser.add_argument('--episode_length', default=125, type=int)
    parser.add_argument('--steps_per_update', default=125, type=int)
    parser.add_argument('--num_updates', default=4, type=int)
    parser.add_argument('--num_agents', default=3, type=int)
    parser.add_argument('--batch_size', default=1024*16, type=int)
    parser.add_argument('--save_interval', default=1000, type=int)
    parser.add_argument('--pol_hidden_dim', default=128, type=int)
    parser.add_argument('--critic_hidden_dim', default=128, type=int)
    parser.add_argument('--attend_heads', default=4, type=int)
    parser.add_argument('--pi_lr', default=0.001, type=float)
    parser.add_argument('--q_lr', default=0.001, type=float)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.001, type=float)
    parser.add_argument('--reward_scale', default=100., type=float)
    parser.add_argument('--use_gpu', action='store_true')

    config = parser.parse_args()

    if torch.cuda.is_available():
        config.use_gpu = True

    run(config)






    