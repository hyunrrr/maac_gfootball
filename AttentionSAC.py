import torch
import torch.nn.functional as F
from torch.optim import Adam
from misc import soft_update, hard_update, enable_gradients, disable_gradients
from critics import AttentionCritic
from agents import AttentionAgent

MSELoss = torch.nn.MSELoss()


class AttentionSAC(object):

    def __init__(self, agent_init_params, sa_size,
                 gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                 reward_scale=10.,
                 pol_hidden_dim=128,
                 critic_hidden_dim=128, attend_heads=4,
                 **kwargs):
        
        self.num_agents = len(sa_size)
        
        self.agents = [AttentionAgent(lr=pi_lr,
                                      hidden_dim=pol_hidden_dim,
                                      **params)
                          for params in agent_init_params]
        
        self.critic =  AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
                                       attend_heads=attend_heads)

        self.target_critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
                                             attend_heads=attend_heads)

        
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr, weight_decay=1e-3)

        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale

        self.pol_dev = 'cpu'
        self.critic_dev = 'cpu'
        self.trgt_pol_dev = 'cpu'
        self.trgt_critic_dev = 'cpu'

        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]


    def step(self, observations, explore=False):
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents, observations)]


    def update_critic(self, sample, soft=True, logger=None, **kwargs):

        obs, acs, rews, next_obs, dones = sample

        next_acs = []
        next_log_pis = []
        for pi, ob in zip(self.target_policies, next_obs):
            curr_next_ac, curr_next_log_pi = pi(ob, return_log_pi=True)
            next_acs.append(curr_next_ac)
            next_log_pis.append(curr_next_log_pi)
        trgt_critic_in = list(zip(next_obs, next_acs))
        critic_in = list(zip(obs, acs))
        next_qs = self.target_critic(trgt_critic_in)
        critic_rets = self.critic(critic_in, regularize=True,
                                  logger=logger, niter=self.niter)

        q_loss = 0
        for a_i, nq, log_pi, (pq, regs) in zip(range(self.num_agents), next_qs, next_log_pis,
                                               critic_rets):

            target_q = (rews[a_i].view(-1, 1) + self.gamma*nq*(1-dones[a_i].view(-1, 1)))

            if soft:
                target_q -= log_pi / self.reward_scale
            q_loss += MSELoss(pq, target_q.detach())

            ### ??
            for reg in regs:
                q_loss += reg # regularizing attention

        q_loss.backward()
        ### ??
        self.critic.scale_shared_grads()
        ### ?? about arg2
        grad_norm = torch.nn.utils.clip_grad_norm(
                self.critic.parameters(), 10*self.num_agents)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        if logger is not None:
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
            logger.add_scalar('grad_norms/q', grad_norm, self.niter)
        self.niter += 1

    def update_policies(self, sample, soft=True, logger=None, **kwargs):
        obs, acs, rews, next_obs, dones = sample
        
        samp_acs = []
        all_probs = []
        all_log_pis = []
        all_pol_regs = []

        for a_i, pi, ob in zip(range(self.num_agents), self.policies, obs):
            curr_ac, probs, log_pi, pol_regs, ent = pi(
                    ob, return_all_probs=True, return_log_pi=True,
                    regularize=True, return_entropy=True)
            logger.add_scalar('agent%i/policy_entropy' % a_i, ent, self.niter)
            
            samp_acs.append(curr_ac)
            all_probs.append(probs)
            all_log_pis.append(log_pi)
            all_pol_regs.append(pol_regs)

        ### to policy update, not using act in buffer, but using pi(ob) again
        critic_in = list(zip(obs, samp_acs))
        critic_rets = self.critic(critic_in, return_all_q=True)
        for a_i, probs, log_pi, pol_regs, (q, all_q) in zip(range(self.num_agents), all_probs,
                                                            all_log_pis, all_pol_regs,
                                                            critic_rets):
            curr_agent = self.agents[a_i]
            v = (all_q * probs).sum(dim=1, keepdim=True)
            pol_target = q - v
            if soft:
                pol_loss = (log_pi * (log_pi / self.reward_scale - pol_target).detach()).mean()
            else:
                pol_loss = (log_pi * (-pol_target).detach()).mean()

            ### ??
            for reg in pol_regs:
                pol_loss += 1e-3 * reg # policy regularization

            # don't want critic to accumulate gradients from policy loss
            disable_gradients(self.critic)
            pol_loss.backward()
            enable_gradients(self.critic)

            ### ??? about arg2            
            grad_norm = torch.nn.utils.clip_grad_norm(
                    curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()

            if logger is not None:
                logger.add_scalar('agent%i/losses/pol_loss' % a_i, pol_loss, self.niter)
                logger.add_scalar('agent%i/grad_norms/pi' % a_i, grad_norm, self.niter)


    def update_all_targets(self):

        soft_update(self.target_critic, self.critic, self.tau)
        for a in self.agents:
            soft_update(a.target_policy, a.policy, self.tau)

    def prep_training(self, device='gpu'):
        self.critic.train()
        self.target_critic.train()
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()

        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
            

    def save(self, filename):
        self.prep_training(device='cpu') # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents],
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict()}}
        torch.save(save_dict, filename)



    @classmethod
    def init_from_env(cls, env, gamma=0.95, tau=0.01,
                      pi_lr=0.01, q_lr=0.01,
                      reward_scale=10.,
                      pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4,
                      **kwargs):
        
        agent_init_params = []
        sa_size = []

        acsp_vec = env.action_space.nvec
        obsp_vec = [ env.observation_space.shape[1] for _ in range(env.observation_space.shape[0])]

        for acsp, obsp in zip(acsp_vec, obsp_vec):
            agent_init_params.append({'num_in_pol': obsp,
                                      'num_out_pol': acsp})
            sa_size.append((obsp, acsp))

        init_dict = {'gamma': gamma, 'tau': tau,
                     'pi_lr': pi_lr, 'q_lr': q_lr,
                     'reward_scale': reward_scale,
                     'pol_hidden_dim': pol_hidden_dim,
                     'critic_hidden_dim': critic_hidden_dim,
                     'attend_heads': attend_heads,
                     'agent_init_params': agent_init_params,
                     'sa_size': sa_size}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance


    ### ??
    @classmethod
    def init_from_model(cls, filename, load_critic=False):

        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)

        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
        return instance













