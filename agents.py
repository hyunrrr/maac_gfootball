from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from misc import hard_update, gumbel_softmax, onehot_from_logits
from policies import DiscretePolicy


class AttentionAgent(object):

    def __init__(self, num_in_pol, num_out_pol, hidden_dim=64,
                 lr=0.01, onehot_dim=0):

        self.policy = DiscretePolicy(num_in_pol, num_out_pol,
                                     hidden_dim=hidden_dim,
                                     onehot_dim=onehot_dim)
        self.target_policy = DiscretePolicy(num_in_pol, num_out_pol,
                                            hidden_dim=hidden_dim,
                                            onehot_dim=onehot_dim)

        hard_update(self.target_policy, self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

    def step(self, obs, explore=False):

        return self.policy(obs, sample=explore)

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])

