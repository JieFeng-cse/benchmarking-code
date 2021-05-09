import numpy as np
import math
import torch
from torch import nn as nn
import rlkit.torch.pytorch_util as ptu

from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.core import eval_np
from rlkit.torch.distributions import TanhNormal
from rlkit.torch.networks import Mlp


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class TanhGaussianPolicyV2(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            qf1,
            qf2,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim+2,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        self.qf1 = qf1
        self.qf2 = qf2
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX
        
        self.locations = np.mgrid[-10:6,-10:11].reshape(2,-1).T.astype('float32') * 0.03
        self.n_location = self.locations.shape[0]
        # self.n_tile = 5
        # self.locations_aug = np.tile(self.locations, (1,n_tile)) * 0.03
    
    @torch.no_grad()
    def get_action(self, obs_np, deterministic=False):
        threshold = 0.15
        obs_np = obs_np[:-2]
        obs_np = np.repeat(obs_np[None], self.n_location, axis=0)
        obs_np = np.concatenate([obs_np, self.locations], axis=1).astype('float32')
        if not deterministic:
            mean, log_std = self.get_mean(obs_np, deterministic=deterministic)
            mean = torch.from_numpy(mean).to(ptu.device)
            log_std = torch.from_numpy(log_std).to(ptu.device)
        else:
            mean = self.get_actions(obs_np, deterministic=deterministic)
            mean = torch.from_numpy(mean).to(ptu.device)
        
        obs = torch.from_numpy(obs_np).to(ptu.device)
        q1_pred = self.qf1(obs, mean).detach()
        q2_pred = self.qf2(obs, mean).detach()

        q = torch.min(torch.stack([q1_pred, q2_pred], dim=0), dim=0)[0]
        values, indices = torch.topk(q, math.ceil(threshold * self.n_location), dim=0)
        sampled_idx = torch.randint(high=math.ceil(threshold * self.n_location), size=(1,)).to(ptu.device)

        actual_idx = indices[sampled_idx]
        location = self.locations[actual_idx]

        mean = mean[actual_idx]
        if not deterministic:
            std = torch.exp(log_std[actual_idx])
            tanh_normal = TanhNormal(mean, std)
            delta = tanh_normal.rsample()
        else:
            delta = torch.tanh(mean)
        delta = delta.squeeze()
        delta = ptu.get_numpy(delta)
        
        action = np.concatenate((location, delta), axis=-1)
        return action, {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def get_mean(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[1:3]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """        
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )


class MakeDeterministic(nn.Module, Policy):
    def __init__(self, stochastic_policy):
        super().__init__()
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)
