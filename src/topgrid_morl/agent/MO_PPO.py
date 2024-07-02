import time
from copy import deepcopy
from typing import List, Optional, Union
from typing_extensions import override

import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import torch as th
import wandb
from mo_gymnasium import MORecordEpisodeStatistics
from torch import nn, optim
from torch.distributions import Categorical

from morl_baselines.common.evaluation import log_episode_info
from morl_baselines.common.morl_algorithm import MOPolicy
from morl_baselines.common.networks import layer_init, mlp

from topgrid_morl.envs.CustomGymEnv import CustomGymEnv
import pandas as pd


class PPOReplayBuffer:
    """Replay buffer for single environment."""
    
    def __init__(self, size: int, obs_shape: tuple, action_shape: tuple, reward_dim: int, device: Union[th.device, str]):
        self.size = size
        self.ptr = 0
        self.device = device
        self.obs = th.zeros((self.size,) + obs_shape).to(device)
        self.actions = th.zeros((self.size,) + action_shape, dtype=th.long).to(device)
        self.logprobs = th.zeros((self.size,)).to(device)
        self.rewards = th.zeros((self.size, reward_dim), dtype=th.float32).to(device)
        self.dones = th.zeros((self.size,)).to(device)
        self.values = th.zeros((self.size, reward_dim), dtype=th.float32).to(device)

    def add(self, obs, actions, logprobs, rewards, dones, values):
        """Add experience to the buffer."""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.logprobs[self.ptr] = logprobs
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.values[self.ptr] = values
        self.ptr = (self.ptr + 1) % self.size

    def get(self, step: int):
        """Get experience from a specific step."""
        return (self.obs[step], self.actions[step], self.logprobs[step], self.rewards[step], self.dones[step], self.values[step])

    def get_all(self):
        """Get all experiences in the buffer."""
        return (self.obs[:self.ptr], self.actions[:self.ptr], self.logprobs[:self.ptr], self.rewards[:self.ptr, :], self.dones[:self.ptr], self.values[:self.ptr, :])
    
    def get_ptr(self): 
        """Get current pointer of the buffer."""
        return self.ptr - 1
    
    def get_values(self):
        """Get all value predictions."""
        return self.values[:self.ptr, :]
    
    def get_rewards(self):
        """Get all rewards."""
        return self.rewards[:self.ptr, :]


def make_env(env_id: str, seed: int, run_name: str, gamma: float) -> gym.Env:
    """Create and configure a new environment instance."""
    env = mo_gym.make(env_id, render_mode="rgb_array")
    reward_dim = env.reward_space.shape[0]
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    for o in range(reward_dim):
        env = mo_gym.utils.MONormalizeReward(env, idx=o, gamma=gamma)
        env = mo_gym.utils.MOClipReward(env, idx=o, min_r=-10, max_r=10)
    env = MORecordEpisodeStatistics(env, gamma=gamma)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def _hidden_layer_init(layer: nn.Module) -> None:
    """Initialize hidden layer weights."""
    layer_init(layer, weight_gain=np.sqrt(2), bias_const=0.0)


def _critic_init(layer: nn.Module) -> None:
    """Initialize critic network weights."""
    layer_init(layer, weight_gain=1.0)


def _value_init(layer: nn.Module) -> None:
    """Initialize value network weights."""
    layer_init(layer, weight_gain=0.01)


class MOPPONet(nn.Module):
    """Multi-Objective PPO Network."""
    
    def __init__(self, obs_shape: tuple, action_dim: int, reward_dim: int, net_arch: List[int] = [64, 64]) -> None:
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.net_arch = net_arch

        self.critic = mlp(input_dim=np.prod(self.obs_shape), output_dim=self.reward_dim, net_arch=net_arch, activation_fn=nn.Tanh)
        self.critic.apply(_hidden_layer_init)
        _critic_init(list(self.critic.modules())[-1])

        self.actor = mlp(input_dim=np.prod(self.obs_shape), output_dim=self.action_dim, net_arch=net_arch, activation_fn=nn.Tanh)
        self.actor.apply(_hidden_layer_init)
        _value_init(list(self.actor.modules())[-1])

    def get_value(self, obs: th.Tensor) -> th.Tensor:
        """Get value prediction."""
        return self.critic(obs.to(self.device))

    def get_action_and_value(self, obs: th.Tensor, action: Optional[th.Tensor] = None) -> tuple:
        """Get action, log probability, entropy, and value."""
        obs = obs.to(self.device)
        logits = self.actor(obs)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(obs)


class MOPPO(MOPolicy):
    """Multi-Objective PPO Algorithm."""
    
    def __init__(
        self,
        id: int,
        networks: MOPPONet,
        weights: np.ndarray,
        env: CustomGymEnv,
        log: bool = False,
        steps_per_iteration: int = 2048,
        num_minibatches: int = 32,
        update_epochs: int = 10,
        learning_rate: float = 3e-4,
        gamma: float = 0.995,
        anneal_lr: bool = False,
        clip_coef: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        clip_vloss: bool = True,
        max_grad_norm: float = 0.5,
        norm_adv: bool = True,
        target_kl: Optional[float] = None,
        gae: bool = True,
        gae_lambda: float = 0.95,
        device: Union[th.device, str] = "cuda",
        seed: int = 42,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(id, device)
        self.id = id
        self.env = env
        self.networks = networks.to(device)
        self.device = device
        self.seed = seed
        if rng is not None:
            self.np_random = rng
        else:
            self.np_random = np.random.default_rng(self.seed)

        self.steps_per_iteration = steps_per_iteration
        self.weights = th.from_numpy(weights).to(self.device)
        self.batch_size = self.steps_per_iteration
        self.num_minibatches = num_minibatches
        self.minibatch_size = self.batch_size // num_minibatches
        self.update_epochs = update_epochs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.anneal_lr = anneal_lr
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.norm_adv = norm_adv
        self.target_kl = target_kl
        self.clip_vloss = clip_vloss
        self.gae_lambda = gae_lambda
        self.log = log
        self.gae = gae

        self.optimizer = optim.Adam(networks.parameters(), lr=self.learning_rate, eps=1e-5)

        self.batch = PPOReplayBuffer(self.steps_per_iteration, self.networks.obs_shape, (1,), self.networks.reward_dim, self.device)

    def __deepcopy__(self, memo):
        """Deepcopy method for the MOPPO class."""
        copied_net = deepcopy(self.networks)
        copied = type(self)(
            self.id,
            copied_net,
            self.weights.detach().cpu().numpy(),
            self.env,
            self.log,
            self.steps_per_iteration,
            self.num_minibatches,
            self.update_epochs,
            self.learning_rate,
            self.gamma,
            self.anneal_lr,
            self.clip_coef,
            self.ent_coef,
            self.vf_coef,
            self.clip_vloss,
            self.max_grad_norm,
            self.norm_adv,
            self.target_kl,
            self.gae,
            self.gae_lambda,
            self.device,
        )

        copied.global_step = self.global_step
        copied.optimizer = optim.Adam(copied_net.parameters(), lr=self.learning_rate, eps=1e-5)
        copied.batch = deepcopy(self.batch)
        return copied

    def change_weights(self, new_weights: np.ndarray):
        """Change the weights used for reward aggregation."""
        self.weights = th.from_numpy(deepcopy(new_weights)).to(self.device)

    def __extend_to_reward_dim(self, tensor: th.Tensor):
        """Extend tensor dimensions to match reward dimensions."""
        dim_diff = self.networks.reward_dim - tensor.dim()
        if dim_diff > 0:
            return tensor.unsqueeze(-1).expand(*tensor.shape, self.networks.reward_dim)
        elif dim_diff < 0:
            return tensor.squeeze(-1)
        else:
            return tensor

    def __collect_samples(self, obs: th.Tensor, done: th.Tensor, max_ep_steps):
        """Collect samples by interacting with the environment."""
        count_episode = 1
        done = False
        cum_reward = 0
        action_list = []
        gym_steps = 0
        grid2op_steps = 0
        while not done and gym_steps < max_ep_steps:
            self.global_step += 1

            with th.no_grad():
                action, logprob, _, value = self.networks.get_action_and_value(obs.to(self.device))
                value = value.view(self.networks.reward_dim)
            
            next_obs, reward, next_done, info = self.env.step(action.item())
            reward = th.tensor(reward).to(self.device).view(self.networks.reward_dim)
            cum_reward += reward
            self.batch.add(obs, action, logprob, reward, done, value)
            action_list.append(action)
            steps_in_gymSteps = info['steps']
            obs, done = th.Tensor(next_obs).to(self.device), th.tensor(next_done).float().to(self.device)

            if "episode" in info.keys():
                log_episode_info(
                    info["episode"],
                    scalarization=np.dot,
                    weights=self.weights,
                    global_timestep=self.global_step,
                    id=self.id,
                )
            
            gym_steps += 1
            grid2op_steps += steps_in_gymSteps
        return obs, done, cum_reward, action_list, grid2op_steps

    def __compute_advantages(self, next_obs, next_done):
        """Compute advantages for the collected samples."""
        with th.no_grad():
            next_value = self.networks.get_value(next_obs).reshape(-1, self.networks.reward_dim)
            if self.gae:
                advantages = th.zeros_like(self.batch.get_rewards()).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.batch.get_ptr())):
                    if t == self.steps_per_iteration - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        _, _, _, _, done_t1, value_t1 = self.batch.get(t + 1)
                        nextnonterminal = 1.0 - done_t1
                        nextvalues = value_t1

                    nextnonterminal = self.__extend_to_reward_dim(nextnonterminal)
                    _, _, _, reward_t, _, value_t = self.batch.get(t)
                    delta = reward_t + self.gamma * nextvalues * nextnonterminal - value_t
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.batch.get_values()
            else:
                returns = th.zeros_like(self.batch.get_rewards()).to(self.device)
                for t in reversed(range(self.steps_per_iteration)):
                    if t == self.steps_per_iteration - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        _, _, _, _, done_t1, _ = self.batch.get(t + 1)
                        nextnonterminal = 1.0 - done_t1
                        next_return = returns[t + 1]

                    nextnonterminal = self.__extend_to_reward_dim(nextnonterminal)
                    _, _, _, reward_t, _, _ = self.batch.get(t)
                    returns[t] = reward_t + self.gamma * nextnonterminal * next_return
                advantages = returns - self.batch.get_values()
        advantages = advantages @ self.weights.float()  # Compute dot product of advantages and weights
        return returns, advantages

    @override
    def eval(self, obs: np.ndarray, w):
        """Evaluate the policy."""
        obs = th.as_tensor(obs).float().to(self.device).unsqueeze(0)
        with th.no_grad():
            action, _, _, _ = self.networks.get_action_and_value(obs)
        return action[0].detach().cpu().numpy()

    @override
    def update(self):
        """Update the policy using the collected samples."""
        obs, actions, logprobs, _, _, values = self.batch.get_all()
        original_batch_size = self.batch_size
        if self.batch_size > self.batch.get_ptr():
            self.batch_size = self.batch.get_ptr()
        b_obs = obs.reshape((-1,) + self.networks.obs_shape).to(self.device)
        b_logprobs = logprobs.reshape(-1).to(self.device)
        b_actions = actions.reshape(-1).to(self.device)
        b_advantages = self.advantages.reshape(-1).to(self.device)
        b_returns = self.returns.reshape(-1, self.networks.reward_dim).to(self.device)
        b_values = values.reshape(-1, self.networks.reward_dim).to(self.device)

        b_inds = np.arange(obs.shape[0])
        clipfracs = []
        v_loss = None
        pg_loss = None
        entropy_loss = None
        old_approx_kl = None
        approx_kl = None
        
        for epoch in range(self.update_epochs):
            self.np_random.shuffle(b_inds)
            
            if self.batch_size > 0: 
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.networks.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with th.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * th.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = th.max(pg_loss1, pg_loss2).mean()
                    
                    newvalue = newvalue.view(-1, self.networks.reward_dim)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + th.clamp(newvalue - b_values[mb_inds], -self.clip_coef, self.clip_coef)
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = th.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.networks.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        self.batch_size = original_batch_size 
        
        if self.log:
            wandb.log(
                {   
                    f"losses_{self.id}/value_loss": v_loss.item() if v_loss is not None else float('nan'),
                    f"charts_{self.id}/learning_rate": self.optimizer.param_groups[0]["lr"],
                    f"losses_{self.id}/policy_loss": pg_loss.item() if pg_loss is not None else float('nan'),
                    f"losses_{self.id}/entropy": entropy_loss.item() if entropy_loss is not None else float('nan'),
                    f"losses_{self.id}/old_approx_kl": old_approx_kl.item() if old_approx_kl is not None else float('nan'),
                    f"losses_{self.id}/approx_kl": approx_kl.item() if approx_kl is not None else float('nan'),
                    f"losses_{self.id}/clipfrac": np.mean(clipfracs),
                    f"losses_{self.id}/explained_variance": explained_var,
                    "global_step": self.global_step,
                }
            )

    def save_model(self, path: str):
        """Save the model parameters to the specified path."""
        th.save(self.networks.state_dict(), path)

    def train(self, num_episodes: int, max_ep_steps: int, reward_dim: int, print_every: int = 100, print_flag: bool = True) -> np.ndarray:
        """Train the policy."""
        reward_matrix = np.zeros((num_episodes, reward_dim))
        actions = []
        total_steps = []
        for i_episode in range(num_episodes):
            if self.anneal_lr:
                new_lr = self.learning_rate * (1 - i_episode / num_episodes)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr

            state = self.env.reset()
            next_obs = th.Tensor(state).to(self.device)
            done = False
            next_done = done
            episode_reward = th.zeros(reward_dim).to(self.device)
            
            action_list_episode = []

            next_obs, next_done, episode_reward, action_list_episode, ep_steps = self.__collect_samples(next_obs, next_done, max_ep_steps)
            self.returns, self.advantages = self.__compute_advantages(next_obs, next_done)
            self.update()

            actions.append([action.cpu().numpy() for action in action_list_episode])
            reward_matrix[i_episode] = episode_reward.cpu().numpy()
            total_steps.append(ep_steps)

            if self.log:
                log_data = {
                    f"charts_{self.id}/episode_reward_sum": episode_reward.sum().item(),
                    f"charts_{self.id}/episode": i_episode,
                    "global_step": self.global_step,
                }
                for j in range(reward_dim):
                    log_data[f"charts_{self.id}/episode_reward_{j}"] = episode_reward[j].item()
                wandb.log(log_data)

            if print_flag and (i_episode + 1) % print_every == 0:
                print(f"Episode {i_episode + 1}/{num_episodes}")
                print(f"  Episode Reward Sum: {episode_reward.sum().item()}")
                for j in range(reward_dim):
                    print(f"  Episode Reward {j}: {episode_reward[j].item()}")
                print(f"  Actions: {actions[-1]}")

        print('Training complete')
        print(total_steps)
        return reward_matrix, actions, total_steps

