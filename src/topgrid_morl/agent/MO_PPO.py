from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import numpy.typing as npt
import torch as th
import wandb
from mo_gymnasium import MORecordEpisodeStatistics
from morl_baselines.common.morl_algorithm import MOPolicy
from morl_baselines.common.networks import layer_init, mlp
from torch import nn, optim
from torch.distributions import Categorical
from typing_extensions import override

from topgrid_morl.envs.CustomGymEnv import CustomGymEnv


class PPOReplayBuffer:
    """Replay buffer for single environment."""

    def __init__(
        self,
        size: int,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        reward_dim: int,
        device: Union[th.device, str],
    ) -> None:
        """
        Initialize the replay buffer.

        Args:
            size (int): Maximum size of the buffer.
            obs_shape (Tuple[int, ...]): Shape of the observations.
            action_shape (Tuple[int, ...]): Shape of the actions.
            reward_dim (int): Dimension of the rewards.
            device (Union[th.device, str]): Device to store the buffer.
        """
        self.size = size
        self.ptr = 0
        self.device = device
        self.obs = th.zeros((self.size,) + obs_shape).to(device)
        self.actions = th.zeros((self.size,) + action_shape, dtype=th.long).to(device)
        self.logprobs = th.zeros((self.size,)).to(device)
        self.rewards = th.zeros((self.size, reward_dim), dtype=th.float32).to(device)
        self.dones = th.zeros((self.size,), dtype=th.float32).to(device)
        self.values = th.zeros((self.size, reward_dim), dtype=th.float32).to(device)

    def add(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        logprobs: th.Tensor,
        rewards: th.Tensor,
        dones: bool,
        values: th.Tensor,
    ) -> None:
        """
        Add a new experience to the buffer.

        Args:
            obs (th.Tensor): Observation.
            actions (th.Tensor): Action taken.
            logprobs (th.Tensor): Log probability of the action.
            rewards (th.Tensor): Rewards received.
            dones (bool): Whether the episode is done.
            values (th.Tensor): Value function estimation.
        """
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.logprobs[self.ptr] = logprobs
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.values[self.ptr] = values
        self.ptr = (self.ptr + 1) % self.size

    def get(
        self, step: int
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Get an experience from a specific step.

        Args:
            step (int): The index of the step.

        Returns:
            Tuple containing observation, action, log probability, reward, done, and value.
        """
        return (
            self.obs[step],
            self.actions[step],
            self.logprobs[step],
            self.rewards[step],
            self.dones[step],
            self.values[step],
        )

    def get_all(
        self,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Get all experiences in the buffer.

        Returns:
            Tuple containing all observations, actions, log probabilities,
            rewards, dones, and values up to the current pointer.
        """
        return (
            self.obs[: self.ptr],
            self.actions[: self.ptr],
            self.logprobs[: self.ptr],
            self.rewards[: self.ptr, :],
            self.dones[: self.ptr],
            self.values[: self.ptr, :],
        )

    def get_ptr(self) -> int:
        """
        Get the current pointer of the buffer.

        Returns:
            int: Current pointer.
        """
        return self.ptr

    def get_values(self) -> th.Tensor:
        """
        Get all value predictions.

        Returns:
            th.Tensor: All value predictions up to the current pointer.
        """
        return self.values[: self.ptr, :]

    def get_rewards(self) -> th.Tensor:
        """
        Get all rewards.

        Returns:
            th.Tensor: All rewards up to the current pointer.
        """
        return self.rewards[: self.ptr, :]


def make_env(env_id: str, seed: int, run_name: str, gamma: float) -> gym.Env:
    """
    Create and configure the environment.

    Args:
        env_id (str): ID of the environment.
        seed (int): Random seed.
        run_name (str): Name of the run.
        gamma (float): Discount factor.

    Returns:
        gym.Env: Configured environment.
    """
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
    """
    Initialize hidden layers.

    Args:
        layer (nn.Module): Neural network layer to initialize.
    """
    layer_init(layer, weight_gain=np.sqrt(2), bias_const=0.0)


def _critic_init(layer: nn.Module) -> None:
    """
    Initialize critic layers.

    Args:
        layer (nn.Module): Neural network layer to initialize.
    """
    layer_init(layer, weight_gain=1.0)


def _value_init(layer: nn.Module) -> None:
    """
    Initialize value layers.

    Args:
        layer (nn.Module): Neural network layer to initialize.
    """
    layer_init(layer, weight_gain=0.01)


class MOPPONet(nn.Module):
    """Neural network for the MOPPO agent."""

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        reward_dim: int,
        net_arch: List[int] = [64, 64],
    ) -> None:
        """
        Initialize the neural network.

        Args:
            obs_shape (Tuple[int, ...]): Shape of the observations.
            action_dim (int): Dimension of the action space.
            reward_dim (int): Dimension of the reward space.
            net_arch (List[int]): Architecture of the neural network.
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.net_arch = net_arch

        self.critic = mlp(
            input_dim=np.prod(self.obs_shape),
            output_dim=self.reward_dim,
            net_arch=net_arch,
            activation_fn=nn.Tanh,
        )
        self.critic.apply(_hidden_layer_init)
        _critic_init(list(self.critic.modules())[-1])

        self.actor = mlp(
            input_dim=np.prod(self.obs_shape),
            output_dim=self.action_dim,
            net_arch=net_arch,
            activation_fn=nn.Tanh,
        )
        self.actor.apply(_hidden_layer_init)
        _value_init(list(self.actor.modules())[-1])

    def get_value(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the value of the given observation.

        Args:
            obs (th.Tensor): Observation.

        Returns:
            th.Tensor: Value of the observation.
        """
        return self.critic(obs)

    def get_action_and_value(
        self, obs: th.Tensor, action: Optional[th.Tensor] = None
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the action and value of the given observation.

        Args:
            obs (th.Tensor): Observation.
            action (Optional[th.Tensor]): Action.

        Returns:
            Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
            Action, log probability, entropy, and value of the observation.
        """
        logits = self.actor(obs)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(obs)


class MOPPO(MOPolicy):
    """Multi-Objective Proximal Policy Optimization algorithm."""

    def __init__(
        self,
        id: int,
        networks: MOPPONet,
        weights: npt.NDArray[np.float64],
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
        """
        Initialize the MOPPO agent.

        Args:
            id (int): Identifier for the agent.
            networks (MOPPONet): Neural network for the agent.
            weights (npt.NDArray[np.float64]): Weights for the objectives.
            env (CustomGymEnv): Custom gym environment.
            log (bool): Whether to log the training process.
            steps_per_iteration (int): Steps per iteration.
            num_minibatches (int): Number of minibatches.
            update_epochs (int): Number of update epochs.
            learning_rate (float): Learning rate.
            gamma (float): Discount factor.
            anneal_lr (bool): Whether to anneal the learning rate.
            clip_coef (float): Clipping coefficient.
            ent_coef (float): Entropy coefficient.
            vf_coef (float): Value function coefficient.
            clip_vloss (bool): Whether to clip value loss.
            max_grad_norm (float): Maximum gradient norm.
            norm_adv (bool): Whether to normalize advantages.
            target_kl (Optional[float]): Target KL divergence.
            gae (bool): Whether to use Generalized Advantage Estimation.
            gae_lambda (float): GAE lambda.
            device (Union[th.device, str]): Device to use.
            seed (int): Random seed.
            rng (Optional[np.random.Generator]): Random number generator.
        """
        super().__init__(id, device)
        self.id = id
        self.env = env
        self.networks = networks
        self.device = device
        self.seed = seed
        self.np_random = rng if rng is not None else np.random.default_rng(self.seed)

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

        self.optimizer = optim.Adam(
            networks.parameters(), lr=self.learning_rate, eps=1e-5
        )
        self.batch = PPOReplayBuffer(
            self.steps_per_iteration,
            self.networks.obs_shape,
            (1,),
            self.networks.reward_dim,
            self.device,
        )

    def __deepcopy__(self, memo: Dict[int, Any]) -> "MOPPO":
        """
        Create a deep copy of the agent.

        Args:
            memo (Dict[int, Any]): Memoization dictionary.

        Returns:
            MOPPO: Deep copied agent.
        """
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
        copied.optimizer = optim.Adam(
            copied_net.parameters(), lr=self.learning_rate, eps=1e-5
        )
        copied.batch = deepcopy(self.batch)
        return copied

    def change_weights(self, new_weights: npt.NDArray[np.float64]) -> None:
        """
        Change the weights for the objectives.

        Args:
            new_weights (npt.NDArray[np.float64]): New weights for the objectives.
        """
        self.weights = th.from_numpy(deepcopy(new_weights)).to(self.device)

    def __extend_to_reward_dim(self, tensor: th.Tensor) -> th.Tensor:
        """
        Extend tensor to the reward dimension.

        Args:
            tensor (th.Tensor): Tensor to extend.

        Returns:
            th.Tensor: Extended tensor.
        """
        dim_diff = self.networks.reward_dim - tensor.dim()
        if dim_diff > 0:
            return tensor.unsqueeze(-1).expand(*tensor.shape, self.networks.reward_dim)
        elif dim_diff < 0:
            return tensor.squeeze(-1)
        else:
            return tensor

    def __collect_samples(
        self, obs: th.Tensor, done: bool, grid2op_steps: int
    ) -> Tuple[th.Tensor, bool, th.Tensor, int]:
        """
        Collect samples by interacting with the environment.

        Args:
            obs (th.Tensor): Initial observation.
            done (bool): Whether the episode is done.
            max_ep_steps (int): Maximum number of steps per episode.

        Returns:
            Tuple containing the next observation, whether the episode is
            done, cumulative reward, list of actions, and total steps.
        """
        for gym_step in range(self.batch_size):
            # fill batch
            if done:
                self.env.reset()

            with th.no_grad():
                action, logprob, _, value = self.networks.get_action_and_value(
                    obs.to(self.device)
                )
                value = value.view(self.networks.reward_dim)

            next_obs, reward, next_done, info = self.env.step(action.item())
            reward = th.tensor(reward).to(self.device).view(self.networks.reward_dim)
            self.batch.add(obs, action, logprob, reward, done, value)
            steps_in_gymstep = info["steps"]
            obs, done = th.Tensor(next_obs).to(self.device), th.tensor(
                next_done
            ).float().to(self.device)
            grid2op_steps += steps_in_gymstep

        return obs, done, grid2op_steps

    def __compute_advantages(
        self, next_obs: th.Tensor, next_done: bool
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute advantages and returns.

        Args:
            next_obs (th.Tensor): Next observation.
            next_done (bool): Whether the next state is done.

        Returns:
            Tuple[th.Tensor, th.Tensor]: Returns and advantages.
        """
        with th.no_grad():
            next_value = self.networks.get_value(next_obs).reshape(
                -1, self.networks.reward_dim
            )
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
                    delta = (
                        reward_t + self.gamma * nextvalues * nextnonterminal - value_t
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                    )
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
        advantages = (
            advantages @ self.weights.float()
        )  # Compute dot product of advantages and weights
        return returns, advantages

    @override
    def eval(
        self, obs: npt.NDArray[np.float64], w: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Evaluate the policy for a given observation and weights.

        Args:
            obs (npt.NDArray[np.float64]): Observation.
            w (npt.NDArray[np.float64]): Weights.

        Returns:
            npt.NDArray[np.float64]: Action.
        """
        obs = th.as_tensor(obs).float().to(self.device).unsqueeze(0)
        with th.no_grad():
            action, _, _, _ = self.networks.get_action_and_value(obs)
        return action[0].detach().cpu().numpy()

    @override
    def update(self):
        """
        Update the policy and value function.
        """
        obs, actions, logprobs, _, _, values = self.batch.get_all()
        print(obs)
        b_obs = obs.reshape((-1,) + self.networks.obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + (self.networks.action_dim,))
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1, self.networks.reward_dim)
        b_values = values.reshape(-1, self.networks.reward_dim)

        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl = (
            None,
            None,
            None,
            None,
            None,
        )

        for epoch in range(self.update_epochs):
            self.np_random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]
                print(mb_inds)
                print(b_obs)
                _, newlogprob, entropy, newvalue = self.networks.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with th.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * th.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = th.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1, self.networks.reward_dim)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + th.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
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

            if (
                self.target_kl is not None
                and approx_kl is not None
                and approx_kl > self.target_kl
            ):
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if self.log:
            wandb.log(
                {
                    f"losses_{self.id}/value_loss": v_loss.item()
                    if v_loss is not None
                    else float("nan"),
                    f"charts_{self.id}/learning_rate": self.optimizer.param_groups[0][
                        "lr"
                    ],
                    f"losses_{self.id}/policy_loss": pg_loss.item()
                    if pg_loss is not None
                    else float("nan"),
                    f"losses_{self.id}/entropy": entropy_loss.item()
                    if entropy_loss is not None
                    else float("nan"),
                    f"losses_{self.id}/old_approx_kl": old_approx_kl.item()
                    if old_approx_kl is not None
                    else float("nan"),
                    f"losses_{self.id}/approx_kl": approx_kl.item()
                    if approx_kl is not None
                    else float("nan"),
                    f"losses_{self.id}/clipfrac": np.mean(clipfracs),
                    f"losses_{self.id}/explained_variance": explained_var,
                    "global_step": self.global_step,
                }
            )

    def save_model(self, path: str) -> None:
        """
        Save the model to the specified path.

        Args:
            path (str): Path to save the model.
        """
        th.save(self.networks.state_dict(), path)

    def train(
        self,
        max_gym_steps: int,
        reward_dim: int,
    ) -> None:
        """
        Train the agent.

        Args:
            max_gym_steps (int): Total gym steps.
            reward_dim (int): Dimension of the reward.
        """
        grid2op_steps = 0
        num_trainings = int(max_gym_steps / self.batch_size)

        for trainings in range(num_trainings):
            state = self.env.reset()
            next_obs = th.Tensor(state).to(self.device)
            done = False

            next_obs, done, grid2op_steps_from_training = self.__collect_samples(
                next_obs, done, grid2op_steps=grid2op_steps
            )
            grid2op_steps += grid2op_steps_from_training
            self.returns, self.advantages = self.__compute_advantages(next_obs, done)
            self.update()
