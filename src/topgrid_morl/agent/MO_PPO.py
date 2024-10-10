import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import h5py
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
from topgrid_morl.utils.Dataloader import DataLoader
from topgrid_morl.utils.Grid2op_eval import evaluate_agent


class PPOReplayBuffer:
    """
    Replay buffer for single environment.

    Attributes:
        size (int): Maximum size of the buffer.
        obs_shape (Tuple[int, ...]): Shape of the observations.
        action_shape (Tuple[int, ...]): Shape of the actions.
        reward_dim (int): Dimension of the rewards.
        device (Union[th.device, str]): Device to store the buffer.
    """

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
            Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
            Observation, action, log probability, reward, done, and value.
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
            Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
            All observations, actions, log probabilities, rewards, dones, and values
            up to the current pointer.
        """
        return (
            self.obs,
            self.actions,
            self.logprobs,
            self.rewards,
            self.dones,
            self.values,
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
    """
    Neural network for the MOPPO agent.

    Attributes:
        obs_shape (Tuple[int, ...]): Shape of the observations.
        action_dim (int): Dimension of the action space.
        reward_dim (int): Dimension of the reward space.
        net_arch (List[int]): Architecture of the neural network.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        reward_dim: int,
        net_arch: List[int] = [256, 256, 256, 256],
        act_fcn=nn.ReLU,
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
        self.act_fcn = act_fcn

        self.critic = mlp(
            input_dim=np.prod(self.obs_shape),
            output_dim=self.reward_dim,
            net_arch=net_arch,
            activation_fn=act_fcn,
        )
        self.critic.apply(_hidden_layer_init)
        _critic_init(list(self.critic.modules())[-1])

        self.actor = mlp(
            input_dim=np.prod(self.obs_shape),
            output_dim=self.action_dim,
            net_arch=net_arch,
            activation_fn=act_fcn,
        )
        self.actor.apply(_hidden_layer_init)
        _value_init(list(self.actor.modules())[-1])
        
        print(self.net_arch)
        print(self.act_fcn)

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

    def reinitialize_last_layer(self):
        """
        Reinitialize the last layers of the actor and critic networks.
        """
        # Reinitialize last layer of critic
        last_critic_layer = self.get_last_linear_layer(self.critic)
        if last_critic_layer is not None:
            _critic_init(last_critic_layer)
            print("Critic's last layer reinitialized.")
        else:
            print("No linear layer found in critic network.")

        # Reinitialize last layer of actor
        last_actor_layer = self.get_last_linear_layer(self.actor)
        if last_actor_layer is not None:
            _value_init(last_actor_layer)
            print("Actor's last layer reinitialized.")
        else:
            print("No linear layer found in actor network.")

    def get_last_linear_layer(self, network):
        """
        Helper method to get the last nn.Linear layer in a network.

        Args:
            network (nn.Module): The neural network module to search.

        Returns:
            nn.Linear or None: The last linear layer if found, else None.
        """
        last_linear = None
        for layer in network.modules():
            if isinstance(layer, nn.Linear):
                last_linear = layer
        return last_linear

class MOPPO(MOPolicy):
    """
    Multi-Objective Proximal Policy Optimization algorithm.

    Attributes:
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

    def __init__(
        self,
        networks: MOPPONet,
        weights: npt.NDArray[np.float64],
        env: CustomGymEnv,
        env_val: CustomGymEnv,
        g2op_env: Any,
        g2op_env_val: Any,
        id: int = 1,
        log: bool = False,
        steps_per_iteration: int = 2048,
        num_minibatches: int = 32,
        update_epochs: int = 10,
        learning_rate: float = 3e-4,
        gamma: float = 0.995,
        anneal_lr: bool = False,
        clip_coef: float = 0.2,
        ent_coef: float = 0.5,
        vf_coef: float = 0.5,
        clip_vloss: bool = True,
        max_grad_norm: float = 0.5,
        norm_adv: bool = True,
        target_kl: Optional[float] = None,
        gae: bool = True,
        gae_lambda: float = 0.95,
        device: Union[th.device, str] = "cuda",
        seed: int = 42,
        generate_reward: bool = False,
        rng: Optional[np.random.Generator] = None,
        anneal_clip_coef: bool = True,  # New parameter
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
        self.env_val = env_val
        self.g2op_env = (g2op_env,)
        self.g2op_env_val = g2op_env_val
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
        self.debug = False
        
        self.anneal_clip_coef = anneal_clip_coef
        self.clip_coef_initial = clip_coef
        self.clip_coef_final = 0.1  # Target clip coefficientself.clip_coef_final = 0.1  # Target clip coefficient

        self.training_rewards = []
        self.evaluation_rewards = []

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

        # Add logs for state-action pairs and rewards
        self.state_action_log = []
        self.rewards_log = []

        self.generate_reward = generate_reward

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
            self.env_val,
            self.g2op_env,
            self.g2op_env_val,
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
            self.seed,
            self.generate_reward,
            self.np_random,
        )

        copied.global_step = self.global_step
        copied.eval_step = self.eval_step
        copied.optimizer = optim.Adam(
            copied_net.parameters(), lr=self.learning_rate, eps=1e-5
        )
        copied.batch = deepcopy(self.batch)
        copied.state_action_log = deepcopy(self.state_action_log)
        copied.rewards_log = deepcopy(self.rewards_log)
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

    def __collect_samples(self, obs: th.Tensor, done: bool) -> Tuple[th.Tensor, bool, int]:
        """
        Collect samples by interacting with the environment.
        """
        batch_size_collected = 0
        
        #self.batch.__init__()
        
        while batch_size_collected < self.batch_size:
            #print(batch_size_collected)
            with th.no_grad():
                action, logprob, entropy, value = self.networks.get_action_and_value(obs=obs) 
                value = value.view(self.networks.reward_dim)

            next_obs, reward, next_done, info, terminated_gym = self.env.step(action.item())
            self.global_step += 1

            reward = th.tensor(reward).to(self.device).view(self.networks.reward_dim)
            next_done = float(next_done)

            # Add the current transition to the batch (since the episode is ongoing)
            
            
            self.batch.add(obs, action, logprob, reward, done, value)
                
                
            batch_size_collected += 1
            self.chronic_steps += info["steps"]


            # Log the training reward for each step
            log_data = {
                f"train/reward_{self.reward_list_ext[i]}": reward[i].item()
                for i in range(self.networks.reward_dim)
            }
            log_data[f"train/grid2opsteps"] = self.chronic_steps
            log_data[f'train/steps/gymstep'] = info["steps"]
            if self.log:
                wandb.log(log_data, step=self.global_step)
            
            if next_done:
                # The episode has ended; reset the environment
                reset_obs = self.env.reset()
                next_obs = th.tensor(reset_obs).to(self.device)
                next_done= float(True)
                if self.debug: 
                    print(f'reset env with {self.chronic_steps} and rewards {reward}')
                self.chronic_steps = 0
            
            # Update obs and done for the next iteration
            obs = th.Tensor(next_obs).to(self.device)
            done = next_done  # Already a float

        return obs, done, self.batch.rewards.mean().item()

    def __compute_advantages(
        self, next_obs: th.Tensor, next_done: bool
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute advantages and returns
        """
        returns: th.Tensor
        with th.no_grad():
            next_value = self.networks.get_value(next_obs).reshape(
                -1, self.networks.reward_dim
            )
            if self.gae:
                advantages: th.Tensor = th.zeros_like(self.batch.rewards).to(
                    self.device
                )
                lastgaelam = 0
                if self.debug ==True: 
                    #   Debug: Print the initial next_value and next_done
                    print(f"Next Value: {next_value}")
                    print(f"Next Done: {next_done}")

                for t in reversed(range(self.batch_size)):
                    if t == self.steps_per_iteration - 1:
                        if self.debug: 
                            print('last iteration within batch')
                            print(next_done)
                        nextnonterminal = 1.0 - next_done #last iteration within batch -> next_done
                        nextvalues = next_value
                        if self.debug: 
                            print(nextnonterminal)
                    else:
                                 
                        _, _, _, _, done_t1, value_t1 = self.batch.get(t + 1)
                        _, _, _, _, _, value_t_debug = self.batch.get(t)
                                                   
                        nextnonterminal = 1.0 - done_t1 #if terminal 0 - if non terminal: value!
                        if self.debug:
                            print('all other entries in batch:')
                            print(done_t1)
                            print(nextnonterminal)
                            print(f'Value {value_t_debug}')
                            print(f'Nextvalue {value_t1}')
                            
                        nextvalues = value_t1
                    nextnonterminal = th.tensor(nextnonterminal)
                    
                    nextnonterminal = self.__extend_to_reward_dim(nextnonterminal)
                    
                    _, _, _, reward_t, _, value_t = self.batch.get(t)
                    delta = (
                        reward_t + self.gamma * nextvalues * nextnonterminal - value_t
                    )
                    advantages[t] = lastgaelam = (
                        delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                    )

                returns = advantages + self.batch.values

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
                    
                advantages = returns - self.batch.values()

        # Debug: Print computed returns and advantages
        if self.debug ==True: 
            print(f"Returns: {returns}")
            print(f"Advantages: {advantages}")

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
    def update(self) -> None:
        """
        Update the policy and value function.
        """
        #eval_done = False
        #eval_state = self.env_val.reset(options={"max step": 28 * 288})
        obs, actions, logprobs, _, _, values = self.batch.get_all()
        if self.debug ==True: 
            print('in update')
        b_obs = obs.reshape((-1,) + self.networks.obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1, self.networks.reward_dim)
        b_values = values.reshape(-1, self.networks.reward_dim)

        b_inds = np.arange(self.batch_size)
        clipfracs = []
        if self.debug ==True: 
                    # Debug: Print policy and value loss info per minibatch
                    print(f"Batch obs: {b_obs}")
                    print(f"Batch actions: {b_actions}")
                    print(f"b_returns: {b_returns}")
                    

        for epoch in range(self.update_epochs):
            if self.debug ==True: 
                print(epoch)
            self.np_random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

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
                if self.debug ==True: 
                    # Debug: Print policy and value loss info per minibatch
                    print(f"Epoch: {epoch}, Batch Start: {start}, Batch End: {end}, Minibatchs_inds: {mb_inds}")
                    print(f"LogProbs Ratio: {ratio}")
                    print(f"Approx KL: {approx_kl}")
                    print(f"ClipFrac: {clipfracs[-1]}")

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * th.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = th.max(pg_loss1, pg_loss2).mean()

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

                if self.debug ==True: 
                    # Debug: Print losses after each minibatch
                    print(f"Policy Loss: {pg_loss}")
                    print(f"Value Loss: {v_loss}")
                    print(f"Entropy Loss: {entropy.mean()}")

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
        if self.debug ==True:     
            # Debug: Print evaluation rewards after each update
            print(f"Evaluation Rewards: {self.evaluation_rewards}")
        
            # Evaluate and log evaluation rewards
        chronics = self.g2op_env_val.chronics_handler.available_chronics()
        #print(chronics)
        eval_rewards = []
        eval_steps = []
        for idx, chronic in enumerate(chronics):
            eval_data = evaluate_agent(
                agent=self,
                env=self.env_val,
                g2op_env=self.g2op_env_val,
                g2op_env_val=self.g2op_env_val,
                weights=self.weights,
                eval_steps=28 * 288,
                eval_counter=self.eval_counter,
                chronic=chronic,
                idx=idx,
                reward_list=self.reward_list,
                seed=self.seed,
            )
            eval_rewards.append(th.stack(eval_data["eval_rewards"]).mean(dim=0))
            eval_steps.append(eval_data["eval_steps"])

        eval_rewards = th.stack(eval_rewards).mean(dim=0)
        eval_steps = np.array(eval_steps).mean()

        self.evaluation_rewards.append(eval_rewards.cpu().numpy())
        
        if self.log:
            log_rew_data = {
                f"eval/reward_{self.reward_list_ext[i]}": eval_rewards[i].item()
                for i in range(self.networks.reward_dim)
            }
            log_step_data = {f"eval/steps": eval_steps}
            wandb.log(log_rew_data, step=self.global_step)
            wandb.log(log_step_data, step=self.global_step)
        
        self.eval_counter += 1
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if self.log:
            wandb.log(
                {
                    f"losses_{self.id}/value_loss": v_loss.item(),
                    f"charts_{self.id}/learning_rate": self.optimizer.param_groups[0][
                        "lr"
                    ],
                    f"charts_{self.id}/clip_coef": self.clip_coef,
                    f"losses_{self.id}/policy_loss": pg_loss.item(),
                    f"losses_{self.id}/entropy": entropy_loss.item(),
                    f"losses_{self.id}/old_approx_kl": old_approx_kl.item(),
                    f"losses_{self.id}/approx_kl": approx_kl.item(),
                    f"losses_{self.id}/clipfrac": np.mean(clipfracs),
                    f"losses_{self.id}/explained_variance": explained_var,
                    "global_step": self.global_step,
                }
            )
        

    def train(
        self,
        max_gym_steps: int,
        reward_dim: int,
        reward_list: list,
    ) -> None:
        """
        Train the agent.

        Args:
            max_gym_steps (int): Total gym steps.
            reward_dim (int): Dimension of the reward.
        """
        print(self.clip_coef)
        self.reward_list = reward_list
        self.reward_list_ext = [
            "L2RPN",
            self.reward_list[0],
            self.reward_list[1],
        ]
        state = self.env.reset()
        self.chronic_steps = 0
        obs = th.Tensor(state).to(self.device)
        done = 0

        num_trainings = int(max_gym_steps / self.batch_size)
        self.eval_step = 0
        self.global_step = 0
        self.eval_counter = 0
        
        for trainings in range(num_trainings):
            state = self.env.reset()
            if self.debug ==True: 
                print('Env Reset in Training Loop')
            self.chronic_steps = 0
            obs = th.Tensor(state).to(self.device)
            done = 0

            next_obs, next_done, _ = self.__collect_samples(obs, done)

            self.returns, self.advantages = self.__compute_advantages(next_obs, next_done)
            self.update()
            
            done = next_done
            obs = next_obs

            if self.anneal_lr:
                frac = 1.0 - (self.global_step / max_gym_steps)
                new_lr = self.learning_rate * frac
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = new_lr
            
            # Anneal clip coefficient if required
            if self.anneal_clip_coef:
                frac = float((self.global_step / max_gym_steps))
                self.clip_coef = self.clip_coef_initial - frac * (self.clip_coef_initial - self.clip_coef_final)
                self.clip_coef = np.round(self.clip_coef,decimals=2)
            

            
            
                    
          
