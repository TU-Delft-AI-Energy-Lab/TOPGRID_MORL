
"""
Aims to replicate Blazej's single-agent PPO one-on-one.
"""
from typing import Union, Any, Dict, Optional
# import packages
import json
import os
import random
from typing import Any

import grid2op
import gymnasium as gym
import numpy as np
#import ray
from grid2op.Action import BaseAction, PowerlineSetAction
from grid2op.Converter import IdToAct
from grid2op.dtypes import dt_float
from grid2op.Environment import BaseEnv
from grid2op.gym_compat import GymEnv, ScalerAttrConverter
from grid2op.Parameters import Parameters
from grid2op.Reward import L2RPNReward
from gymnasium.spaces import Discrete
from lightsim2grid import LightSimBackend
"""
from ray import tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.tune import CLIReporter
"""
from grid2op.Opponent import (
    BaseActionBudget,
    RandomLineOpponent,
)

import grid2op
import numpy as np

import logging
from typing import OrderedDict, Tuple

import gymnasium as gym
import numpy as np
import torch
"""
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import Dict, List, ModelConfigDict, TensorType
"""
from torch import nn

# Limit values suitable for use as close to a -inf logit. These are useful
# since -inf / inf cause NaNs during backprop.
FLOAT_MIN = -3.4e38
FLOAT_MAX = 3.4e38


class SimpleMlp(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        """
        Initialize the model.

        Parameters:
        ----------
        obs_space: gym.spaces.Space
            The observation space of the environment.
        action_space: gym.spaces.Space
            The action space of the environment.
        num_outputs: int
            The number of outputs of the model.

        model_config: Dict
            The configuration of the model as passed to the rlib trainer.
            Besides the rllib model parameters, should contain a sub-dict
            custom_model_config that stores the boolean for "use_parametric"
            and "env_obs_name" for the name of the observation.
        name: str
            The name of the model captured in model_config["model_name"]
        """

        # Call the parent constructor.
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Fetch the network specification
        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
            model_config.get("post_fcnet_hiddens", [])
        )

        self.vf_share_layers = model_config.get("vf_share_layers")
        self.parametric_action_space = model_config["custom_model_config"].get(
            "use_parametric", False
        )
        self.env_obs_name = model_config["custom_model_config"].get(
            "env_obs_name", "grid"
        )
        if not isinstance(self.env_obs_name, list):
            self.env_obs_name = [self.env_obs_name]
        logging.info(
            f"Using parametric action space equals {self.parametric_action_space}"
        )

        layers = []
        if self.parametric_action_space:  # do not parametrize the action mask
            prev_layer_size = int(
                np.prod(np.array(obs_space.shape)) - action_space.n
            )  # dim of the observation space
        else:
            prev_layer_size = int(
                np.prod(np.array(obs_space.shape))
            )  # dim of the observation space

        # Create hidden layers
        for size in hiddens:
            layers += [nn.Linear(prev_layer_size, size), nn.ReLU(inplace=True)]
            prev_layer_size = size

        self._logits = nn.Linear(prev_layer_size, num_outputs)
        self._hidden_layers = nn.Sequential(*layers)

        # Value function spec
        self._value_branch_separate = None
        if not self.vf_share_layers:  # if we want to separate value function
            # Build a parallel set of hidden layers for the value net.
            if self.parametric_action_space:  # do not parametrize the action mask
                prev_vf_layer_size = int(
                    np.prod(np.array(obs_space.shape)) - action_space.n
                )  # dim of the observation space
            else:
                prev_vf_layer_size = int(np.prod(np.array(obs_space.shape)))
            vf_layers = []
            for size in hiddens:
                vf_layers += [
                    nn.Linear(prev_vf_layer_size, size),
                    nn.ReLU(inplace=True),
                ]
                prev_vf_layer_size = size
            self._value_branch_separate = nn.Sequential(*vf_layers)

        self._value_branch = nn.Linear(prev_layer_size, 1)

        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """
        Forward pass of the MLP model.

        Args:
            input_dict (Dict[str, TensorType]): A dictionary containing the input tensors.
            state (List[TensorType]): A list of state tensors.
            seq_lens (TensorType): Tensor representing the sequence lengths.

        Returns:
            Tuple[TensorType, List[TensorType]]: A tuple containing the logits tensor and the updated state list.
        """
        if self.parametric_action_space:
            # Change incompatible with flat parametric action space
            regular_obs = torch.concat(
                list(input_dict["obs"]["regular_obs"].values()), dim=1
            )
            chosen_sub = input_dict["obs"]["chosen_substation"]
            obs = torch.cat([regular_obs, chosen_sub], dim=1)  # [BATCH_DIM, obs_dim]
            inf_mask = torch.clamp(
                torch.log(input_dict["obs"]["action_mask"]), FLOAT_MIN, FLOAT_MAX
            )
        else:
            if isinstance(input_dict["obs_flat"], OrderedDict):
                # logging.warning("applying custom flattening")
                # Flatten the dictionary and convert to a tensor
                obs = torch.cat(list(input_dict["obs_flat"].values()), dim=-1).float()
                if obs.ndim == 1:  # edge case of batch size 1
                    obs = obs.unsqueeze(0)
            else:
                obs = input_dict["obs_flat"].float()

        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)

        logits = self._logits(self._features)
        if self.parametric_action_space:
            logits += inf_mask

        if (torch.isnan(logits).any().item()) or (torch.isinf(logits).any().item()):
            logging.warning("Logits contain NaN values")
        return logits, state

    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            return self._value_branch(
                self._value_branch_separate(self._last_flat_in)
            ).squeeze(1)
        return self._value_branch(self._features).squeeze(1)

    def import_from_h5(self, h5_file: str) -> None:
        """Imports weights from an h5 file.

        Args:
            h5_file: The h5 file name to import weights from.

        Example:
            >>> from ray.rllib.algorithms.ppo import PPO
            >>> algo = PPO(...)  # doctest: +SKIP
            >>> algo.import_policy_model_from_h5("/tmp/weights.h5") # doctest: +SKIP
            >>> for _ in range(10): # doctest: +SKIP
            >>>     algo.train() # doctest: +SKIP
        """
        raise NotImplementedError


class SingleAgentCallback(DefaultCallbacks):
    """Implements custom callbacks metric for single agent."""

    def on_episode_end(
        self,
        *,
        episode: Any,
        worker: Any = None,
        base_env: BaseEnv = None,
        policies: Any = None,
        env_index: Any = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Callback function called at the end of each episode.

        Args:
            episode (Episode): The episode object containing information about the episode.
            worker (RolloutWorker): The worker object responsible for executing the episode.
            base_env (Environment): The base environment used for the episode.
            policies (Dict[str, Policy]): A dictionary of policies used for the episode.
            env_index (int): The index of the environment in the multi-environment setup.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        # Make sure this episode is really done.
        episode.custom_metrics["num_env_steps"] = episode.last_info_for()["steps"]


class CustomDiscreteActions(Discrete):
    """
    Class that customizes the action space.

    Example usage:

    import grid2op
    from grid2op.Converter import IdToAct

    env = grid2op.make("rte_case14_realistic")

    all_actions = # a list of of desired actions
    converter = IdToAct(env.action_space)
    converter.init_converter(all_actions=all_actions)


    env.action_space = ChooseDiscreteActions(converter=converter)


    """

    def __init__(self, converter: Any):
        """init"""
        self.converter = converter
        Discrete.__init__(self, n=converter.n)

    def from_gym(self, gym_action: int) -> BaseAction:
        """from_gym"""
        return self.converter.convert_act(gym_action)

    def close(self) -> None:
        """close"""


class ScaledL2RPNReward(L2RPNReward):
    """
    Scaled version of L2RPNReward such that the reward falls between 0 and 1.
    Additionally -0.5 is awarded for illegal actions.
    """

    def __init__(self, logger: Any = None):
        super().__init__()
        self.reward_min: float = -0.5
        self.reward_illegal: float = -0.5
        self.reward_max: float = 1.0
        self.num_lines: int

    def initialize(self, env: BaseEnv) -> None:
        """
        inits
        """
        self.num_lines = env.backend.n_line

    def __call__(
        self,
        action: BaseAction,
        env: BaseEnv,
        has_error: bool,
        is_done: bool,
        is_illegal: bool,
        is_ambiguous: bool,
    ) -> float:
        """
        Calculate the reward based on the given parameters.

        Parameters:
        - action: The action taken in the environment.
        - env: The environment object.
        - has_error: Flag indicating if there was an error in the environment.
        - is_done: Flag indicating if the episode is done.
        - is_illegal: Flag indicating if the action is illegal.
        - is_ambiguous: Flag indicating if the action is ambiguous.

        Returns:
        - res: The calculated reward.
        """
        if not is_done and not has_error:
            line_cap = self.__get_lines_capacity_usage(env)
            res = np.sum(line_cap) / self.num_lines
        else:
            # no more data to consider, no powerflow has been run, reward is what it is
            res = self.reward_min
        return res

    @staticmethod
    def __get_lines_capacity_usage(env: BaseEnv) -> Any:
        """
        Calculate the lines capacity usage score for the given environment.

        Parameters:
        - env: The environment object.

        Returns:
        - lines_capacity_usage_score: The lines capacity usage score as a numpy array.
        """
        ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        thermal_limits += 1e-1  # for numerical stability
        relative_flow = np.divide(ampere_flows, thermal_limits, dtype=dt_float)

        min_flow = np.minimum(relative_flow, dt_float(1.0))
        lines_capacity_usage_score = np.maximum(
            dt_float(1.0) - min_flow**2, np.zeros(min_flow.shape, dtype=dt_float)
        )
        return lines_capacity_usage_score


class CustomGymEnv(GymEnv):
    """Implements a grid2op env in gym."""

    def __init__(self, env: BaseEnv):
        super().__init__(env)
        self.idx = 0
        self.reconnect_line = []
    
    #make the change in the rewards generic#
    def set_rewards(self, rewards_list):
        self.rewards = rewards_list

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Any:
        """resets"""
        g2op_obs = self.init_env.reset()
        return self.observation_space.to_gym(g2op_obs)

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """steps"""
        g2op_act = self.action_space.from_gym(action)

        if self.reconnect_line:
            for line in self.reconnect_line:
                g2op_act = g2op_act + line

            self.reconnect_line = []
        #create reward array
        g2op_obs, reward1, done, info = self.init_env.step(g2op_act)
        len(self.rewards)
        reward = np.array([reward1, info['rewards'][self.rewards[0]], info['rewards'][self.rewards[1]]])

        # to_reco = ~g2op_obs.line_status
        # self.reconnect_line = []
        # if np.any(to_reco):
        #     reco_id = np.where(to_reco)[0]
        #     for line_id in reco_id:
        #         g2op_act = self.init_env.action_space(
        #             {"set_line_status": [(line_id, +1)]}
        #         )

        #         self.reconnect_line.append(g2op_act)

        # specifically opponent
        if info["opponent_attack_duration"] == 1:
            line_id_attacked = np.argwhere(info["opponent_attack_line"]).flatten()[0]
            g2op_act = self.init_env.action_space(
                {"set_line_status": [(line_id_attacked, 1)]}
            )
            self.reconnect_line.append(g2op_act)

        gym_obs = self.observation_space.to_gym(g2op_obs)
        return gym_obs, reward, done, info


class GridGym(gym.Env):
    """
    loads the grid2op env into gym
    """

    def __init__(self, env_config: dict[str, Any]):
        # create gym env
        self.grid2op_env = grid2op.make(
            "rte_case14_realistic",
            reward_class=ScaledL2RPNReward,
            backend=LightSimBackend(),
            opponent_attack_cooldown=144,
            opponent_attack_duration=48,
            opponent_budget_per_ts=(48 / 144) + 1e-5,
            opponent_init_budget=144,
            opponent_action_class=PowerlineSetAction,
            opponent_class=RandomLineOpponent,
            opponent_budget_class=BaseActionBudget,
            kwargs_opponent={
                "lines_attacked": [
                    "3_4_6",
                    "11_12_13",
                    "3_6_15",
                    "3_8_16",
                    "6_8_19",
                ]
            },
        )

        thermal_limits = [
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            760,
            450,
            760,
            380,
            380,
            760,
            380,
            760,
            380,
            380,
            380,
            2000,
            2000,
        ]
        self.grid2op_env.set_thermal_limit(thermal_limits)

        self.env_gym = CustomGymEnv(self.grid2op_env)

        # define observations
        self.env_gym.observation_space = self.env_gym.observation_space.keep_only_attr(
            [
                "rho",
                "gen_p",
                "load_p",
                "topo_vect",
                "p_or",
                "p_ex",
                "timestep_overflow",
            ]
        )

        # scale observations
        self.env_gym.observation_space = self.env_gym.observation_space.reencode_space(
            "gen_p",
            ScalerAttrConverter(substract=0.0, divide=self.grid2op_env.gen_pmax),
        )
        self.env_gym.observation_space = self.env_gym.observation_space.reencode_space(
            "timestep_overflow",
            ScalerAttrConverter(
                substract=0.0,
                divide=Parameters().NB_TIMESTEP_OVERFLOW_ALLOWED,  # assuming no custom params
            ),
        )

        for attr in ["p_ex", "p_or", "load_p"]:
            underestimation_constant = (
                1.2  # constant to account that our max/min are underestimated
            )
            max_arr, min_arr = np.load(
                os.path.join(
                    WORKING_DIR,
                    "scaling_arrays",
                    "rte_case14_realistic",
                    f"{attr}.npy",
                )
            )

            self.env_gym.observation_space = (
                self.env_gym.observation_space.reencode_space(
                    attr,
                    ScalerAttrConverter(
                        substract=underestimation_constant * min_arr,
                        divide=underestimation_constant * (max_arr - min_arr),
                    ),
                )
            )

        self.observation_space = gym.spaces.Dict(
            dict(self.env_gym.observation_space.items())
        )

        # define actions from medha
        path = os.path.join(
            WORKING_DIR,
            "action_spaces/rte_case14_realistic/",
            "medha_DN_onechange.json",
        )
        with open(path, "rt", encoding="utf-8") as action_set_file:
            self.all_actions = list(
                (
                    self.grid2op_env.action_space(action_dict)
                    for action_dict in json.load(action_set_file)
                )
            )

        # add do nothing action
        do_nothing_action = self.grid2op_env.action_space({})
        self.all_actions.insert(0, do_nothing_action)

        converter = IdToAct(
            self.grid2op_env.action_space
        )  # initialize with regular the environment of the regular action space
        converter.init_converter(all_actions=self.all_actions)

        self.env_gym.action_space = CustomDiscreteActions(converter=converter)
        self.action_space = gym.spaces.Discrete(self.env_gym.action_space.n)

        # set parameters
        self.steps = 0
        self.rho_threshold = env_config["rho_threshold"]

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Resets the environment and returns the initial observation.

        Args:
            seed (int, optional): Random seed for the environment. Defaults to None.
            options (dict, optional): Additional options for resetting the environment. Defaults to None.

        Returns:
            tuple: A tuple containing the initial observation and an empty dictionary.
        """
        obs = self.env_gym.reset()

        # additional loop for if it completes it immediately # NOTE: Latest addition
        done = True
        while done:
            # find first step that surpasses threshold
            done = False
            self.steps = 0
            while (max(obs["rho"]) < self.rho_threshold) and (not done):
                obs, _, done, _ = self.env_gym.step(0)
                # obs, _, done, _, _ = self.env_gym.step(0)
                # obs, _, done, _ = self.env_gym.step(self.do_nothing_actions[0])
                self.steps += 1
        return obs, {}

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """steps"""
        cum_reward: float = 0.0
        obs, reward, done, info = self.env_gym.step(action)
        self.steps += 1
        cum_reward += reward
        while (max(obs["rho"]) < self.rho_threshold) and (not done):
            obs, reward, done, _ = self.env_gym.step(0)
            self.steps += 1
            cum_reward += reward

        if done:
            info["steps"] = self.steps
        return obs, cum_reward, done, False, info

    def render(self) -> None:
        """renders"""
        raise NotImplementedError("render not implemented")

