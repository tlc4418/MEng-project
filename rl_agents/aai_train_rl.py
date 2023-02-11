import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import gym
import random
import jax
import jax.dlpack
import yaml
import pickle
import torch as th
import torch.utils.dlpack
import haiku as hk
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import math

from pathlib import Path
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import DQN
from gym_unity.envs import UnityToGymWrapper
from animalai.envs.environment import AnimalAIEnvironment
from slot_attention_and_alignnet.src.utils import forward_fn
from slot_attention_and_alignnet.src.models import (
    SlotAttentionAE,
    AlignedSlotAttention,
    BGSlotAttentionAE,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor


class objdict(dict):
    def __init__(self, dict):
        if dict is not None:
            [self.__setattr__(k, v) for k, v in dict.items()]

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        obs = (observations - 0.5) * 2.0
        ret = self.linear(self.cnn(obs))
        return ret


class SlotAttentionExtractor(BaseFeaturesExtractor):
    """
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Load SA model
        model_class = SlotAttentionAE
        cfg = objdict(
            yaml.safe_load(
                open(
                    Path(
                        "/media/home/thomas/thomas_files/slot_attention_and_alignnet/config/slot_attention_clevr.yaml"
                    )
                )
            )
        )
        self.rngseq = hk.PRNGSequence(40)
        net = hk.transform_with_state(jax.partial(forward_fn, net=model_class, cfg=cfg))
        with open(
            f"/media/home/thomas/thomas_files/slot_attention_and_alignnet/runs/sa_aai_goals_closer_mlp_48_run_6/model/params_220000.pkl",
            "rb",
        ) as f:
            params, state = pickle.load(f)
        self.fixed_net_apply = jax.jit(
            lambda rng, im: net.apply(params, state, rng, im, True)
        )

        ## If we want an additional linear nn
        ## Compute shape by doing one forward pass
        # sample_inp = jnp.asarray(
        #     observation_space.sample()[None], dtype=float
        # ).transpose((0, 2, 3, 1))
        # out, _ = self.fixed_net_apply(next(self.rngseq), sample_inp)
        # concat_out = jnp.concatenate(out["slots"].squeeze(), axis=None)

        ## Linear nn from concat slot dims to feature dims
        # self.linear = nn.Sequential(
        #     nn.Linear(concat_out.shape[0], features_dim), nn.ReLU()
        # )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Pre-process observations
        inp = jax.dlpack.from_dlpack(
            torch.utils.dlpack.to_dlpack(observations.permute((0, 2, 3, 1)))
        )
        inp = (jnp.array(inp) - 0.5) * 2.0

        # Pass into model
        slots = self.fixed_net_apply(next(self.rngseq), inp)[0]
        slots = slots["slots"]

        # Concat slots and return
        B, _, _ = slots.shape
        slots = jnp.reshape(slots, (B, -1))
        slots = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(slots))

        ## Other option for concatenation
        # slots = jnp.sum(slots, axis=1, keepdims=False)
        # slots = self.linear(slots)
        return slots


class AlignNetExtractor(BaseFeaturesExtractor):
    """
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        seq_length: int = 5,
    ):
        super().__init__(observation_space, features_dim)
        self.seq_len = seq_length

        # Load SA+AlignNet model
        model_class = AlignedSlotAttention
        cfg = objdict(
            yaml.safe_load(
                open(
                    Path(
                        "/media/home/thomas/thomas_files/slot_attention_and_alignnet/config/alignnet_base.yaml"
                    )
                )
            )
        )
        self.rngseq = hk.PRNGSequence(42)
        net = hk.transform_with_state(jax.partial(forward_fn, net=model_class, cfg=cfg))
        with open(
            f"/media/home/thomas/thomas_files/slot_attention_and_alignnet/runs/align_sprite_run_32/model/params_100000.pkl",
            "rb",
        ) as f:
            params, state = pickle.load(f)
        self.fixed_net_apply = jax.jit(
            lambda rng, im: net.apply(params, state, rng, im, True)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Pre-process observations
        inp = np.swapaxes(
            np.array(
                np.split(
                    np.array(
                        jax.dlpack.from_dlpack(
                            torch.utils.dlpack.to_dlpack(
                                observations.permute((0, 2, 3, 1))
                            )
                        )
                    ),
                    self.seq_len,
                    -1,
                )
            ),
            0,
            1,
        )
        inp = (jnp.array(inp) - 0.5) * 2.0

        # Pass into model
        slots = self.fixed_net_apply(next(self.rngseq), inp)[0]
        slots = slots["slots"]

        # Concat slots and return as tensor
        B, _, _ = slots.shape
        slots = jnp.reshape(slots, (B // self.seq_len, -1))
        slots = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(slots))
        return slots


def train_agent_single_config(configuration_file):
    runname = "RL_DQN_SA_multi_3_final"
    aai_env = AnimalAIEnvironment(
        seed=123,
        file_name="aai_environment/env/AnimalAI",
        arenas_configurations=configuration_file,
        play=False,
        base_port=5002,
        inference=False,
        useCamera=True,
        resolution=128,
        useRayCasts=False,
        decisionPeriod=5,
    )

    env = UnityToGymWrapper(
        aai_env, uint8_visual=True, allow_multiple_obs=False, flatten_branched=True
    )

    # For observation frame stacking
    # env = Monitor(env, "./dqn_tensorboard/")
    # env = DummyVecEnv([lambda: env])
    # env = VecFrameStack(env, n_stack=5)

    # Control which feature extractor and model to use
    policy_kwargs = dict(
        features_extractor_class=SlotAttentionExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )
    model = DQN(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=("./dqn_tensorboard/RL_SA"),
        exploration_fraction=0.5,
        exploration_final_eps=0.1,
        target_update_interval=500,
        buffer_size=50000,
        learning_starts=20000,
        learning_rate=1e-4,
        batch_size=32,
    )

    no_saves = 20
    no_steps = 300000
    reset_num_timesteps = True
    for i in range(no_saves):
        model.learn(no_steps, reset_num_timesteps=reset_num_timesteps)
        model.save("rl_agents/results/" + runname + "/model_" + str((i + 1) * no_steps))
        reset_num_timesteps = False
    env.close()


# Loads a random competition configuration unless a link to a config is given as an argument.
if __name__ == "__main__":
    if len(sys.argv) > 1:
        configuration_file = sys.argv[1]
    else:
        competition_folder = "aai_environment/configs/competition/"
        configuration_files = os.listdir(competition_folder)
        configuration_random = random.randint(0, len(configuration_files))
        configuration_file = "rl_agents/rl_train_config.yaml"
    train_agent_single_config(configuration_file=configuration_file)
