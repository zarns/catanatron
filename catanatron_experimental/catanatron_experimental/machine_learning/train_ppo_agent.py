import math
import os
os.environ["WANDB_DISABLE_SYMLINKS"] = "True"
from typing import Any
import atexit

import gymnasium as gym
import torch as th
from torch import nn
import numpy as np
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
# from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import wandb
from wandb.integration.sb3 import WandbCallback

from catanatron import Color
from catanatron_experimental.machine_learning.players.value import (
    ValueFunctionPlayer,
)
from catanatron.state_functions import get_actual_victory_points


LOAD = False


def mask_fn(env) -> np.ndarray:
    valid_actions = env.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[valid_actions] = 1
    return np.array([bool(i) for i in mask])


def build_partial_rewards(vps_to_win):
    def partial_rewards(game, p0_color):
        winning_color = game.winning_color()
        if winning_color is None:
            return 0

        total = 0
        if p0_color == winning_color:
            total += 0.80
        else:
            total -= 0.80
        enemy_vps = [
            get_actual_victory_points(game.state, color)
            for color in game.state.colors
            if color != p0_color
        ]
        enemy_avg_vp = sum(enemy_vps) / len(enemy_vps)
        my_vps = get_actual_victory_points(game.state, p0_color)
        # can at most win by vps_to_win - 1
        vp_diff = (my_vps - enemy_avg_vp) / (vps_to_win - 1)

        total += 0.20 * vp_diff
        return total

    return partial_rewards


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        cnn1=256,
        cnn2=128,
        features_dim: int = 256,
    ):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space["board"].shape[0]
        self.cnn = nn.Sequential(
            # nn.Conv2d(n_input_channels, 32, kernel_size=5),
            nn.Conv2d(n_input_channels, cnn1, (5, 3), stride=(2, 2)),
            nn.BatchNorm2d(cnn1),
            nn.Tanh(),
            nn.Conv2d(cnn1, cnn2, 3),  # image before is 21 - 5 + 1 = 17, 11 - 3 + 1 = 9
            nn.BatchNorm2d(cnn2),
            nn.Tanh(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            tensor = th.as_tensor(observation_space.sample()["board"][None]).float()
            n_flatten = self.cnn(tensor).shape[1]

        n_numeric_features = observation_space["numeric"].shape[0]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + n_numeric_features, features_dim), nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        board_features = self.cnn(observations["board"])
        concatenated_tensor = th.cat([board_features, observations["numeric"]], dim=1)
        return self.linear(concatenated_tensor)


# TODO:
#   Try different learning rates
#   Try bigger net
#   vectorization env
#   use of GPU
#   Add L2 Regularization.
#   self-play (increasing difficulty)
#   Hyperparameter tuning
# More info on metrics here: https://stable-baselines3.readthedocs.io/en/master/common/logger.html?highlight=logger#explanation-of-logger-output
# Try schedule on LR. 1.5M with 3e-5, then go to 3e-6.


def main():
    # ===== Params:
    total_timesteps = 1_000_000  # 1,000,000 takes around ~1:41:58.88
    cnn_arch = [1024, 512, 256]
    net_arch = [dict(vf=[128, 128], pi=[128, 128])]
    activation_fn = th.nn.Tanh
    lr = 3e-5
    vps_to_win = 10
    env_name = "catanatron_gym:catanatron-v1"
    map_type = "BASE"
    enemies = [ValueFunctionPlayer(Color.RED)]
    reward_function = build_partial_rewards(vps_to_win)
    representation = "vector"
    batch_size = 4096
    gamma = 0.99
    normalized = False
    selfplay = False

    # ===== Build Experiment Name
    iters = round(math.log(total_timesteps, 10))
    arch_str = (
        activation_fn.__name__
        + "x".join([str(i) for i in net_arch[:-1]])
        + "+"
        + "vf="
        + "x".join([str(i) for i in net_arch[-1]["vf"]])
        + "+"
        + "pi="
        + "x".join([str(i) for i in net_arch[-1]["pi"]])
    )
    if representation == "mixed":
        arch_str = "Cnn" + "x".join([str(i) for i in cnn_arch]) + arch_str
    enemy_desc = "".join(e.__class__.__name__ for e in enemies)
    experiment_name = f"ppo-{selfplay}-{normalized}-{iters}-{batch_size}-{gamma}-{enemy_desc}-{reward_function.__name__}-{representation}-{arch_str}-{lr}lr-{vps_to_win}vp-{map_type}map"
    path = os.path.join("models", experiment_name)
    print(experiment_name)

    # WandB config =====
    config = {
        "learning_rate": lr,
        "total_timesteps": total_timesteps,
        "net_arch": net_arch,
        "activation_fn": activation_fn,
        "vps_to_win": vps_to_win,
        "map_type": map_type,
        "enemies": enemies,
        "reward_function": reward_function,
        "representation": representation,
        "batch_size": batch_size,
        "gamma": gamma,
        "normalized": normalized,
        "cnn_arch": cnn_arch,
        "selfplay": selfplay,
        "experiment_name": experiment_name,
    }
    run = wandb.init(
        project="catanatron",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )

    def print_name():
        print(experiment_name)

    atexit.register(print_name)

    # Init Environment and Model
    env = gym.make(
        env_name,
        config={
            "map_type": map_type,
            "vps_to_win": vps_to_win,
            "enemies": enemies,
            "reward_function": reward_function,
            "representation": representation,
            "normalized": True,
        },
    )
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking

    # Print the observation space to verify its type
    print("Observation Space:", env.observation_space)
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        model = MaskablePPO.load(path, env)
        print("Loaded", experiment_name)
    except Exception as e:
        print(f"Failed to load the model from {path}: {e}")
        print("Creating a new model.")
        policy_kwargs: Any = dict(activation_fn=activation_fn, net_arch=net_arch)
        if representation == "mixed":
            policy_kwargs["features_extractor_class"] = CustomCNN
            policy_kwargs["features_extractor_kwargs"] = dict(features_dim=512)
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            gamma=gamma,
            batch_size=batch_size,
            policy_kwargs=policy_kwargs,
            learning_rate=lr,
            verbose=1,
            tensorboard_log="./logs/mppo_tensorboard/" + experiment_name,
        )

    # Save a checkpoint every 1000 steps
    wandb_callback = WandbCallback(
        model_save_path=f"models/{run.id}",
        # save_model=False
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000, save_path="./logs/", name_prefix=experiment_name
    )
    callback = CallbackList([checkpoint_callback, wandb_callback])

    if selfplay:
        selfplay_iterations = 10
        for i in range(selfplay_iterations):
            breakpoint()
            model.learn(
                total_timesteps=int(total_timesteps / selfplay_iterations),
                callback=callback,
            )
            model.save(path)
    else:
        model.learn(total_timesteps=total_timesteps, callback=callback)
        model.save(path)
        # Save the model manually to the desired path
        model.save("C:/Users/mason/programming/catanatron/model.zip")

    run.finish()


if __name__ == "__main__":
    main()