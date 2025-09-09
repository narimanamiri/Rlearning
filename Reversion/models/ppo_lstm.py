from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
from gymnasium import spaces

class LstmFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super(LstmFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.window_size = observation_space.shape[0]
        self.feature_dim = observation_space.shape[1]
        self.lstm = nn.LSTM(self.feature_dim, 64, batch_first=True)
        self.linear = nn.Linear(64, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        lstm_out, _ = self.lstm(observations)
        last_out = lstm_out[:, -1, :]
        return self.linear(last_out)

def make_ppo_lstm_policy(env, config):
    policy_kwargs = dict(
        features_extractor_class=LstmFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    model = PPO(
        "MlpLstmPolicy",
        env,
        learning_rate=config['model']['learning_rate'],
        n_steps=config['model']['n_steps'],
        batch_size=config['model']['batch_size'],
        n_epochs=config['model']['n_epochs'],
        gamma=config['model']['gamma'],
        gae_lambda=config['model']['gae_lambda'],
        clip_range=config['model']['clip_range'],
        ent_coef=config['model']['ent_coef'],
        max_grad_norm=config['model']['max_grad_norm'],
        verbose=1,
        seed=config['model']['seed'],
        policy_kwargs=policy_kwargs
    )
    return model