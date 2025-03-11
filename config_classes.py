from dataclasses import dataclass

@dataclass
class EarlyStoppingConfig:
    init_threshold: float
    init_episodes: int
    test_threshold: float
    test_episodes: int
    cooldown: int

@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    epsilon_decay: float = 0.998
    epsilon_min: float = 0.1
    target_update_freq: int = 10
    epsilon_decay_type: str = "timestep"
    episodes_per_decay: int = 1
    timesteps_per_decay: int = 50
    buffer_size: int = 12000
    batch_size: int = 32