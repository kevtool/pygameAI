from dataclasses import dataclass

@dataclass
class EarlyStoppingConfig:
    init_threshold: float
    test_threshold: float
    test_episodes: int