"""Training module for poker strategies using counterfactual regret minimization."""

from summerhillpoker.training.cfr import CFRTrainer, DeepCFR
from summerhillpoker.training.reinforcement import ReinforcementLearner
from summerhillpoker.training.dqn import DQNTrainer, DQNAgent

__all__ = [
    'CFRTrainer',
    'DeepCFR',
    'ReinforcementLearner',
    'DQNTrainer',
    'DQNAgent',
]