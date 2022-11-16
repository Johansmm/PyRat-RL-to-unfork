# Author: Johan Mejia
# Challenge on Reinforcement learning
# The goal of this challenge is used reinforcement learning to train an agent to play PyRat.
# We perform Q-Learning using a simple regressor to predict the q-values associated with each
# of the four possible movements.
# This regressor is implemented with pytorch

# Usage : python qlearn.py
# See all documentation with the commamnd `python qlearn.py -h``

# When training is finished, use AIs/agentRL.py to play a pyrat game with the
# appropriate parameters.

from argparse import ArgumentParser

from AIs.models import PerceptronLit, RLTrainer
from AIs.modules import RLLitDataModule
from simulator import PyRat


def parse_args():
    parser = ArgumentParser("Train a perceptron module using RL q-learn approach")
    parser = RLTrainer.add_argparse_args(parser)
    parser = RLLitDataModule.add_argparse_args(parser)
    parser = PyRat.add_argparse_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    """CLI to train a model with RL Q-learning"""
    args = parse_args()

    # Define model through a sample of the enviroment
    enviroment = PyRat.from_argparse_args(args, opponent_reset=True)
    model = PerceptronLit(x_sample=enviroment.observe())
    train_loader = RLLitDataModule.from_argparse_args(args)

    # Define trainer and launch the training
    trainer = RLTrainer.from_argparse_args(args, enviroment=enviroment)
    trainer.fit(model, train_loader)
