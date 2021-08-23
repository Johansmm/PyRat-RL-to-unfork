# Author: Carlos Lassance, Myriam Bontonou, Nicolas Farrugia, Johan Mejia
# Challenge on Reinforcement learning
# The goal of this challenge is used reinforcement learning to train an agent to play PyRat.
# We perform Q-Learning using a simple regressor to predict the Q-values associated with each
# of the four possible movements.
# This regressor is implemented with pytorch

# Usage : python main.py
# Change the parameters directly in this file.

# When training is finished, copy both the AIs/agentRL.py file and the saves folder into your
# pyrat folder, and run a pyrat game with the appropriate parameters using the agentRL.py as AI.

# The game.py file describes the simulation environment, including the generation of reward
# and the observation that is fed to the agent.

# The rl.py file describes the reinforcement learning procedure, including Q-learning,
# Experience replay, and a pytorch model to learn the Q-function.
# SGD is used to approximate the Q-function.

import sys
import os
import json
import numpy as np
import time
import random
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from .. import simulator as game
import rl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# This set of parameters can be changed in your experiments.
# Definitions :
# - An iteration of training is called an Epoch. It correspond to a full play of a PyRat game.
# - An experience is a set of vectors < s, a, r, s'> describing the consequence of being in state s,
# doing action a and receiving reward r.
# Look at the file rl.py to see how the experience replay buffer is implemented.
# - A batch is a set of experiences we use for training during one epoch. We draw batches from
# the experience replay buffer.

epoch = 10000  # Total number of epochs that will be done
max_memory = 1000  # Maximum number of experiences we are storing
number_of_batches = 8  # Number of batches per epoch
batch_size = 64  # Number of experiences we use for training per batch
width = 21  # Size of the playing field
height = 15  # Size of the playing field
cheeses = 40  # Number of cheeses in the game


# Actions take 1 - p_value times or randomly
def p_value(win_rate): return 0.8 * np.exp(-0.02 * win_rate)


# If load, then the last saved result is loaded and training is continued.
# Otherwise, training is performed from scratch starting with random parameters.
load = False
save = False


def choose_action(model, obs):
    with torch.no_grad():
        logits = model(obs)
        action = torch.argmax(logits, dim=1).cpu().item()
    return action


env = game.PyRat(width=width, height=height, cheeses=cheeses, start_random=False)
exp_replay = rl.ExperienceReplay(max_memory=max_memory, discount=0.85)
model = rl.NLinearModels(env.observe()[0], weights_file='save_rl/qlearning_cami.pt')
print("[INFO] Input size and model", env.observe()[0].shape, model)
print("[INFO] Model save in: ", model.weights_file)

# There are a few methods (= functions) that you should look into.
# For the game.Pyrat() class, you should check out the .observe and the .act methods which
# will be used to get the environment, and send an action.
# For the exp_replay() class, you should check the remember and get_batch methods.
# For the NLinearModels class, you should check the train_on_batch function for training, and
# the forward method for inference (to estimate all pi values given a state).

if load:
    model.load()
model.to(device)
or_cheeses = cheeses


def play(model, epochs, train=True):
    win_cnt = 0
    lose_cnt = 0
    draw_cnt = 0
    win_hist = []
    cheeses = []
    steps = 0.
    last_W = 0
    last_D = 0
    last_L = 0
    real = False
    losses = []

    # Define a loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-1)
    # optimizer = optim.Adam(model.parameters(), lr = 5e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 2000, gamma=0.8)
    prob = [1, 1, 1, 1]  # Intial probabilities
    p_val = p_value(100) if load else p_value(0)
    print(p_val)
    if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()  # clear if it exists
    for e in tqdm(range(epochs)):
        env.reset()
        game_over = False

        # Get the current state of the environment
        current_st = torch.FloatTensor(env.observe()).to(device)

        # Play a full game until game is over
        model.eval()
        rewards, actions = [], []
        step, count = 0, 0
        while not game_over:
            # Do not forget to transform the input of model into torch tensor
            previous_st = current_st

            # Predict the Q value for the current state
            action = choose_action(model, previous_st)
            if train and (p_val * p_value(step) > random.uniform(0, 1) and count < 7):
                action = random.choices(range(4), weights=prob, k=1)[0]
                count += 1  # Only five random moves par game

            # Apply action, get rewards and new state
            current_st, reward, game_over = env.act(action)
            current_st = torch.FloatTensor(current_st).to(device)
            rewards.append(reward)
            actions.append(action)

            # Statistics
            if game_over:
                steps += env.round
                if env.score > env.enemy_score:
                    win_cnt += 1
                elif env.score == env.enemy_score:
                    draw_cnt += 1
                else:
                    lose_cnt += 1
                cheese = env.score

            # Create an experience array using previous state, the performed action, the obtained
            # reward and the new state. The vector has to be in this order.
            # Store in the experience replay buffer an experience and end game.
            # Do not forget to transform the previous state and the new state into torch tensor.
            exp_replay.remember([previous_st, action, reward, current_st], game_over)
            step += 1

        # Statistics
        win_hist.append(win_cnt)
        cheeses.append(cheese)
        if train:
            # Train using experience replay. For each batch, get a set of experiences
            # (state, action, new state) that were stored in the buffer.
            # Use this batch to train the model.
            loss = 0.
            model.train()
            for _ in range(number_of_batches):
                inputs, targets = exp_replay.get_batch(model, batch_size, device=device)
                loss += rl.train_on_batch(model, inputs, targets, criterion, optimizer)
            loss /= number_of_batches
            losses.append(loss)
            if 'scheduler' in locals() and scheduler is not None:
                scheduler.step()

        if (e + 1) % 100 == 0:  # Statistics every 100 epochs
            cheese_np = np.array(cheeses)
            mem_actions = np.eye(4)[exp_replay.get_actions()].sum(axis=0)
            prob = mem_actions.sum() - mem_actions  # Search new actions!
            p_val = p_value((win_cnt - last_W) + (draw_cnt - last_D))
            print(prob)
            print("[INFO] Actions in memory: ", mem_actions)
            print("[INFO] Actions current game: ", actions)
            print("[INFO] Rewards current game: ", rewards)
            print("[INFO] Total rewards: ", sum(exp_replay.get_rewards()))
            print("[INFO] P_value = ", p_val)
            if not train:
                real = True
            # if real:
            #     env.cheeses = or_cheeses
            #     real = False
            # else:
            #     env.cheeses = random.choice(range(or_cheeses-10, or_cheeses + 3))
            #     real = True
            print("[INFO] New cheeses: ", env.cheeses)

            if train:
                string = "Epoch {:03d}/{:03d} | Losses {:0.6f} | Cheese count {} | "
                "Last 100 Cheese {}| W/D/L {}/{}/{} | 100 W/D/L {}/{}/{} | 100 Steps {}"
                string = string.format(
                    e, epochs, loss, cheese_np.sum(),
                    cheese_np[-100:].sum(), win_cnt, draw_cnt, lose_cnt,
                    win_cnt - last_W, draw_cnt - last_D, lose_cnt - last_L, steps / 100)
            else:
                string = "Epoch {:03d}/{:03d} | Cheese count {} | Last 100 Cheese {}| "
                "W/D/L {}/{}/{} | 100 W/D/L {}/{}/{} | 100 Steps {}"
                string = string.format(
                    e, epochs, cheese_np.sum(),
                    cheese_np[-100:].sum(), win_cnt, draw_cnt, lose_cnt,
                    win_cnt - last_W, draw_cnt - last_D, lose_cnt - last_L, steps / 100)

            steps = 0.
            last_W = win_cnt
            last_D = draw_cnt
            last_L = lose_cnt
            print(string)

    if train:
        name = os.path.split(model.weights_file.replace(
            '.pt', '.pkl').replace('weights_', 'losses'))[-1]
        pickle.dump([losses], open(name, "wb"))


print("Training")
play(model, epoch, True)
if save:
    model.save()
print("Training done")
print("Testing")
play(model, 1000 if epoch > 1000 else epoch, False)
print("Testing done")
