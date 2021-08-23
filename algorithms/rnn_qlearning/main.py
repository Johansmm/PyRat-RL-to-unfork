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

epoch = 3000  # Total number of epochs that will be done
max_memory = 1000  # Maximum number of experiences we are storing
number_of_batches = 16  # Number of batches per epoch
batch_size = 64  # Number of experiences we use for training per batch
width = 10  # Size of the playing field
height = 10  # Size of the playing field
cheeses = 15  # Number of cheeses in the game
past = 1  # Number of past to predict


# Actions take 1 - p_value times or randomly
def p_value(win_rate): return 0.0 * np.exp(-0.03 * win_rate)


# If load, then the last saved result is loaded and training is continued.
# Otherwise, training is performed from scratch starting with random parameters.
load = True
save = True


def choose_action(model, obs):
    with torch.no_grad():
        logits = model(obs)
        # logits = model(obs, model.reset_hiden())
        action = torch.argmax(logits, dim=1).cpu().item()
    return action


env = game.PyRat(width=width, height=height, cheeses=cheeses)
exp_replay = rl.ExperienceReplay(max_memory=max_memory, discount=0.85, discount_norm=True)
state = env.observe()
state = np.stack([state] * (past + 1), axis=1)  # past + (present = 1)
# model = rl.NRNNModels(state[0], past_instant = past)
model = rl.NLinearModels(state[0], past_instant=past)
print("[INFO] Input size: and model", state[0].shape, model)

# There are a few methods (= functions) that you should look into.
# For the game.Pyrat() class, you should check out the .observe and the .act methods which
# will be used to get the environment, and send an action.
# For the exp_replay() class, you should check the remember and get_batch methods.
# For the NLinearModels class, you should check the train_on_batch function for training, and
# the forward method for inference (to estimate all pi values given a state).

if load:
    model.load()
model.to(device)


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

    # Define a loss function and optimizer
    criterion = rl.policy_loss
    # optimizer = optim.SGD(model.parameters(), lr = 1e-3)
    optimizer = optim.Adam(model.parameters(), lr=1e-2,)  # weight_decay = 1e-5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 2000, gamma = 0.5)
    prob = [1, 1, 1, 1]  # Intial probabilities
    p_val = p_value((win_cnt - last_W) + (draw_cnt - last_D))
    if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()  # clear if it exists
    for e in tqdm(range(epochs)):
        env.reset()
        game_over = False

        # Get the current state of the environment
        current_st = torch.FloatTensor(env.observe())
        previous_st = torch.zeros_like(current_st)

        # Play a full game until game is over
        model.eval()
        ori_rewards, ori_actions, pvalues = [], [], []
        step = 0
        while not game_over:
            # Predict the Q value for the current-previous states
            state = torch.stack([previous_st, current_st], dim=1).to(
                device)  # (1, time, ...features)
            action = choose_action(model, state)
            pvalues.append(p_val * p_value(step))
            if train and pvalues[-1] > random.uniform(0, 1):
                action = random.choices(range(4), weights=prob, k=1)[0]

            # Apply action, get rewards and new state
            previous_st = current_st
            current_st, reward, game_over = env.act(action)
            current_st = torch.FloatTensor(current_st)
            ori_rewards.append(reward)
            ori_actions.append(action)

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
            exp_replay.remember([state, action, reward], game_over)
            step += 1

        win_hist.append(win_cnt)  # Statistics
        cheeses.append(cheese)  # Statistics
        if train:
            # Train using experience replay. For each batch, get a set of experiences
            # (state, action, new state) that were stored in the buffer.
            # Use this batch to train the model.
            loss = 0.
            model.train()
            for _ in range(number_of_batches):
                inputs, actions, rewards = exp_replay.get_batch(
                    batch_size=batch_size, device=device)
                loss += rl.train_on_batch(model, inputs, (actions, rewards), criterion, optimizer)
            loss /= number_of_batches
            if 'scheduler' in locals() and scheduler is not None:
                scheduler.step()

        if (e + 1) % 100 == 0:  # Statistics every 100 epochs
            cheese_np = np.array(cheeses)
            mem_actions = np.eye(4)[exp_replay.get_actions()].sum(axis=0)
            prob = mem_actions.sum() - mem_actions  # Search new actions!
            p_val = p_value((win_cnt - last_W) + (draw_cnt - last_D))
            print(prob)
            print("[INFO] Actions in memory: ", mem_actions)
            print("[INFO] Actions current game: ", ori_actions)
            print("[INFO] Rewards current game: ", ori_rewards)
            print("[INFO] Total rewards: ", sum(exp_replay.get_rewards()))
            print("[INFO] P_value = ", p_val)
            print("[INFO] P_values = ", pvalues)
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


print("Training")
play(model, epoch, True)
if save:
    model.save()
print("Training done")
print("Testing")
play(model, 1000 if epoch > 1000 else epoch, False)
print("Testing done")
