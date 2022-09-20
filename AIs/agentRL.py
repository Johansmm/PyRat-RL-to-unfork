# Template file to create an AI for the game PyRat
# http://formations.telecom-bretagne.eu/pyrat

# python3 pyrat.py -d 0 -md 0 -p 15 -x 10 -y 10 --rat AIs/agentRL.py\
#                  --python AIs/manh.py --tests 1 --nonsymmetric --turn_time 500

###############################
# Team name to be displayed in the game
import torch.nn as nn
import torch
import numpy as np
TEAM_NAME = "Q-Learner"

###############################
# When the player is performing a move, it actually sends a character to the main program
# The four possibilities are defined here
MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

# Global variables
global model, exp_replay, input_tm1, action, score

# Function to create a numpy array representation of the maze


def input_of_parameters(player, maze, opponent, mazeHeight, mazeWidth, piecesOfCheese):
    im_size = (2 * mazeHeight - 1, 2 * mazeWidth - 1, 4)
    canvas = np.zeros(im_size)
    (x, y) = player
    center_x, center_y = mazeWidth-1, mazeHeight-1
    for (x_cheese, y_cheese) in piecesOfCheese:
        canvas[y_cheese + center_y - y, x_cheese + center_x - x, 0] = 1
    (x_enemy, y_enemy) = opponent
    canvas[y_enemy+center_y-y, x_enemy+center_x-x, 1] = 1
    # canvas[center_y,center_x,2] = 1
    # canvas = np.concatenate([canvas.reshape(-1), [y_enemy - y, x_enemy - x]])

    # Add valid movements
    y_pos_mov = (-1 if y > 0 else 0, 1 if y + 1 < mazeHeight else 0)
    x_pos_mov = (-1 if x > 0 else 0, 1 if x + 1 < mazeWidth else 0)
    canvas[center_y + y_pos_mov[0]:center_y + y_pos_mov[1] + 1,
           center_x + x_pos_mov[0]:center_x + x_pos_mov[1] + 1, 2] = 1
    canvas[center_y, center_x, 2] = 0

    # Enemy position
    for (x_cheese, y_cheese) in piecesOfCheese:
        canvas[y_cheese + center_y - y_enemy, x_cheese + center_x - x_enemy, 3] = 1
    return canvas[None]


class NLinearModels(nn.Module):
    def __init__(self, x_example, number_of_regressors=4, weights_file='saves/qlearning.pt'):
        super(NLinearModels, self).__init__()
        in_features = x_example.reshape(-1).shape[0]
        self.linear2 = nn.Linear(in_features, number_of_regressors)
        self.weights_file = weights_file

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.linear2(x)

    def load(self):
        self.load_state_dict(torch.load(self.weights_file))

    def save(self):
        torch.save(self.state_dict(), self.weights_file)


###############################
# Preprocessing function
# The preprocessing function is called at the start of a game
# It can be used to perform intensive computations that can be
# used later to move the player in the maze.
###############################
# Arguments are:
# mazeMap : dict(pair(int, int), dict(pair(int, int), int))
# mazeWidth : int
# mazeHeight : int
# playerLocation : pair(int, int)
# opponentLocation : pair(int,int)
# piecesOfCheese : list(pair(int, int))
# timeAllowed : float
###############################
# This function is not expected to return anything
def preprocessing(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation,
                  piecesOfCheese, timeAllowed):
    global model, exp_replay, input_tm1, action, score
    input_tm1 = input_of_parameters(playerLocation, mazeMap,
                                    opponentLocation, mazeHeight, mazeWidth, piecesOfCheese)
    action = -1
    score = 0
    model = NLinearModels(input_tm1[0])
    model.load()
    model.to('cpu')

###############################
# Turn function
# The turn function is called each time the game is waiting
# for the player to make a decision (a move).
###############################
# Arguments are:
# mazeMap : dict(pair(int, int), dict(pair(int, int), int))
# mazeWidth : int
# mazeHeight : int
# playerLocation : pair(int, int)
# opponentLocation : pair(int, int)
# playerScore : float
# opponentScore : float
# piecesOfCheese : list(pair(int, int))
# timeAllowed : float
###############################
# This function is expected to return a move


def turn(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore,
         opponentScore, piecesOfCheese, timeAllowed):
    global model, input_tm1, action, score
    input_t = input_of_parameters(playerLocation, mazeMap, opponentLocation,
                                  mazeHeight, mazeWidth, piecesOfCheese)
    input_t = torch.FloatTensor(input_t)
    output = model(input_t)
    action = torch.argmax(output[0]).item()
    score = playerScore
    return [MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN][action]


def postprocessing(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation,
                   playerScore, opponentScore, piecesOfCheese, timeAllowed):
    pass
