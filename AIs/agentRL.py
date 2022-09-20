# With this template, we are building an AI based on
# Reinforcement Learning theory

# A neuronal network will predict the move that will lead to the
# highest possible score at the end of the game, given a current game state.
# Therefore, the network will try to maximize the cumulative reward of the game given
# the probability to make a decision (policy) from a gaming state.

# A typical use would be:
# python pyrat.py -d 0 -md 0 -p 15 --rat AIs/agentRL.py --python AIs/manh.py\
#                 --tests 100 --nonsymmetric --turn_time 500

###############################

import torch.nn as nn
import torch
import numpy as np

from .utils import ALL_MOVES
TEAM_NAME = "Q-Learner"

# Global variables
model = None
device = "cpu"

###############################


class PerceptronModel(nn.Module):
    """Predict the action that the player shall take, given an input state of game

    Parameters
    ----------
    x_example : Tuple[torch.Tensor, np.ndarray]
        Input example, to calculate the input shape
    number_of_regressors : int, optional
        Output shape, by default 4
    dropout : float, optional
        Dropout probability for inputs, by default 0.4.
    weights_path : str, optional
        Path where weights will be loaded, by default 'saves/best_qlearning.pt'
    """

    def __init__(self, x_example, number_of_regressors=4, dropout=0.01,
                 weights_path='saves/best_qlearning.pt'):
        super(PerceptronModel, self).__init__()
        in_features = x_example.reshape(-1).shape[0]
        self.linear = nn.Linear(in_features, number_of_regressors)
        self.dropout = nn.Dropout(p=dropout)
        self.weights_path = weights_path

    def forward(self, x):
        """Forward an input maze through the network

        Parameters
        ----------
        x : torch.Tensor
            Maze of size (B, H, W, C)

        Returns
        -------
        torch.Tensor
            Output prediction of size (B, number_of_regressors)
        """
        # Input flatten
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        return self.linear(x)

    def load(self, weights_path=None):
        """Load weights from ``weights_path``

        Parameters
        ----------
        weights_path : str, optional
            Path of weights, by default None
        """
        wpath = weights_path or self.weights_path
        self.load_state_dict(torch.load(wpath))

    def save(self, weights_path=None):
        """Save the model to ``weights_path``

        Parameters
        ----------
        weights_path : str, optional
            Path of weights, by default None
        """
        wpath = weights_path or self.weights_path
        torch.save(self.state_dict(), wpath)


def maze_preprocessing(player, opponent, maze, mazeHeight, mazeWidth, piecesOfCheese):
    """Build input for RL-player

    Parameters
    ----------
    player : Tuple[int, int]
        Current player position
    opponent : Tuple[int, int]
        Current opponent position
    maze : List[List[int]]
        Current game description
    mazeHeight : int
        Height of the maze
    mazeWidth : int
        Width of maze
    piecesOfCheese : List[Tuple[int, int]]
        List of remaining cheeses

    Returns
    -------
    List[List[int]]
        Maze preprocessed
    """
    # Create empty canvas
    im_size = (2 * mazeHeight - 1, 2 * mazeWidth - 1, 4)
    canvas = np.zeros(im_size)
    (x, y) = player
    center_x, center_y = mazeWidth - 1, mazeHeight - 1

    # View of cheeses of player in first layer
    for (x_cheese, y_cheese) in piecesOfCheese:
        canvas[y_cheese + center_y - y, x_cheese + center_x - x, 0] = 1

    # Enemy position in second layer
    (x_enemy, y_enemy) = opponent
    canvas[y_enemy + center_y - y, x_enemy + center_x - x, 1] = 1
    # canvas[center_y, center_x, 2] = 1
    # canvas = np.concatenate([canvas.reshape(-1), [y_enemy - y, x_enemy - x]])

    # Add valid movements in third layer
    y_pos_mov = (-1 if y > 0 else 0, 1 if y + 1 < mazeHeight else 0)
    x_pos_mov = (-1 if x > 0 else 0, 1 if x + 1 < mazeWidth else 0)
    canvas[center_y + y_pos_mov[0]:center_y + y_pos_mov[1] + 1,
           center_x + x_pos_mov[0]:center_x + x_pos_mov[1] + 1, 2] = 1
    canvas[center_y, center_x, 2] = 0

    # View of cheeses of enemy in last layer
    for (x_cheese, y_cheese) in piecesOfCheese:
        canvas[y_cheese + center_y - y_enemy, x_cheese + center_x - x_enemy, 3] = 1

    # Return canvas as batch
    return canvas[None]


def preprocessing(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation,
                  piecesOfCheese, timeAllowed):
    """The preprocessing function is called at the start of a game.

    It can be used to perform intensive computations that can be, used later
    to move the player in the maze.

    This function is not expected to return anything

    Parameters
    ----------
    mazeMap : List[List[int]]
        Current game description
    mazeWidth : int
        Width of maze
    mazeHeight : int
        Height of the maze
    playerLocation : Tuple[int, int]
        Current player position
    opponentLocation : Tuple[int, int]
        Current opponent position
    piecesOfCheese : List[Tuple[int, int]]
        List of remaining cheeses
    timeAllowed : int
        Timeout before making a decision, in milliseconds
    """

    global model, device, score

    # Read model only one time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        input_tm1 = maze_preprocessing(playerLocation, opponentLocation, mazeMap,
                                       mazeHeight, mazeWidth, piecesOfCheese)
        model = PerceptronModel(input_tm1[0])
        model.load()
        model.eval()
        model.to(device)

    score = 0


def turn(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore,
         opponentScore, piecesOfCheese, timeAllowed):
    """Build the input maze to pass through the network, in order to predict the action to take.

    Parameters
    ----------
    mazeMap : List[List[int]]
        Current game description
    mazeWidth : int
        Width of maze
    mazeHeight : int
        Height of the maze
    playerLocation : Tuple[int, int]
        Current player position
    opponentLocation : Tuple[int, int]
        Current opponent position
    playerScore : float
        Current player score
    opponentScore : float
        Current opponent score
    piecesOfCheese : List[Tuple[int, int]]
        List of remaining cheeses
    timeAllowed : int
        Timeout before making a decision, in milliseconds

    Returns
    -------
    str
        The movement to be performed between four choises: ["D", "R", "L", "U"]
    """
    global model, device, action, score
    input_t = maze_preprocessing(playerLocation, opponentLocation, mazeMap,
                                 mazeHeight, mazeWidth, piecesOfCheese)
    input_t = torch.FloatTensor(input_t).to(device)
    output = model(input_t)
    action = torch.argmax(output[0]).cpu().item()
    score = playerScore
    return ALL_MOVES[action]


def postprocessing(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation,
                   playerScore, opponentScore, piecesOfCheese, timeAllowed):
    pass
