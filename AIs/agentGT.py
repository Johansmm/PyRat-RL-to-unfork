# With this template,
# we are building an AI that will apply
# combinatorial game theory tools against a greedy opponent.

# Unless you know what you are doing,
# you should use this template with a very limited number of pieces of cheese,
# as it is very demanding in terms of computations.

# A typical use would be:
# python pyrat.py -d 0 -md 0 -p 7 --rat AIs/agentGT.py --python AIs/manh.py --nonsymmetric

# If enough computation time is allowed,
# it is reasonable to grow the number of pieces of cheese up to around 17.
# For example:

# python pyrat.py -d 0 -md 0 -p 15 --rat AIs/agentGT.py --python AIs/manh.py --synchronous\
#                 --tests 100 --nodrawing --nonsymmetric

# In this example, we can obtain scores in the order of: "win_python": 0.08 "win_rat": 0.91

import numpy as np
from .utils import (MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP,
                    make_move, move_to_target, update_scores)
from . import manh
from resources.imports import logger

# During our turn we continue going to the next target, unless the piece of cheese it originally
# contained has been taken.
# In such case, we pass at the next better target. At the beginning, there is not a initial target
current_targets = []
final_score = 0.0


def turn_of_opponent(opponentLocation, piecesOfCheese, oponent=manh):
    """Reproduce the turn of the opponent, only with its position and location of cheeses

    Parameters
    ----------
    opponentLocation : Tuple[int, int]
        Current opponent position
    piecesOfCheese : List[Tuple[int, int]]
        Position of cheeses
    oponent : object
        object that contains the function ``turn``

    Returns
    -------
    Tuple[int, int]
        New opponent's location
    """
    opponent_move = oponent.turn(None, np.inf, np.inf, opponentLocation,
                                 None, None, None, piecesOfCheese, 0)
    # If the opponent did not move, it keeps him in the same position
    return opponent_move


# We dont need the preprocessing function
def preprocessing(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation,
                  piecesOfCheese, timeAllowed):
    pass


def best_targets(playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese):
    """Recursive function that goes through the trees of possible plays.

    It takes as arguments a given situation, and return a best targets piece of cheese for
    the player, such that aiming to grab these piece of cheese will eventually lead to a
    maximum score.

    Parameters
    ----------
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

    Returns
    -------
    Tuple[Tuple[int, int], float]
        Best target and final score
    """

    # First we should check how many pieces of cheese each player has to see if the match is over.
    # It is the case if no pieces of cheese are left,
    # or if playerScore or opponentScore is more than half the total number
    # playerScore + opponentScore + piecesOfCheese
    totalPieces = len(piecesOfCheese) + playerScore + opponentScore
    if playerScore > totalPieces / 2 or opponentScore > totalPieces / 2 or len(piecesOfCheese) == 0:
        return [(-1, -1)], playerScore

    # If the match is not over, then the player can aim for any of the remaining pieces of cheese
    # So we will simulate the game to each of the pieces, which will then by recurrence test all
    # the possible trees.
    # It will remember each step as soon as possible, return the complete way to win.
    best_score_so_far = -1
    best_target_so_far = []
    for target in piecesOfCheese:
        # Play's until go to target
        end_state = simulate_game_until_target(
            target, playerLocation, opponentLocation,
            playerScore, opponentScore, piecesOfCheese.copy())
        # Recover the next better movements. Save the moves if are better
        list_targets, score = best_targets(*end_state)
        if score > best_score_so_far:
            best_score_so_far = score
            best_target_so_far = [target] + list_targets
        # If player knows that will win, it makes no sense to continue
        if score > totalPieces / 2 and score > end_state[-2]:
            break

    return best_target_so_far, best_score_so_far


def simulate_game_until_target(target, playerLocation, opponentLocation, playerScore,
                               opponentScore, piecesOfCheese):
    """Simulate what will happen until we reach the target

    Parameters
    ----------
    target : Tuple[int, int]
        Coordinates of the cheese to be reached for the player
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

    Returns
    -------
    Tuple[Tuple[int, int], Tuple[int, int], float, float, List[Tuple[int, int]]]
        End state of the game until reache the targer
    """

    # While the target cheese has not yet been eaten by either player
    # We simulate how the game will evolve until that happens
    while target in piecesOfCheese:
        # Update playerLocation (position of your player) using updatePlayerLocation
        playerLocation = move_to_target(playerLocation, target)
        # Every time that we move the opponent also moves. update the position of the opponent
        # using turn_of_opponent and move
        opponentLocation = make_move(opponentLocation, turn_of_opponent(
            opponentLocation, piecesOfCheese))
        # Finally use the function update_scores to see if any of the players is in the same
        # square of a cheese.
        playerScore, opponentScore = update_scores(
            playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese)
    return playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese


def turn(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore,
         opponentScore, piecesOfCheese, timeAllowed):
    """Given a state of the game, this function takes an action for the player based on the
    combinatorial game theory tools against a greedy opponent.

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
    global current_targets, final_score
    # If player does not know the way to win, predicts it
    if len(current_targets) == 0:
        current_targets, final_score = best_targets(
            playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese)
    # Therefore, it checks if next target exists in the piecesOfCheese
    old_target = None
    while not (len(current_targets) == 0 or current_targets[0] in piecesOfCheese):
        old_target = current_targets.pop(0)
    cur_target = (-1, -1) if len(current_targets) == 0 else current_targets[0]

    # Print new target only if it changed
    if old_target is not None:
        logger.info(f"My new targets is {cur_target} and I will finish with "
                    f"{final_score} pieces of cheese.")

    # Return the movement towards the target
    if cur_target[1] > playerLocation[1]:
        return MOVE_UP
    if cur_target[1] < playerLocation[1]:
        return MOVE_DOWN
    if cur_target[0] > playerLocation[0]:
        return MOVE_RIGHT
    return MOVE_LEFT
