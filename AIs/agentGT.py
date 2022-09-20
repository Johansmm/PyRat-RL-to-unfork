# With this template,
# we are building an AI that will apply
# combinatorial game theory tools against a greedy opponent.

# Unless you know what you are doing,
# you should use this template with a very limited number of pieces of cheese,
# as it is very demanding in terms of computations.

# A typical use would be:
# python pyrat.py -d 0 -md 0 -p 7 --rat AIs/agentGT.py --python AIs/manh.py --nonsymmetric

# If enough computation time is allowed,
# it is reasonable to grow the number of pieces of cheese up to around 15.
# For example:

# python pyrat.py -d 0 -md 0 -p 13 --rat AIs/agentGT.py --python AIs/manh.py --synchronous\
#                 --tests 100 --nodrawing --nonsymmetric

# In this example, we can obtain scores in the order of: "win_python": 0.07 "win_rat": 0.93

import numpy as np
from .utils import MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP, move, move_to_target
from . import manh

# During our turn we continue going to the next target, unless the piece of cheese it originally
# contained has been taken.
# In such case, we compute the new best target to go to
current_target = (-1, -1)


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


def best_target(playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese):
    """Recursive function that goes through the trees of possible plays.

    It takes as arguments a given situation, and return a best target piece of cheese for
    the player, such that aiming to grab this piece of cheese will eventually lead to a
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
        return (-1, -1), playerScore

    # If the match is not over, then the player can aim for any of the remaining pieces of cheese
    # So we will simulate the game to each of the pieces, which will then by recurrence test all
    # the possible trees.
    best_score_so_far = -1
    best_target_so_far = (-1, -1)
    for target in piecesOfCheese:
        end_state = simulate_game_until_target(
            target, playerLocation, opponentLocation,
            playerScore, opponentScore, piecesOfCheese.copy())
        _, score = best_target(*end_state)
        if score > best_score_so_far:
            best_score_so_far = score
            best_target_so_far = target

    return best_target_so_far, best_score_so_far


def checkEatCheese(playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese):
    """Function that update list of remaining cheeses

    Each player win +1 point iff they are alone on the square with a cheese. If both players are
    in the same square and there is a cheese on the square each player gets 0.5 points.
    The cheeses that were taken are removed from the list **inplace**.

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
    Tuple[float, float]
        Scores updated
    """
    # Check if player is on cheese
    player_on_cheese = playerLocation in piecesOfCheese
    if player_on_cheese:
        playerScore += 1.0
        piecesOfCheese.remove(playerLocation)

    # Now check if opponent is on another cheese
    opponent_on_cheese = opponentLocation in piecesOfCheese
    if opponent_on_cheese:
        opponentScore += 1.0
        piecesOfCheese.remove(opponentLocation)

    # Penalizes players if they obtained the same cheese
    if player_on_cheese and playerLocation == opponentLocation:
        playerScore -= 0.5
        opponentScore += 0.5
    return playerScore, opponentScore


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
        opponentLocation = move(opponentLocation, turn_of_opponent(
            opponentLocation, piecesOfCheese))
        # Finally use the function checkEatCheese to see if any of the players is in the same
        # square of a cheese.
        playerScore, opponentScore = checkEatCheese(
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
    global current_target
    if current_target not in piecesOfCheese:
        current_target, score = best_target(
            playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese)
        print(f"My new target is {current_target} and I will finish with {score} pieces of cheese")

    # Return the movement towards the target
    if current_target[1] > playerLocation[1]:
        return MOVE_UP
    if current_target[1] < playerLocation[1]:
        return MOVE_DOWN
    if current_target[0] > playerLocation[0]:
        return MOVE_RIGHT
    return MOVE_LEFT
