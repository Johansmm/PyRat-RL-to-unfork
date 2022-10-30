MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'
ALL_MOVES = [MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP]


def make_move(location, move):
    """Update an input location (x,y) with the desired movement

    Parameters
    ----------
    location : Tuple[int, int]
        Current location
    move : str
        One of [MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP]

    Returns
    -------
    Tuple[int, int]
        The updated location
    """

    if move == MOVE_UP:
        return (location[0], location[1] + 1)
    if move == MOVE_DOWN:
        return (location[0], location[1] - 1)
    if move == MOVE_LEFT:
        return (location[0] - 1, location[1])
    if move == MOVE_RIGHT:
        return (location[0] + 1, location[1])
    raise ValueError(F"Unknown move = {move}. Availables: {ALL_MOVES}")


def distance(la, lb):
    """Calculate the distance between two locations

    Parameters
    ----------
    la : Tuple[int, int]
        First location
    lb : Tuple[int, int]
        Second location

    Returns
    -------
    int
        Distance between locations
    """
    ax, ay = la
    bx, by = lb
    return abs(bx - ax) + abs(by - ay)


def move_to_target(location, target):
    """Move the agent on the labyrinth.

    Given that one agent only moves once and can't move diagonally, it suffices to move in
    the direction of the target, moving vertically first and then horizontally

    Parameters
    ----------
    location : Tuple[int, int]
        Current location
    target : Tuple[int, int]
        Coordinates of cheese target

    Returns
    -------
    Tuple[int, int]
        The updated location
    """
    if target[1] > location[1]:
        return make_move(location, MOVE_UP)
    if target[1] < location[1]:
        return make_move(location, MOVE_DOWN)
    if target[0] > location[0]:
        return make_move(location, MOVE_RIGHT)
    if target[0] < location[0]:
        return make_move(location, MOVE_LEFT)
    # We are in the target !
    return location


def update_scores(playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese):
    """Function that update list of remaining cheeses and return the new score

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
        Scores of player and opponent updated
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
