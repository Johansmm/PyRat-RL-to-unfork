#####################################
# This file is only useful for the Introduction to AI course
# This is useless for PyRat
#####################################

from .utils import MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP, distance


def preprocessing(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation,
                  piecesOfCheese, timeAllowed):
    pass


def turn(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation,
         playerScore, opponentScore, piecesOfCheese, timeAllowed):
    closest_poc = (-1, -1)
    best_distance = mazeWidth + mazeHeight
    for poc in piecesOfCheese:
        if distance(poc, playerLocation) < best_distance:
            best_distance = distance(poc, playerLocation)
            closest_poc = poc
    ax, ay = playerLocation
    bx, by = closest_poc
    if bx > ax:
        return MOVE_RIGHT
    if bx < ax:
        return MOVE_LEFT
    if by > ay:
        return MOVE_UP
    if by < ay:
        return MOVE_DOWN
    pass
