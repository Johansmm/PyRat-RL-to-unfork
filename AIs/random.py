##############################################################
# The turn function should always return a move to indicate where to go
# The four possibilities are defined here
##############################################################

# Import of random module
import random
from .utils import ALL_MOVES

##############################################################
# Please put your code here (imports, variables, functions...)
##############################################################


def random_move():
    return random.choice(ALL_MOVES)


##############################################################
# The preprocessing function is called at the start of a game
# It can be used to perform intensive computations that can be
# used later to move the player in the maze.
# ------------------------------------------------------------
# maze_map : dict(pair(int, int), dict(pair(int, int), int))
# maze_width : int
# maze_height : int
# player_location : pair(int, int)
# opponent_location : pair(int,int)
# pieces_of_cheese : list(pair(int, int))
# time_allowed : float
##############################################################

def preprocessing(maze_map, maze_width, maze_height, player_location, opponent_location,
                  pieces_of_cheese, time_allowed):
    # Nothing to do here
    pass


##############################################################
# The turn function is called each time the game is waiting
# for the player to make a decision (a move).
# ------------------------------------------------------------
# maze_map : dict(pair(int, int), dict(pair(int, int), int))
# maze_width : int
# maze_height : int
# player_location : pair(int, int)
# opponent_location : pair(int,int)
# player_score : float
# opponent_score : float
# pieces_of_cheese : list(pair(int, int))
# time_allowed : float
##############################################################

def turn(maze_map, maze_width, maze_height, player_location, opponent_location,
         player_score, opponent_score, pieces_of_cheese, time_allowed):
    # Returns a random move each turn
    return random_move()
