import os
import numpy as np
import argparse
import importlib.util

from AIs import manh
from AIs.utils import MOVE_LEFT, MOVE_RIGHT, MOVE_DOWN, MOVE_UP, ALL_MOVES, update_scores
from resources.imports.maze import generate_pieces_of_cheese


class PyRat(object):
    """Pyrat simulator, which performs a synchronous game between two players,
    eliminating the animations to obtain faster games.

    Similar behavior can be obtained on pyrat.py, from the command line:

    .. code-block:: console
        python pyrat.py -d 0 -md 0 --test 1 --synchronous --nodrawing

    width : int, optional
        Width of the maze (-x, --width), by default 21
    height : int, optional
        Height of the maze (-y, --height),, by default 15
    round_limit : int, optional
        Max number of turns (-mt, --max_turns), by default 200
    cheeses : int, optional
        Number of pieces of cheese (-p, --pieces), by default 41
    symmetric : bool, optional
        Enforce symmetry of the maze (not --nonsymetric), by default False
    start_random : bool, optional
        Players start at random location in the maze (--start_random), by default True
    opponent : object, optional
        Oponent AIs (--python), by default manh
    """

    def __init__(self, width=21, height=15, round_limit=200, cheeses=41, symmetric=False,
                 start_random=True, opponent=manh):
        self.preprocess = False
        self.symmetric = symmetric
        self.start_random = start_random
        self.height = height
        self.width = width
        self.cheeses = cheeses
        self.piecesOfCheese = list()
        self.round_limit = round_limit
        self.round = 0
        self.score = 0
        self.opponent = opponent
        self.reset()

    def _update_state(self, action, enemy_action):
        """Players' action, which update the state of the game

        Parameters
        ----------
        action : str
            Player action
        enemy_action : str
            Enemy action
        """
        (xx, yy) = self.enemy
        enemy_action_x = 0
        enemy_action_y = 0

        if enemy_action == MOVE_DOWN and yy > 0:
            enemy_action_y = -1
        elif enemy_action == MOVE_UP and yy < self.height - 1:
            enemy_action_y = +1
        elif enemy_action == MOVE_LEFT and xx > 0:
            enemy_action_x = -1
        elif enemy_action == MOVE_RIGHT and xx < self.width - 1:
            enemy_action_x = +1
        elif enemy_action not in ALL_MOVES:
            print("FUUUU: Opponent uncertain movement. Stay in same position.")
        self.enemy = (xx + enemy_action_x, yy + enemy_action_y)

        self.round += 1
        action_x = 0
        action_y = 0
        if action == MOVE_LEFT:
            action_x = -1
        elif action == MOVE_RIGHT:
            action_x = 1
        elif action == MOVE_UP:
            action_y = 1
        elif action == MOVE_DOWN:
            action_y = -1
        elif action is None:
            print("FUUUU: Player uncertain movement. Stay in same position.")
        else:
            raise ValueError("INVALID MOVEMENT PLAYER")
        (x, y) = self.player
        new_x = x + action_x
        new_y = y + action_y
        self.illegal_move = False
        if new_x < 0 or new_x > self.width - 1 or new_y < 0 or new_y > self.height - 1:
            new_x = x
            new_y = y
            self.illegal_move = True
        self.player = (new_x, new_y)
        self._draw_state()

    def _draw_state(self):
        """Draw list of cheese and players' position on a maze of size (height, width, 3), where:
            - cheese are in first dimension,
            - player is in the second one and
            - opponent is in the last one.

        Returns
        -------
        np.ndarray
            Current state of the maze with players' and cheeses position.
        """
        im_size = (self.height, self.width, 3)
        self.canvas = np.zeros(im_size)

        # Print cheeses in first layer of canvas
        for (x_cheese, y_cheese) in self.piecesOfCheese:
            self.canvas[y_cheese, x_cheese, 0] = 1

        # Print player position in second layer
        self.canvas[self.player[1], self.player[0], 1] = 1

        # Print opponent position in second layer
        self.canvas[self.enemy[1], self.enemy[0], 2] = 1
        return self.canvas

    def _get_reward(self):
        """Update scores and return reward of player

        Returns
        -------
        float
            Reward get for last action, between {0, 0.5, 1.0}
        """
        if self.round > self.round_limit:
            return -1  # Lost for max_round

        old_score = self.score
        self.score, self.enemy_score = update_scores(
            self.player, self.enemy, self.score, self.enemy_score, self.piecesOfCheese)
        return self.score - old_score

    def _is_over(self):
        """Check if game is over

        Returns
        -------
        bool
            Game's status
        """
        if self.score > self.cheeses / 2 or self.enemy_score > self.cheeses / 2 or (
            (self.score == self.enemy_score) and self.score >= self.cheeses / 2
        ) or self.round > self.round_limit or len(self.piecesOfCheese) == 0:
            return True
        else:
            return False

    def observe(self, full=False):
        """Return the current maze map

        Parameters
        ----------
        full : bool, optional
            Return full description of game, by default False

        Returns
        -------
        List[objects] (full=True)
            Full state of the game (maze_map, width, height, player_pos, enemy_pos,
                                    player_score, enemy_score, piecesOfCheese, allowed_time)
        np.ndarray (full=False)
            Maze map of size (1, height, width, 3)
        """
        maze_map = np.expand_dims(self.canvas, axis=0)
        if full:
            return (maze_map, self.width, self.height, self.player, self.enemy, self.score,
                    self.enemy_score, self.piecesOfCheese, 30000)
        else:
            return maze_map

    def act(self, action):
        """Advance one step in the game, where the following steps are executed:
        1. Get action of enemy.
        2. Update scores and :attr:`piecesOfCheese` from player and enemy actions.
        3. Observes the state of the game

        Parameters
        ----------
        action : str
            Player action, one of ['D', 'U', 'L', 'R']

        Returns
        -------
        np.ndarray
            Maze map of size (1, height, width, 3)
        float
            Player's reward for this action
        bool
            Game's status: is finished or not
        """
        enemy_action = self.opponent.turn(None, self.width, self.height, self.enemy,
                                          self.player, self.enemy_score, self.score,
                                          self.piecesOfCheese, 3000)
        self._update_state(action, enemy_action)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over

    def reset(self):
        """Reset scores, positions and create a new list of cheeses
        """
        self.piecesOfCheese, self.player, self.enemy = generate_pieces_of_cheese(
            self.cheeses, self.width, self.height, self.symmetric,
            (0, 0), (self.width - 1, self.height - 1), self.start_random)
        self.round = 0
        self.illegal_move = False
        self.score = 0
        self.enemy_score = 0
        self._draw_state()
        if not self.preprocess:
            self.opponent.preprocessing(None, self.width, self.height,
                                        self.enemy, self.player, self.piecesOfCheese, 30000)
            self.preprocess = True


def parse_args():
    parser = argparse.ArgumentParser("Simulator of PyRat")
    parser.add_argument('--rat', type=str, metavar="rat_file", default="",
                        help='Program to control the rat (local file)')
    parser.add_argument('--python', type=str, metavar="python_file", default="",
                        help='Program to control the python (local file)')
    parser.add_argument('-x', '--width', type=int, metavar="x", default=31,
                        help='Width of the maze, by default %(default)s')
    parser.add_argument('-y', '--height', type=int, metavar="y", default=29,
                        help='Height of the maze, by default %(default)s')
    parser.add_argument('-p', '--pieces', type=int, metavar="p", default=41,
                        help='Number of pieces of cheese, by default %(default)s')
    parser.add_argument('--nonsymmetric', action="store_true",
                        help='Do not enforce symmetry of the maze')
    parser.add_argument('-mt', '--max_turns', type=int, metavar="mt", default=2000,
                        help='Max number of turns, by default %(default)s')
    parser.add_argument('--start_random', action="store_true",
                        help='Players start at random location in the maze')
    return parser.parse_args()


def _read_player(filename):
    try:
        player = importlib.util.spec_from_file_location("AIs.player", filename)
        module = importlib.util.module_from_spec(player)
        player.loader.exec_module(module)
    except Exception:
        if filename != "":
            raise ValueError(f"Impossible to load player of {filename}")
        player = importlib.util.spec_from_file_location(
            "player", os.path.join("resources", "imports", "dummy_player.py"))
        module = importlib.util.module_from_spec(player)
        player.loader.exec_module(module)
    return module


def play_game(args):
    args_dict = vars(args)
    # Read players from python script
    player = _read_player(args_dict.pop("rat"))
    opponent = args_dict.pop("python")
    if opponent != "":
        args_dict["opponent"] = _read_player(opponent)

    # Create the enviroment of game
    args_dict["round_limit"] = args_dict.pop("max_turns")
    args_dict["symmetric"] = not args_dict.pop("nonsymmetric")
    args_dict["cheeses"] = args_dict.pop("pieces")
    game = PyRat(**args_dict)
    player.preprocessing(None, game.width, game.height, game.enemy, game.player,
                         game.piecesOfCheese, 30000)

    # Run the game
    while not game._is_over():
        # From the state of the game, the player can make a decision
        action = player.turn(*game.observe(full=True))
        # This action directly affects the environment
        game.act(action)

    # Print records of game
    print(f"[INFO] final player's score: {game.score}")
    print(f"[INFO] final opponent's score: {game.enemy_score}")


if __name__ == "__main__":
    play_game(parse_args())
