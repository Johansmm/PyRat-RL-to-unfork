import os
import sys
import json
import copy
import random
import time
import numpy as np
import argparse
import importlib.util
import inspect

from AIs import manh
from AIs.utils import MOVE_LEFT, MOVE_RIGHT, MOVE_DOWN, MOVE_UP, ALL_MOVES, update_scores
from resources.imports.maze import generate_pieces_of_cheese
from resources.imports import logger


def reload_players(*players):
    """Return the modules of players after reload them

    Parameters
    ----------
    players : List[module]
        player(s) to reload

    Returns
    -------
    List[module]
        player(s) reloaded

    """
    if len(players) == 1:
        return players[0].__loader__.load_module()
    else:
        return tuple(player.__loader__.load_module() for player in players)


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
    random_seed : int, optional
        Random seed, by default None
    opponent_reset : bool, optional
        Reload opponent when call :func:`reset`, by default False
    """

    def __init__(self, width=21, height=15, round_limit=200, cheeses=41, symmetric=False,
                 start_random=True, opponent=manh, random_seed=None, opponent_reset=False,
                 **kwargs):
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
        self.opponent_reset = opponent_reset
        self.reset(random_seed=random_seed)

    @property
    def action_space(self):
        return ALL_MOVES

    @classmethod
    def add_argparse_args(cls, parent_parser, use_argument_group=True):
        """Add argparse argumentos to a parent parser.

        Parameters
        ----------
        parent_parser : ArgumentParser
            The custom cli arguments parser, which will be extended by
            the class's default arguments.
        use_argument_group:
            By default, this is True, and uses ``add_argument_group`` to add
            a new group. If False, this will use old behavior.

        Returns
        -------
            If use_argument_group is True, returns ``parent_parser`` to keep old
            workflows. If False, will return the new ArgumentParser object.

        Notes
        -----
            Inspire from :mod:`pytorch_lightning.utilities.argparse`
        """
        if use_argument_group:
            group_name = f"{cls.__module__}.{cls.__qualname__}"
            parser = parent_parser.add_argument_group(group_name)
        else:
            parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('-x', '--width', type=int, metavar="x", default=31,
                            help='Width of the maze, by default %(default)s')
        parser.add_argument('-y', '--height', type=int, metavar="y", default=29,
                            help='Height of the maze, by default %(default)s')
        parser.add_argument('-p', '--pieces', type=int, metavar="p", default=41, dest="cheese",
                            help='Number of pieces of cheese, by default %(default)s')
        parser.add_argument('--nonsymmetric', action="store_false", dest="symmetric",
                            help='Do not enforce symmetry of the maze')
        parser.add_argument('-mt', '--max_turns', type=int, metavar="mt", default=2000,
                            dest="round_limit", help='Max number of turns, by default %(default)s')
        parser.add_argument('--start_random', action="store_true",
                            help='Players start at random location in the maze')
        parser.add_argument('--random_seed', type=int, metavar="random_seed", default=None,
                            help='Random seed to use in order to generate a specific maze')

        if use_argument_group:
            return parent_parser
        return parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        """Create an instance from CLI arguments.

        Parameters
        ----------
        args : Union[Namespace, ArgumentParser]
            The parser or namespace to take arguments from.
        kwargs : dict
            Additional keyword arguments that may override ones in the parser or namespace.

        Returns
        -------
        :obj:`PyRat`
            Instance of the class started with the arguments

        Notes
        -----
            Code take from :mod:`pytorch_lightning.utilities.argparse`
        """
        params = vars(args)

        # We only want to pass in valid Trainer args, the rest may be user specific
        valid_kwargs = inspect.signature(cls.__init__).parameters
        trainer_kwargs = {name: params[name] for name in valid_kwargs if name in params}
        trainer_kwargs.update(**kwargs)
        return cls(**trainer_kwargs)

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
            self.miss_enemy += 1
            logger.debug("FUUUU: Opponent uncertain movement. Stay in same position.")
        new_x, new_y = (xx + enemy_action_x, yy + enemy_action_y)
        if new_x < 0 or new_x > self.width - 1 or new_y < 0 or new_y > self.height - 1:
            new_x, new_y = xx, yy
            self.miss_enemy += 1
        self.enemy = (new_x, new_y)

        self.round += 1
        action_x = 0
        action_y = 0
        self.illegal_move = False
        if action == MOVE_LEFT:
            action_x = -1
        elif action == MOVE_RIGHT:
            action_x = 1
        elif action == MOVE_UP:
            action_y = 1
        elif action == MOVE_DOWN:
            action_y = -1
        elif action is None:
            self.illegal_move = True
            logger.debug("FUUUU: Player uncertain movement. Stay in same position.")
        else:
            raise ValueError("INVALID MOVEMENT PLAYER")
        (x, y) = self.player
        new_x = x + action_x
        new_y = y + action_y
        if new_x < 0 or new_x > self.width - 1 or new_y < 0 or new_y > self.height - 1:
            new_x = x
            new_y = y
            self.illegal_move = True
        if self.illegal_move:
            self.miss_player += 1
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

    def reset(self, random_seed=None):
        """Reset scores, positions and create a new list of cheeses

        Parameters
        ----------
        random_seed : int, optional
            Random seed, by default None
        """
        self.random_seed = random.randint(0, sys.maxsize) if random_seed is None else random_seed
        random.seed(self.random_seed)
        self.piecesOfCheese, self.player, self.enemy = generate_pieces_of_cheese(
            self.cheeses, self.width, self.height, self.symmetric,
            (0, 0), (self.width - 1, self.height - 1), self.start_random)
        self.round = 0
        self.illegal_move = False
        self.score = 0
        self.enemy_score = 0
        self._draw_state()
        if not self.preprocess or self.opponent_reset:
            self.opponent = reload_players(self.opponent)
            self.opponent.preprocessing(None, self.width, self.height,
                                        self.enemy, self.player, self.piecesOfCheese, 30000)
            self.preprocess = True
        self.miss_enemy = 0
        self.miss_player = 0


class Statistics():
    """Save all record's history of several games

    Attributes
    ----------
    moves : list
        Record of movements per game
    score_python : list
        Record of python's score per game
    score_rat : list
        Record of rat's score per game
    win_python : float
        Python's total score
    win_rat : float
        Rat's total score
    num_games : int
        Number of games append in stats

    Parameters
    ----------
    ini_score_rat : float, optional
        Initial rat's score, by default 0
    ini_score_python : float, optional
        Initial python's score, by default 0
    ini_moves : int, optional
        Initial moves per player, by default 0
    """

    def __init__(self, ini_score_rat=0.0, ini_score_python=0.0, ini_moves=0):
        self.reset()
        self._params = [k for k, v in vars(self).items() if isinstance(v, (list, float, int))]
        # Delete time variables
        self._params = [k for k in self._params if "time" not in k]
        if ini_score_rat != 0 or ini_score_python != 0 or ini_moves != 0:
            self.__add__((ini_score_rat, ini_score_python, ini_moves))

    def get_stats(self):
        """Get summary of stats

        Returns
        -------
        dict
            Statistics

        Raises
        ------
        ValueError
            :attr:`num_game` must be higher than 0
        """
        if self.num_games == 0:
            raise ValueError("Please introduce at least one game!.")
        turn_time = (self.current_time - self.start_time) / sum(self.moves)
        turn_time = f"{turn_time:.3e} s" if turn_time < 1 else f"{turn_time:.3f} s"
        return {
            "miss_rat": sum(self.miss_rat),
            "miss_python": sum(self.miss_python),
            "moves_per_player": sum(self.moves) / self.num_games,
            "score_rat": sum(self.score_rat) / self.num_games,
            "score_python": sum(self.score_python) / self.num_games,
            "win_rat": self.win_rat / self.num_games,
            "win_python": self.win_python / self.num_games,
            "num_games": self.num_games,
            "turn_time": turn_time,
        }

    def __add__(self, stats):
        """Add new stats

        Parameters
        ----------
        stats : Union[Tuple, Statistics]
            New statistics

        Returns
        -------
        Statistics
            Statistics updated
        """
        if isinstance(stats, Statistics):
            new_obj = copy.deepcopy(self)
            for param in self._params:
                setattr(new_obj, param, getattr(self, param) + getattr(stats, param))
            # We add the total time of previous statistics
            new_obj.start_time -= stats.current_time - stats.start_time
            return new_obj

        if len(stats) == 3:
            score_rat, score_python, moves = stats
        elif len(stats) == 5:
            score_rat, score_python, moves, miss_rat, miss_python = stats
        else:
            raise ValueError(f"Impossible to unzip statistics: {stats}")
        self.score_rat.append(score_rat)
        self.score_python.append(score_python)
        if score_rat > score_python:
            self.win_rat += 1
        elif score_rat < score_python:
            self.win_python += 1
        else:
            self.win_rat += 0.5
            self.win_python += 0.5
        self.moves.append(moves)
        self.miss_rat.append(miss_rat)
        self.miss_python.append(miss_python)
        self.num_games += 1
        self.current_time = time.time()
        return self

    def reset(self):
        """Reset statistics
        """
        self.score_python = []
        self.score_rat = []
        self.moves = []
        self.win_python = 0
        self.win_rat = 0
        self.num_games = 0
        self.miss_python = []
        self.miss_rat = []
        self.start_time = self.current_time = time.time()


def parse_args():
    parser = argparse.ArgumentParser("Simulator of PyRat")
    parser.add_argument('--rat', type=str, metavar="rat_file", default="",
                        help='Program to control the rat (local file)')
    parser.add_argument('--python', type=str, metavar="python_file", default="",
                        help='Program to control the python (local file)')
    parser.add_argument('--num_games', type=int, metavar="num_games", default=1,
                        help='Number of games to launch (for statistics), by default %(default)s')
    parser = PyRat.add_argparse_args(parser, use_argument_group=False)
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
    # Read players from python script
    player = _read_player(args.rat)
    args.opponent = _read_player(args.python)
    player_name = player.__name__
    opponent_name = args.opponent.__name__

    # Create the enviroment of game
    game = PyRat.from_argparse_args(args, opponent_reset=True)
    player.preprocessing(None, game.width, game.height, game.enemy, game.player,
                         game.piecesOfCheese, 30000)
    stats = Statistics()

    # Run all games
    for match in range(args.num_games):
        print(f"Using seed {game.random_seed}")
        if args.num_games > 1:
            print(f"Match {match+1}/{args.num_games}")
        while not game._is_over():
            # From the state of the game, the player can make a decision
            action = player.turn(*game.observe(full=True))
            # This action directly affects the environment
            game.act(action)

        # Print records of game
        score = f"{game.score}/{game.enemy_score}"
        if game.score < game.enemy_score:
            logger.info(f"The Python ({opponent_name}) won the match! ({score})")
        elif game.score > game.enemy_score:
            logger.info(f"The Rat ({player_name}) won the match! ({score})")
        else:
            logger.info(f"The Rat ({player_name}) and the Python ({opponent_name}) "
                        f"got the same number of pieces of cheese! ({score})")

        # Update statistics
        stats += (game.score, game.enemy_score, game.round, game.miss_player, game.miss_enemy)

        # To next game, reset all status (including players)
        game.reset()
        player = reload_players(player)

    # Print summary
    print(json.dumps(stats.get_stats(), indent=4))


if __name__ == "__main__":
    play_game(parse_args())
