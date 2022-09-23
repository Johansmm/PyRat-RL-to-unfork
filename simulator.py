import numpy as np

from AIs import manh
from AIs.utils import MOVE_LEFT, MOVE_RIGHT, MOVE_DOWN, MOVE_UP, ALL_MOVES
from resources.imports.maze import generate_pieces_of_cheese


class PyRat(object):
    def __init__(self, width=21, height=15, round_limit=200, cheeses=40, symmetric=False,
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
        """
        Input: actions and states
        Ouput: new states and reward
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
        if self.round > self.round_limit:
            return -1  # Lost for max_round

        # Check if player is on cheese
        player_on_cheese = self.player in self.piecesOfCheese
        reward = 0
        if player_on_cheese:
            reward = 1.0
            self.piecesOfCheese.remove(self.player)

        # Now check if opponent is on another cheese
        opponent_on_cheese = self.enemy in self.piecesOfCheese
        if opponent_on_cheese:
            self.enemy_score += 1.0
            self.piecesOfCheese.remove(self.enemy)

        # Penalizes players if they obtained the same cheese
        if player_on_cheese and self.player == self.enemy:
            reward = 0.5
            self.enemy_score += 0.5
        self.score += reward
        return reward

    def _is_over(self):
        if self.score > self.cheeses / 2 or self.enemy_score > self.cheeses / 2 or (
            (self.score == self.enemy_score) and self.score >= self.cheeses / 2
        ) or self.round > self.round_limit or len(self.piecesOfCheese) == 0:
            return True
        else:
            return False

    def observe(self):
        return np.expand_dims(self.canvas, axis=0)

    def act(self, action):
        enemy_action = self.opponent.turn(None, self.width, self.height, self.enemy,
                                          self.player, self.enemy_score, self.score,
                                          self.piecesOfCheese, 3000)
        self._update_state(action, enemy_action)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over

    def reset(self):
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
