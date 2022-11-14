"""Save experiences to train a model with reinforcement learning
"""
import numpy as np
from collections import deque
import pickle

from torch.utils.data.dataset import IterableDataset


class ExperienceReplay(object):
    """During gameplay all experiences < s, a, r, s' > are stored in a replay memory.

    During training, batches of randomly drawn experiences are used to generate the input and
    target for training.

    Parameters
    ----------
    max_memory : int, optional
        The maximum number of experiences we want to store, by default 100
    discount : float, optional
        The discount factor for future experience, by default 0.9
    memory_path : str, optional
        Attached in save/load functions to handler a file with memory values,
            by default "saves/memory.pkl"

    Attributes
    ----------
    memory: list
        A list of experiences, stored seperately in a nested array:
            [..., [experience, game_over], [experience, game_over], ...]
    """

    def __init__(self, max_memory=100, discount=0.9, memory_path: str = "saves/memory.pkl"):
        self.memory = deque(maxlen=max_memory)
        self.discount = discount
        self.memory_path = memory_path

    def remember(self, experience, game_over):
        """Save the tuple [experience, game_over] into the memory

        Parameters
        ----------
        experience : List[np.ndarray, int, float, np.ndarray]
            A list of states, rewards and action
        game_over : bool
            Whether the experience completed the game or not
        """
        # Save an experience to memory
        self.memory.append([experience, game_over])

    def get_batch(self, batch_size):
        """Return a random batch of experience

        Parameters
        ----------
        batch_size : int
            Number of experience to return

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Experience in batches
        """
        # How many experiences do we have?
        len_memory = len(self.memory) - 1

        # Batch size will be replaced if it is less than total of memory
        batch_size = min(len_memory, batch_size)

        # We randomly draw experiences to learn from
        indices = np.random.choice(len_memory, batch_size, replace=False)
        experience, game_over = zip(*(self.memory[idx] for idx in indices))
        states, actions, rewards, next_states = zip(*experience)

        # If the game ended, the expected reward Q(s,a) should be the final reward r.
        # Otherwise the target value is r + gamma * max Q(s',a')
        next_rewards = [self.memory[idx.item()][0][2] if not go else 0.0 for idx,
                        go in zip(indices + 1, game_over)]

        states = np.stack(states, axis=0).astype(np.float32)
        next_states = np.stack(next_states, axis=0).astype(np.float32)
        actions = np.stack(actions, axis=0)
        rewards = np.stack(rewards, axis=0) + self.discount * np.stack(next_rewards, axis=0)
        return states, actions, rewards.astype(np.float32), next_states

    def get_rewards(self):
        """Get the list of rewards in the memory

        Returns
        -------
        List[float]
            Rewards
        """
        return np.stack([self.memory[i][0][2] for i in range(len(self.memory))])

    def get_actions(self):
        """Get the list of actions in the memory

        Returns
        -------
        List[int]
            actions
        """
        return np.stack([self.memory[i][0][1] for i in range(len(self.memory))])

    def load(self):
        """Load previous memory"""
        self.memory = pickle.load(open(self.memory_path, "rb"))

    def save(self):
        """Save the memory'"""
        pickle.dump(self.memory, open(self.memory_path, "wb"))


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceReplay which will be updated with
    new experiences during training.

    Take from "lightning_examples/reinforce-learning-DQN#Memory"

    Parameters
    ----------
    buffer : ExperienceReplay
        replay buffer
    buffer_size : int, optional
        number of experiences to sample at a time in memory, by default all
    """

    def __init__(self, buffer: ExperienceReplay, buffer_size: int = None) -> None:
        self.buffer = buffer
        memlen = buffer.memory.maxlen
        if buffer_size is None:
            buffer_size = memlen
        elif buffer_size > memlen:
            raise ValueError(
                f"Buffer size ({buffer_size}) can not exceed the memory lenght ({memlen}).")
        self.buffer_size = buffer_size

    def __iter__(self):
        states, actions, rewards, new_states = self.buffer.get_batch(self.buffer_size)
        for i in range(len(actions)):
            yield states[i], actions[i], rewards[i], new_states[i]

    def remember(self, experience, game_over):
        self.buffer.remember(experience, game_over)
