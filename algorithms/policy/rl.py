import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.distributions.categorical import Categorical

# Part 1 - Model to learn the Q-function
# This part defines a simple model that learns a mapping between the canvas and the Q-values
# associated to each possible movements.
# So it's a model with a size corresponding to the canvas size, and four outputs.


class NLinearModels(nn.Module):
    def __init__(self, x_example, number_of_regressors=4, weights_file='saves/policy.pt'):
        super(NLinearModels, self).__init__()
        # n = x_example.shape[-1]
        in_features = x_example.reshape(-1).shape[0]
        # self.linear1 = nn.Linear(in_features, in_features//n)
        self.linear2 = nn.Linear(in_features, number_of_regressors)
        self.drop = nn.Dropout(p=0.3)
        self.weights_file = weights_file

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)  # (Batch, time = past + present, features)
        # x = torch.sigmoid(self.linear1(x)) # (Batch, time * out_features)
        return torch.sigmoid(self.drop(self.linear2(x)))  # (Batch, time * out_features)

    def load(self):
        self.load_state_dict(torch.load(self.weights_file))

    def save(self):
        torch.save(self.state_dict(), self.weights_file)


def train_on_batch(model, inputs, targets, criterion, optimizer):
    # Zero the parameter gradients
    optimizer.zero_grad()
    # Forward + backward + optimize
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()


def policy_loss(logits, targets):
    actions, rewards = targets
    actions, rewards = actions.squeeze(), rewards.squeeze()
    return -(Categorical(logits=logits).log_prob(actions) * rewards).mean()
    # return  (cross_entropy(logits, actions, reduction = 'none') * rewards).mean()

# Part 2 - Experience Replay
# This part has to be read and understood in order to code the main.py file.


class ExperienceReplay(object):
    """
    During gameplay all experiences < s, a, r, s' > are stored in a replay memory.
    During training, batches of randomly drawn experiences are used to generate the input and
    target for training.
    """

    def __init__(self, max_memory=100, discount=.9, discount_norm=True):
        """
        Setup
        max_memory: the maximum number of experiences we want to store
        memory: a list of experiences
        discount: the discount factor for future experience

        In the memory the information whether the game ended at the experience is stored
        seperately in a nested array
        [...,
        [experience, game_over],
        [experience, game_over],
        ...]
        """
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount
        self.discount_norm = discount_norm

    def remember(self, experience, game_over):
        # Save an experience to memory
        self.memory.append([experience, game_over])
        # We don't want to store infinite memories, so if we have too many, we just delete
        # the oldest one
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def _discount_rewards(self):  # Method to discount all rewards based on previous one
        discounted_rewards = np.zeros(len(self.memory))
        R = 0
        for t in reversed(range(len(self.memory))):  # R(t) estimation
            if self.memory[t][1]:
                R = 0  # Restard R if game over
            R = R * self.discount + self.memory[t][0][2]  # R*discount + reward
            discounted_rewards[t] = R
        # Rewards normalized
        if self.discount_norm:
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)
        return discounted_rewards

    def get_batch(self, batch_size=10, num_actions=4, device='cpu'):
        # How many experiences do we have?
        len_memory = len(self.memory)

        # Dimensions of the game field
        env_dim = list(self.memory[0][0][0].shape)
        env_dim[0] = min(len_memory, batch_size)

        # We want to return an input and target vector with inputs from an observed state...
        inputs = torch.zeros(env_dim).to(device)

        # Inference by policy
        drewards = self._discount_rewards()
        rewards = torch.zeros((inputs.shape[0], 1), dtype=torch.float).to(device)
        actions = torch.zeros((inputs.shape[0], 1), dtype=torch.long).to(device)
        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            """
            Here we load one transition <s, a, r> from memory
            state_t: initial state s
            action_t: action taken a
            reward_t: reward earned r
            """
            state, actions[i], _ = self.memory[idx][0]

            # We keep the state as the input
            try:
                inputs[i:i+1] = state
            except:
                inputs[i:i+1] = torch.from_numpy(state)

            # Change rewards for discounted reward
            rewards[i] = drewards[idx]
        return inputs, actions, rewards

    def get_rewards(self):
        return [self.memory[i][0][2] for i in range(len(self.memory))]

    def get_actions(self):
        return [self.memory[i][0][1] for i in range(len(self.memory))]

    def load(self):
        self.memory = pickle.load(open("save_rl/memory.pkl", "rb"))

    def save(self):
        pickle.dump(self.memory, open("save_rl/memory.pkl", "wb"))
