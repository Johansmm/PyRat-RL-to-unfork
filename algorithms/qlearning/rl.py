import numpy as np
import pickle
import torch
import torch.nn as nn


# Part 1 - Model to learn the Q-function
# This part defines a simple model that learns a mapping between the canvas and the Q-values
# associated to each possible movements.
# So it's a model with a size corresponding to the canvas size, and four outputs.

class NLinearModels(nn.Module):
    def __init__(self, x_example, number_of_regressors=4, weights_file='saves/qlearning.pt'):
        super(NLinearModels, self).__init__()
        in_features = x_example.reshape(-1).shape[0]
        # self.linear1 = nn.Linear(in_features, 21*15)
        self.linear2 = nn.Linear(in_features, number_of_regressors)
        self.weights_file = weights_file

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        # x = torch.sigmoid(self.linear1(x))
        return self.linear2(x)

    def load(self):
        self.load_state_dict(torch.load(self.weights_file))

    def save(self):
        torch.save(self.state_dict(), self.weights_file)


class NCNNModels(nn.Module):
    def __init__(self, x_example, number_of_regressors=4, weights_file='saves/qlearning.pt'):
        super(NCNNModels, self).__init__()
        in_features = np.array(x_example.shape[:2])
        self.conv1 = nn.Conv2d(x_example.shape[2], 32, 3)
        in_features = (in_features - 3)//1 + 1
        self.conv2 = nn.Conv2d(32, 64, 3)
        in_features = (in_features - 3)//1 + 1
        self.linear = nn.Linear(np.prod(in_features)*64, number_of_regressors)
        self.active_fun = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.weights = weights_file

    def forward(self, x):
        x = x.transpose(1, 3)
        x = self.active_fun(self.conv1(x))
        x = self.active_fun(self.conv2(x))
        x = x.reshape(x.shape[0], -1)
        return self.linear(x)

    def load(self):
        self.load_state_dict(torch.load(self.weights))

    def save(self):
        torch.save(self.state_dict(), self.weights)


def train_on_batch(model, inputs, targets, criterion, optimizer):
    # Zero the parameter gradients
    optimizer.zero_grad()
    # Forward + backward + optimize
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

# Part 2 - Experience Replay
# This part has to be read and understood in order to code the main.py file.


class ExperienceReplay(object):
    """
    During gameplay all experiences < s, a, r, s' > are stored in a replay memory.
    During training, batches of randomly drawn experiences are used to generate the input and
    target for training.
    """

    def __init__(self, max_memory=100, discount=.9):
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

    def remember(self, experience, game_over):
        # Save an experience to memory
        self.memory.append([experience, game_over])
        # We don't want to store infinite memories, so if we have too many, we just delete
        # the oldest one
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10, num_actions=4, device='cpu'):
        # How many experiences do we have?
        len_memory = len(self.memory)

        # Dimensions of the game field
        env_dim = list(self.memory[0][0][0].shape)
        env_dim[0] = min(len_memory, batch_size)

        # We want to return an input and target vector with inputs from an observed state...
        inputs = torch.zeros(env_dim).to(device)  # ...and the target r + gamma * max Q(s’,a’)
        # Note that our target is a matrix, with possible fields not only for the action taken
        # but also for the other possible actions. The actions do not take the same values as
        # the prediction to not affect them.
        Q = torch.zeros((inputs.shape[0], num_actions)).to(device)

        # We randomly draw experiences to learn from
        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            """
            Here we load one transition <s, a, r, s’> from memory
            state_t: initial state s
            action_t: action taken a
            reward_t: reward earned r
            state_tp1: the state that followed s’
            """
#            idx = -1
            state, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]
            # We also need to know whether the game ended at this state
            # if not self.memory[idx][1]: reward_t = disc_rewards[idx + 1] # Game not end -> reward
            # in future step
            # else: reward_t = disc_rewards[idx] # Game end -> reward in present step
            # reward_t = disc_rewards[idx]

            # Add the state s to the input
            inputs[i:i+1] = state
            # First, we fill the target values with the predictions of the model.
            # They will not be affected by training (since the training loss for them is 0).
            with torch.no_grad():
                Q[i] = model(state)

                # If the game ended, the expected reward Q(s,a) should be the final reward r.
                # Otherwise the target value is r + gamma * max Q(s’,a’)
                Q[i, action_t] = reward_t  # If the game ended, the reward is the final reward

                if not game_over:
                    next_round = model(state_tp1)
                    Q[i, action_t] += self.discount * torch.max(next_round)
        return inputs, Q

    def get_rewards(self):
        return [self.memory[i][0][2] for i in range(len(self.memory))]

    def get_actions(self):
        return [self.memory[i][0][1] for i in range(len(self.memory))]

    def load(self):
        self.memory = pickle.load(open("save_rl/memory.pkl", "rb"))

    def save(self):
        pickle.dump(self.memory, open("save_rl/memory.pkl", "wb"))
