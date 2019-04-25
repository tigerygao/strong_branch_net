import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class StrongBranchNet(torch.nn.Module):
    def __init__(self, num_inputs, hidden_nodes=[50, 100, 100, 20]):
        self.linear1 = nn.Linear(num_inputs, hidden_nodes[0])
        self.linear2 = nn.Linear(hidden_nodes[0], hidden_nodes[1])
        self.linear3 = nn.Linear(hidden_nodes[1], hidden_nodes[2])
        self.linear4 = nn.Linear(hidden_nodes[2], hidden_nodes[3])
        self.linear5 = nn.Linear(hidden_nodes[3], 1)

        self.activ1 = nn.ReLU()
        self.activ2 = nn.ReLU()
        self.activ3 = nn.ReLU()
        self.activ4 = nn.ReLU()
        self.activ5 = nn.Sigmoid()

    def forward(self, input):
        x = self.activ1(self.linear1(input))
        x = self.activ2(self.linear2(x))
        x = self.activ3(self.linear3(x))
        x = self.activ4(self.linear4(x))
        x = self.activ5(self.linear5(x))
        return score


class StrongBranchMimic():
    def __init__(hyperparams=[], options=[]):
        self.net = StrongBranchNet(10)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(net.parameters())

    def train_net(state, bestcand):
        num_cands = len(state[0])
        input = self.compute_input(state)
        y = [0]*num_cands
        y[bestcand] = 1
        num_repeat_pos = num_cands - 2
        for i in range(num_repeat_pos):
            input.append(input[bestcand])
            y.append(1)
        y = np.asarray(y)

        y_hat = self.net(input)
        loss = self.criterion(y_hat, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_input(state):
        input = np.expand_dims(np.array(state[0]), axis=1)
        #input = np.concatenate(input, np.full((input.shape[0],1), state[1]), axis=1)
        common = np.asarray([state[1]])
        common = np.append(common, [np.average(state[0])])
        common = np.append(common, [np.])


        common = np.tile(common,(input.shape[0],1))
        input = np.concatenate(input, common, axis=1)



    def predict(state):
        input = self.compute_input(state)
        y_hat = self.net(input)
        return torch.argmax(y_hat, dim=0)




if __name__ == '__main__':
    mimic = StrongBranchMimic(None, None)
    state = ([1.5, 4, 3, -2, 4.3, -2.1], 10, [2, 3, 0.4, 1.1, -0.9, 1])
    best_cand = 5

    mimic.train_net(state, best_cand)
