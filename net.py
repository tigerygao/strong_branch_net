import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class StrongBranchNet(torch.nn.Module):
    def __init__(self, num_inputs, hidden_nodes=[30, 50, 50, 10]):
        super(StrongBranchNet, self).__init__()

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
        return x


class StrongBranchMimic():
    def __init__(self, hyperparams=[], options=[]):
        self.net = StrongBranchNet(6)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters())

    def train(self, state, bestcand):
        num_cands = len(state[0])
        input = self.compute_input(state)
        y = [0]*num_cands
        y[bestcand] = 1
        num_repeat_pos = num_cands - 2
        for i in range(num_repeat_pos):
            input = np.concatenate((input, np.expand_dims(input[bestcand], axis=0)), axis=0)
            y.append(1)

        y = Variable(torch.from_numpy(np.expand_dims(np.array(y), axis=1)).float())
        input = Variable(torch.from_numpy(input).float())

        y_hat = self.net(input)
        loss = self.criterion(y_hat, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def trainOnce(self, state, bestcand):
        num_cands = len(state[0])
        input = self.compute_input(state)
        y = [0]*num_cands
        y[bestcand] = 1
        num_repeat_pos = num_cands - 2
        for i in range(num_repeat_pos):
            input = np.concatenate((input, np.expand_dims(input[bestcand], axis=0)), axis=0)
            y.append(1)

        y = Variable(torch.from_numpy(np.expand_dims(np.array(y), axis=1)).float())
        input = Variable(torch.from_numpy(input).float())

        y_hat = self.net(input)
        loss = self.criterion(y_hat, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def compute_input(self, state):
        input = np.expand_dims(np.array(state[0]), axis=1)

        common = np.expand_dims(np.array([state[1]]), axis=1)
        common = np.concatenate((common, np.expand_dims([np.average(state[0])], axis=1)), axis=1) #avg of solution values
        common = np.concatenate((common, np.expand_dims([np.std(state[0])], axis=1)), axis=1) #std of solution values
        common = np.concatenate((common, np.expand_dims([np.average(state[2])], axis=1)), axis=1) #avg of obj. coeffs
        common = np.concatenate((common, np.expand_dims([np.std(state[2])], axis=1)), axis=1) #std of obj. coeffs

        common = np.tile(common,(input.shape[0],1))
        input = np.concatenate((input, common), axis=1)

        return input

        #TODO:
        #x_minus_mu = state[0] - np.average(state[0])
        #x_minus_mu_div_std = x_minus_mu / np.std(state[0])





    def predict(self, state):
        input = Variable(torch.from_numpy(self.compute_input(state)).float())
        y_hat = self.net(input)
        return torch.argmax(y_hat, dim=0)




if __name__ == '__main__':
    torch.manual_seed(0);

    mimic = StrongBranchMimic([])
    state = ([1.5, 4, 3, -2, 4.3, -2.1], 10, [2, 3, 0.4, 1.1, -0.9, 1])
    best_cand = 2

    for i in range(100):
        mimic.train_net(state, best_cand)

    #new_state = ([4, 1.2, 1, -3.4, 4.1, 0], 10, [3, 2, 0.4, 1, -0.9, 1])
    new_state = ([1.5, 4, 3, -2, 4.3, -2.1], 10, [2, 3, 0.4, 1.1, -0.9, 1])
    pred = mimic.predict(new_state)
    print(pred)
