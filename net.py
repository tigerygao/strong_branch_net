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
        self.net = StrongBranchNet(19)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters())

    def train_net(self, state, bestcand):
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
        ##DYNAMIC FEATURES##
        input = np.expand_dims(np.array(state[0]), axis=1) #solution vals
        x_minus_mu = state[0] - np.average(state[0])
        x_minus_mu_div_std = np.expand_dims(x_minus_mu / np.std(state[0]), axis=1)
        input = np.concatenate((input, x_minus_mu_div_std), axis=1) #(solution val - mean)/std

        fractionality = np.expand_dims(np.absolute((input[:,0] - np.floor(input[:,0])) - 0.5), axis=1)
        input = np.concatenate((input, fractionality), axis=1) #solution fracs
        frac_minus_mu_div_std = (fractionality - np.average(fractionality))/np.std(fractionality)
        input = np.concatenate((input, frac_minus_mu_div_std), axis=1) #(frac - mean)/std

        input = np.hstack((input, np.expand_dims(np.array(state[2]), axis=1))) #obj. coeffs
        coeff_minus_mu = state[2] - np.average(state[2])
        coeff_minus_mu_div_std = np.expand_dims(coeff_minus_mu / np.std(state[2]), axis=1)
        input = np.concatenate((input, coeff_minus_mu_div_std), axis=1) #(obj. coeff - mean)/std

        ##STATIC FEATURES##
        common = np.expand_dims(np.array([state[1]]), axis=1) #obj. val

        common = np.concatenate((common, np.expand_dims([np.average(state[0])], axis=1)), axis=1) #avg of solution values
        common = np.concatenate((common, np.expand_dims([np.std(state[0])], axis=1)), axis=1) #std of solution values
        common = np.concatenate((common, np.expand_dims([np.max(state[0])], axis=1)), axis=1) #max solution val
        common = np.concatenate((common, np.expand_dims([np.min(state[0])], axis=1)), axis=1) #min solution val

        common = np.concatenate((common, np.expand_dims([np.average(fractionality)], axis=1)), axis=1) #avg of frac
        common = np.concatenate((common, np.expand_dims([np.std(fractionality)], axis=1)), axis=1) #std of frac
        common = np.concatenate((common, np.expand_dims([np.max(fractionality)], axis=1)), axis=1) #max frac
        common = np.concatenate((common, np.expand_dims([np.min(fractionality)], axis=1)), axis=1) #min frac

        common = np.concatenate((common, np.expand_dims([np.average(state[2])], axis=1)), axis=1) #avg of obj. coeffs
        common = np.concatenate((common, np.expand_dims([np.std(state[2])], axis=1)), axis=1) #std of obj. coeffs
        common = np.concatenate((common, np.expand_dims([np.max(state[2])], axis=1)), axis=1) #max obj. coeff
        common = np.concatenate((common, np.expand_dims([np.min(state[2])], axis=1)), axis=1) #min obj. coeff


        common = np.tile(common,(input.shape[0],1))
        input = np.concatenate((input, common), axis=1)


        return input




    def predict(self, state):
        input = Variable(torch.from_numpy(self.compute_input(state)).float())
        y_hat = self.net(input)
        return torch.argmax(y_hat, dim=0)



    def compute_input_automated(self, state):
        #state[0]: solution vector
        #state[1]: objective value
        #state[2]: objective coefficient vector

        static = np.expand_dims(np.array([state[1]]), axis=1)

        #solution values
        input = np.expand_dims(np.array(state[0]), axis=1)
        stats = self.compute_stats(input)
        input = np.hstack((input, stats[-1]))
        for i in range(len(stats)-1):
            static = np.hstack((static, np.expand_dims(np.array([stats[i]]),axis=1)))
        '''
        static = np.hstack((static, np.expand_dims(np.array([stats[0]]),axis=1)))
        static = np.hstack((static, np.expand_dims(np.array([stats[1]]),axis=1)))
        static = np.hstack((static, np.expand_dims(np.array([stats[2]]),axis=1)))
        static = np.hstack((static, np.expand_dims(np.array([stats[3]]),axis=1)))
        '''

        #solution fractionalities
        fractionalities = np.expand_dims(np.absolute((input[:,0] - np.floor(input[:,0])) - 0.5), axis=1)
        stats = self.compute_stats(fractionalities)
        input = np.hstack((input, fractionalities, stats[-1]))
        for i in range(len(stats)-1):
            static = np.hstack((static, np.expand_dims(np.array([stats[i]]),axis=1)))

        #objective coefficients
        obj_coeffs = np.expand_dims(np.array(state[2]), axis=1)
        stats = self.compute_stats(obj_coeffs)
        input = np.hstack((input, obj_coeffs, stats[-1]))
        for i in range(len(stats)-1):
            static = np.hstack((static, np.expand_dims(np.array([stats[i]]),axis=1)))

        static = np.tile(static,(input.shape[0],1))
        input = np.concatenate((input, static), axis=1)
        return input


    def compute_stats(self, arr):
        mean = np.average(arr)
        std = np.std(arr)
        max = np.max(arr)
        min = np.min(arr)
        normalized = (arr - mean)/std
        return (mean, std, max, min, normalized)



if __name__ == '__main__':
    mimic = StrongBranchMimic([])
    state = ([4, -3.3, 1.1, -3.2, 4.6, 0], 5, [-3, 2, 0.4, 3.1, -2.9, 1.3])
    #state = ([1.5, 4, 3, -2, 4.3, -2.1], 10, [2, 3, 0.4, 1.1, -0.9, 1])
    best_cand = 2

    np.set_printoptions(suppress=True)
    print(mimic.compute_input(state))
    print(mimic.compute_input_automated(state))
    print(mimic.compute_input(state) == mimic.compute_input_automated(state))

    #mimic.train_net(state, best_cand)
    #for i in range(100):
    #    mimic.train_net(state, best_cand)

    #new_state = ([4, 1.2, 1, -3.4, 4.1, 0], 10, [3, 2, 0.4, 1, -0.9, 1])
    new_state = ([1.5, 4, 3, -2, 4.3, -2.1], 10, [2, 3, 0.4, 1.1, -0.9, 1])
    #for i in range(10):
    #    pred = mimic.predict(new_state)
    #    print(pred)
