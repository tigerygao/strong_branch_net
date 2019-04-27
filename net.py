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
    def __init__(self, num_inputs, hidden_nodes, epochs, hyperparams=[], options=[]):
        #self.net = StrongBranchNet(num_inputs, hidden_nodes)
        self.NUM_INPUTS = num_inputs
        self.net = StrongBranchNet(num_inputs)
        #self.net.cuda(); # Sad!

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.net.to(self.device);

        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.epochs = epochs;
        self.trainingData = [];
        self.trainingLabels = [];

    def addSample(self, state, bestcand):
        #print("Inside add Sample");
        #self.trainingData = [self.trainingData, state];
        #self.trainingLabels = [self.trainingLabels, bestcand];

        self.trainingData.append(state);
        self.trainingLabels.append(bestcand);

    def train(self, state, bestcand):
        #print("Made it to train");

        num_cands = len(state[0])
        #print("Made it to 1");
        input = self.compute_input(state)
        #print("Made it to 2");
        y = [0]*num_cands
        #print("Made it to 3");
        y[bestcand] = 1
        #print("Made it to 4");
        num_repeat_pos = num_cands - 2
        #print("Made it to train loop");
        for i in range(num_repeat_pos):
            input = np.concatenate((input, np.expand_dims(input[bestcand], axis=0)), axis=0)
            y.append(1)

        y = Variable(torch.from_numpy(np.expand_dims(np.array(y), axis=1)).float())
        input = Variable(torch.from_numpy(input).float())

        #print("Sending to GPU");
        y = y.to(self.device);
        input = input.to(self.device);

        y_hat = self.net(input)
        loss = self.criterion(y_hat, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    #def trainOnce(self, state2d=trainingData, bestcand2d=trainingLabels): Can't do this, so adding getter functions
    #def trainOnce(self, state2d, bestcand2d):
    def trainOnce(self, data, bestcands):
        #print("inside trainOnce");
        inputs = np.empty((0,self.NUM_INPUTS), float)
        ys = []
        for i in range(len(data)):
            input = self.compute_input(data[i])
            num_cands = len(data[i][0])
            y = [0]*num_cands
            y[bestcands[i]] = 1
            num_repeat_pos = num_cands - 2
            for j in range(num_repeat_pos):
                input = np.concatenate((input, np.expand_dims(input[bestcands[i]], axis=0)), axis=0)
                y.append(1)
            inputs = np.vstack((inputs, input))
            ys = ys + y

        for e in range(self.epochs):
            print("Epoch %d" % e);
            '''
            for i in range(len(state2d)): # Maybe randomize this instead of doing it in the same order? TODO
                state = state2d[i];
                bestcand = bestcand2d[i]
                #print("before self.train");
                self.train(state, bestcand);
            '''
            # Need to give all tensors at once for GPU efficiency!!!

            y = Variable(torch.from_numpy(np.expand_dims(np.array(ys), axis=1)).float())
            input = Variable(torch.from_numpy(inputs).float())

            #print("Sending to GPU");
            y = y.to(self.device);
            input = input.to(self.device);

            y_hat = self.net(input)
            loss = self.criterion(y_hat, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()




    def getTrainingData(self):
        return self.trainingData;

    def getTrainingLabels(self):
        return self.trainingLabels;


    def compute_input(self, state):
        input = np.expand_dims(np.array(state[0]), axis=1)
        fractionality = np.absolute((input - np.floor(input)) - 0.5)
        input = np.concatenate((input, fractionality), axis=1)

        #print("Got to 0");
        common = np.expand_dims(np.array([state[1]]), axis=1)
        #print("Got to 1");
        #print(state[0]);
        #print(type(state[0]));
        common = np.concatenate((common, np.expand_dims([np.average(state[0])], axis=1)), axis=1) #avg of solution values
        #print("Got to 2");
        common = np.concatenate((common, np.expand_dims([np.std(state[0])], axis=1)), axis=1) #std of solution values
        #print("Got to 3");
        common = np.concatenate((common, np.expand_dims([np.average(state[2])], axis=1)), axis=1) #avg of obj. coeffs
        #print("Got to 4");
        common = np.concatenate((common, np.expand_dims([np.std(state[2])], axis=1)), axis=1) #std of obj. coeffs
        #print("Got to 5");

        common = np.tile(common,(input.shape[0],1))
        input = np.concatenate((input, common), axis=1)

        return input

        #TODO:
        #x_minus_mu = state[0] - np.average(state[0])
        #x_minus_mu_div_std = x_minus_mu / np.std(state[0])





    def predict(self, state):
        #print("Inside predict");
        input = Variable(torch.from_numpy(self.compute_input(state)).float())
        #print("after input");
        y_hat = self.net(input.to(self.device))
        #print("after y_hat");
        return torch.argmax(y_hat, dim=0)




if __name__ == '__main__':
    torch.manual_seed(0);

    mimic = StrongBranchMimic(7, [20, 20, 20, 20], 10)

    state = ([1.5, 4, 3, -2, 4.3, -2.1], 10, [2, 3, 0.4, 1.1, -0.9, 1])
    best_cand = 2

    for i in range(100):
        mimic.addSample(state, best_cand)

    mimic.trainOnce(mimic.getTrainingData(), mimic.getTrainingLabels());

    #new_state = ([4, 1.2, 1, -3.4, 4.1, 0], 10, [3, 2, 0.4, 1, -0.9, 1])
    new_state = ([1.5, 4, 3, -2, 4.3, -2.1], 10, [2, 3, 0.4, 1.1, -0.9, 1])
    pred = mimic.predict(new_state)
    print(pred)
