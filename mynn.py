from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Taken from https://github.com/pytorch/examples/blob/master/mnist/main.py


from torch.autograd import Variable
from torch.utils.data.dataset import Dataset  # For custom datasets

# Taken from https://github.com/utkuozbulak/pytorch-custom-dataset-examples/blob/master/src/custom_datasets.py
# and https://www.kaggle.com/pinocookie/pytorch-dataset-and-dataloader

class GraphsDataset(Dataset):

	def __init__(self, file_path, transform=None):
		self.data = pd.read_csv(file_path)
		self.width = self.data.shape[1]
		self.transform = transform
		#self.transform = transforms.Compose([transforms.ToTensor()]);

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		# load image as ndarray type (Height * Width * Channels)
		# be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
		# in this example, i don't use ToTensor() method of torchvision.transforms
		# so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
		#image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((1, 28, 28))

		edges_df = self.data.iloc[index, 0:self.width-1];
		label_df = self.data.iloc[index, self.width-1]

		#edges = torch.tensor(edges_df);
		#label = torch.tensor(label_df);

		edges = edges_df.values.astype(np.float32);
		label = label_df;

		if self.transform is not None:
			#image = self.transform(image)
			edges = self.transform(edges)

		return edges, label



"""
class GraphsTesting(Dataset):

	def __init__(self, file_path, transform=None):
		self.data = pd.read_csv(file_path)
		
		[self.height, self.width] = self.data.shape;
		self.data = self.data[self.height*pctTrain:self.height, :];

		self.width = self.data.shape[1];
		self.transform = transform
		#self.transform = transforms.Compose([transforms.ToTensor()]);

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		# load image as ndarray type (Height * Width * Channels)
		# be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
		# in this example, i don't use ToTensor() method of torchvision.transforms
		# so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
		#image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((1, 28, 28))

		edges_df = self.data.iloc[index, 0:self.width-1];
		label_df = self.data.iloc[index, self.width-1]

		#edges = torch.tensor(edges_df);
		#label = torch.tensor(label_df);

		edges = edges_df.values.astype(np.float32);
		label = label_df;

		if self.transform is not None:
			#image = self.transform(image)
			edges = self.transform(edges)

		return edges, label
"""


class XLayerNet(nn.Module):
	def __init__(self, D_in, H):
		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		
		D_in: input dimension
		H: dimension of hidden layers
		D_out: output dimension // Just put it in H
		"""
		super(XLayerNet, self).__init__()
		self.linear1 = nn.Linear(D_in, H[0])
		self.linear2 = nn.Linear(H[0], H[1])
		self.linear3 = nn.Linear(H[1], H[2])
		#self.linear4 = nn.Linear(H[2], H[3])
		#self.linear5 = nn.Linear(H[3], H[4])

	def forward(self, x):
		"""
		In the forward function we accept a Variable of input data and we must 
		return a Variable of output data. We can use Modules defined in the 
		constructor as well as arbitrary operators on Variables.
		"""
		x = F.relu(self.linear1(x))
		x = F.relu(self.linear2(x))
		x = F.relu(self.linear3(x))
		#x = F.relu(self.linear4(x))
		#x = F.relu(self.linear5(x))

		return F.log_softmax(x, dim=1)



	
def train(args, model, device, train_loader, optimizer, epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss /= len(test_loader.dataset)

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))

def main():
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch Custom Dataset Example')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N',
						help='input batch size for training (default: 64)')
	parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
						help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=10, metavar='N',
						help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
						help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
						help='SGD momentum (default: 0.5)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
						help='how many batches to wait before logging training status')
	parser.add_argument('--graph', type=int, default=4, metavar='N',
        				        help='4, 5 or 6');
	parser.add_argument('--pct-train', type=int, default=50, metavar='N',
        				        help='% of dataset that will be training');

	parser.add_argument('--save-model', action='store_true', default=True,
						help='For Saving the current Model')
	args = parser.parse_args()
	use_cuda = not args.no_cuda and torch.cuda.is_available()

	torch.manual_seed(args.seed)

	device = torch.device("cuda" if use_cuda else "cpu")

	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

	cols = -1;
	custom_graphs = None;

	if args.graph == 4:
		graphs_training = GraphsDataset(str(args.pct_train) + "split/4_train.csv")
		graphs_testing = GraphsDataset(str(args.pct_train) + "split/4_test.csv")
		cols = 6;
		#args.batch_size = 16;
	elif args.graph == 5:
		#custom_graphs = DatasetGraphs('5.csv')
		#graphs_training = GraphsDataset('5.csv')
		#graphs_testing = GraphsDataset('5.csv')
		graphs_training = GraphsDataset(str(args.pct_train) + "split/5_train.csv")
		graphs_testing = GraphsDataset(str(args.pct_train) + "split/5_test.csv")
		cols = 10;
		#args.batch_size = 128;
	elif args.graph == 6:
		#custom_graphs = DatasetGraphs('6.csv')
		#graphs_training = GraphsDataset('6.csv')
		#graphs_testing = GraphsDataset('6.csv')
		graphs_training = GraphsDataset(str(args.pct_train) + "split/6_train.csv")
		graphs_testing = GraphsDataset(str(args.pct_train) + "split/6_test.csv")
		cols = 15;
	else:
		print ("Unknown graph size");
		exit;


	print(args.batch_size);
	args.batch_size = (int)(args.batch_size / 10);
	print(args.batch_size);

	
	"""
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST('../data', train=True, download=True,
					   transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ])),
		batch_size=args.batch_size, shuffle=True, **kwargs)
	"""

	train_loader = torch.utils.data.DataLoader(
		graphs_training, batch_size=args.batch_size, shuffle=True, **kwargs);


	"""
	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST('../data', train=False, transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ])),
		batch_size=args.test_batch_size, shuffle=True, **kwargs)
	"""
	
	test_loader = torch.utils.data.DataLoader(
		graphs_testing, batch_size=args.test_batch_size, shuffle=True, **kwargs);



	#model = Net().to(device)
	layers = [100, 100, 2]; # LAST ELEMENT MUST BE 2
	#model = MnistCNNModel()
	model = XLayerNet(cols, layers).to(device);
	#model = TwoLayerNet(6, layers).to(device, dtype=torch.float32); # didn't work

	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

	for epoch in range(1, args.epochs + 1):
		train(args, model, device, train_loader, optimizer, epoch)
		test(args, model, device, test_loader)

	if (args.save_model):
		torch.save(model.state_dict(), "hw1_graph_" + str(args.graph) + ".pt")
		
if __name__ == '__main__':
	main()
