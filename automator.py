
from admipex1 import admipex1



print ("start");
print ("First get full SB by increasing sb_limit");
dataset = "data/air04.mps.gz";
strong_branching_limit = 100; # Set very high to be 100% (full?) strong branching
num_features = 6; # How modify the set of features from here?
hidden_layers = [30, 50, 50, 10];
epochs = 3;

print("%s, %d, %d, %s, %d" % (dataset, strong_branching_limit, num_features, str(hidden_layers), epochs));
admipex1(dataset, strong_branching_limit, num_features, hidden_layers, epochs);


print("Next try small sized network");

dataset = "data/air04.mps.gz";
strong_branching_limit = 45; # Set very high to be 100% (full?) strong branching
num_features = 6; # How modify the set of features from here?
hidden_layers = [10, 20, 20, 10];
epochs = 5;

print("%s, %d, %d, %s, %d" % (dataset, strong_branching_limit, num_features, str(hidden_layers), epochs));
admipex1(dataset, strong_branching_limit, num_features, hidden_layers, epochs);

dataset = "data/air04.mps.gz";
strong_branching_limit = 45; # Set very high to be 100% (full?) strong branching
num_features = 6; # How modify the set of features from here?
hidden_layers = [10, 20, 20, 10];
epochs = 10;

print("%s, %d, %d, %s, %d" % (dataset, strong_branching_limit, num_features, str(hidden_layers), epochs));
admipex1(dataset, strong_branching_limit, num_features, hidden_layers, epochs);


dataset = "data/air04.mps.gz";
strong_branching_limit = 45; # Set very high to be 100% (full?) strong branching
num_features = 6; # How modify the set of features from here?
hidden_layers = [10, 20, 20, 10];
epochs = 20;

print("%s, %d, %d, %s, %d" % (dataset, strong_branching_limit, num_features, str(hidden_layers), epochs));
admipex1(dataset, strong_branching_limit, num_features, hidden_layers, epochs);


print("Now med sized network");

dataset = "data/air04.mps.gz";
strong_branching_limit = 45; # Set very high to be 100% (full?) strong branching
num_features = 6; # How modify the set of features from here?
hidden_layers = [30, 50, 50, 10];
epochs = 5;

print("%s, %d, %d, %s, %d" % (dataset, strong_branching_limit, num_features, str(hidden_layers), epochs));
admipex1(dataset, strong_branching_limit, num_features, hidden_layers, epochs);


dataset = "data/air04.mps.gz";
strong_branching_limit = 45; # Set very high to be 100% (full?) strong branching
num_features = 6; # How modify the set of features from here?
hidden_layers = [30, 50, 50, 10];
epochs = 10;

print("%s, %d, %d, %s, %d" % (dataset, strong_branching_limit, num_features, str(hidden_layers), epochs));
admipex1(dataset, strong_branching_limit, num_features, hidden_layers, epochs);


dataset = "data/air04.mps.gz";
strong_branching_limit = 45; # Set very high to be 100% (full?) strong branching
num_features = 6; # How modify the set of features from here?
hidden_layers = [30, 50, 50, 10];
epochs = 20;

print("%s, %d, %d, %s, %d" % (dataset, strong_branching_limit, num_features, str(hidden_layers), epochs));
admipex1(dataset, strong_branching_limit, num_features, hidden_layers, epochs);



print("Now try big a$$ network");

dataset = "data/air04.mps.gz";
strong_branching_limit = 45; # Set very high to be 100% (full?) strong branching
num_features = 6; # How modify the set of features from here?
hidden_layers = [30, 100, 100, 10];
epochs = 5;

print("%s, %d, %d, %s, %d" % (dataset, strong_branching_limit, num_features, str(hidden_layers), epochs));
admipex1(dataset, strong_branching_limit, num_features, hidden_layers, epochs);


dataset = "data/air04.mps.gz";
strong_branching_limit = 45; # Set very high to be 100% (full?) strong branching
num_features = 6; # How modify the set of features from here?
hidden_layers = [30, 100, 100, 10];
epochs = 10;

print("%s, %d, %d, %s, %d" % (dataset, strong_branching_limit, num_features, str(hidden_layers), epochs));
admipex1(dataset, strong_branching_limit, num_features, hidden_layers, epochs);


dataset = "data/air04.mps.gz";
strong_branching_limit = 45; # Set very high to be 100% (full?) strong branching
num_features = 6; # How modify the set of features from here?
hidden_layers = [30, 100, 100, 10];
epochs = 20;

print("%s, %d, %d, %s, %d" % (dataset, strong_branching_limit, num_features, str(hidden_layers), epochs));
admipex1(dataset, strong_branching_limit, num_features, hidden_layers, epochs);



print ("end");



