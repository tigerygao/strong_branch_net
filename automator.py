
from admipex1 import admipex1
import time
from datetime import datetime

# just for preallocating size of arrays, if you run more tests than this just increase this number
max = 1000;

# List of things we can change in test runs
dataset                         = [];
strong_branching_limit          = [];
num_features                    = [];
hidden_layers                   = []; # for now must be 4 layers
epochs                          = [];
#num_random_seeds                = []; # Hold off for now

# Holds output filenames
outputs = [];


# If you want to add another test run, put it here

run = 0;
dataset[run]                      = "air04.mps.gz";
strong_branching_limit[run]       = 10000000; # Set very high to be 100% (full?) strong branching
num_features[run]                 = 7; # How modify the set of features from here?
hidden_layers[run]                = [10, 20, 20, 10];
epochs[run]                       = 5;
outputs[run] = datetime.today().strftime('%Y-%M-%D_%H-%M-%S_') \
                    + dataset[run] + "_" \
                    + str(strong_branching_limit[run]) + "_" \
                    + num_features[run] + "_" \
                    + "[" + ','.join(str(l) for l in hidden_layers[run]) + "]_" \
                    + str(epochs[run]) \
                    + ".csv";


run = run + 1;
dataset[run]                      = "air04.mps.gz";
strong_branching_limit[run]       = 45; # Set very high to be 100% (full?) strong branching
num_features[run]                 = 7; # How modify the set of features from here?
hidden_layers[run]                = [10, 20, 20, 10];
epochs[run]                       = 5;

run = run + 1;
dataset[run]                      = "air04.mps.gz";
strong_branching_limit[run]       = 45; # Set very high to be 100% (full?) strong branching
num_features[run]                 = 7; # How modify the set of features from here?
hidden_layers[run]                = [10, 20, 20, 10];
epochs[run]                       = 5;

run = run + 1;
dataset[run]                      = "air04.mps.gz";
strong_branching_limit[run]       = 45; # Set very high to be 100% (full?) strong branching
num_features[run]                 = 7; # How modify the set of features from here?
hidden_layers[run]                = [10, 20, 20, 10];
epochs[run]                       = 5;

run = run + 1;
dataset[run]                      = "air04.mps.gz";
strong_branching_limit[run]       = 45; # Set very high to be 100% (full?) strong branching
num_features[run]                 = 7; # How modify the set of features from here?
hidden_layers[run]                = [10, 20, 20, 10];
epochs[run]                       = 5;

run = run + 1;
dataset[run]                      = "air04.mps.gz";
strong_branching_limit[run]       = 45; # Set very high to be 100% (full?) strong branching
num_features[run]                 = 7; # How modify the set of features from here?
hidden_layers[run]                = [10, 20, 20, 10];
epochs[run]                       = 5;



print("Doing %d test runs" % run);

''' # Just strong branching by setting sbl very high
print ("start");
print ("First get full SB by increasing sb_limit");
dataset = "data/air04.mps.gz";
strong_branching_limit = 100000; # Set very high to be 100% (full?) strong branching
num_features = 6; # How modify the set of features from here?
hidden_layers = [30, 50, 50, 10];
epochs = 3;

start = time.clock();
print("%s, %d, %d, %s, %d" % (dataset, strong_branching_limit, num_features, str(hidden_layers), epochs));
admipex1(dataset, strong_branching_limit, num_features, hidden_layers, epochs);
end = time.clock();
print("Runtime: %s" % str(end-start));
'''

print("Next try small sized network");

dataset = "data/air04.mps.gz";
#dataset = "data/sentoy.mps";
strong_branching_limit = 45; # Set very high to be 100% (full?) strong branching
num_features = 7; # How modify the set of features from here?
hidden_layers = [10, 20, 20, 10];
epochs = 5;

start = time.clock();
print("%s, %d, %d, %s, %d" % (dataset, strong_branching_limit, num_features, str(hidden_layers), epochs));
admipex1(dataset, strong_branching_limit, num_features, hidden_layers, epochs);
end = time.clock();
print("Runtime: %s" % str(end-start));

#dataset = "data/air04.mps.gz";
#strong_branching_limit = 45; # Set very high to be 100% (full?) strong branching
num_features = 6; # How modify the set of features from here?
hidden_layers = [10, 20, 20, 10];
epochs = 10;

start = time.clock();
print("%s, %d, %d, %s, %d" % (dataset, strong_branching_limit, num_features, str(hidden_layers), epochs));
admipex1(dataset, strong_branching_limit, num_features, hidden_layers, epochs);
end = time.clock();
print("Runtime: %s" % str(end-start));


#dataset = "data/air04.mps.gz";
#strong_branching_limit = 45; # Set very high to be 100% (full?) strong branching
num_features = 6; # How modify the set of features from here?
hidden_layers = [10, 20, 20, 10];
epochs = 20;

start = time.clock();
print("%s, %d, %d, %s, %d" % (dataset, strong_branching_limit, num_features, str(hidden_layers), epochs));
admipex1(dataset, strong_branching_limit, num_features, hidden_layers, epochs);
end = time.clock();
print("Runtime: %s" % str(end-start));


print("Now med sized network");

#dataset = "data/air04.mps.gz";
#strong_branching_limit = 45; # Set very high to be 100% (full?) strong branching
num_features = 6; # How modify the set of features from here?
hidden_layers = [30, 50, 50, 10];
epochs = 5;

start = time.clock();
print("%s, %d, %d, %s, %d" % (dataset, strong_branching_limit, num_features, str(hidden_layers), epochs));
admipex1(dataset, strong_branching_limit, num_features, hidden_layers, epochs);
end = time.clock();
print("Runtime: %s" % str(end-start));


#dataset = "data/air04.mps.gz";
#strong_branching_limit = 45; # Set very high to be 100% (full?) strong branching
num_features = 6; # How modify the set of features from here?
hidden_layers = [30, 50, 50, 10];
epochs = 10;

start = time.clock();
print("%s, %d, %d, %s, %d" % (dataset, strong_branching_limit, num_features, str(hidden_layers), epochs));
admipex1(dataset, strong_branching_limit, num_features, hidden_layers, epochs);
end = time.clock();
print("Runtime: %s" % str(end-start));


#dataset = "data/air04.mps.gz";
#strong_branching_limit = 45; # Set very high to be 100% (full?) strong branching
num_features = 6; # How modify the set of features from here?
hidden_layers = [30, 50, 50, 10];
epochs = 20;

start = time.clock();
print("%s, %d, %d, %s, %d" % (dataset, strong_branching_limit, num_features, str(hidden_layers), epochs));
admipex1(dataset, strong_branching_limit, num_features, hidden_layers, epochs);
end = time.clock();
print("Runtime: %s" % str(end-start));



print("Now try big a$$ network");

#dataset = "data/air04.mps.gz";
#strong_branching_limit = 45; # Set very high to be 100% (full?) strong branching
num_features = 6; # How modify the set of features from here?
hidden_layers = [30, 100, 100, 10];
epochs = 5;

start = time.clock();
print("%s, %d, %d, %s, %d" % (dataset, strong_branching_limit, num_features, str(hidden_layers), epochs));
admipex1(dataset, strong_branching_limit, num_features, hidden_layers, epochs);
end = time.clock();
print("Runtime: %s" % str(end-start));


#dataset = "data/air04.mps.gz";
#strong_branching_limit = 45; # Set very high to be 100% (full?) strong branching
num_features = 6; # How modify the set of features from here?
hidden_layers = [30, 100, 100, 10];
epochs = 10;

start = time.clock();
print("%s, %d, %d, %s, %d" % (dataset, strong_branching_limit, num_features, str(hidden_layers), epochs));
admipex1(dataset, strong_branching_limit, num_features, hidden_layers, epochs);
end = time.clock();
print("Runtime: %s" % str(end-start));


#dataset = "data/air04.mps.gz";
#strong_branching_limit = 45; # Set very high to be 100% (full?) strong branching
num_features = 6; # How modify the set of features from here?
hidden_layers = [30, 100, 100, 10];
epochs = 20;

start = time.clock();
print("%s, %d, %d, %s, %d" % (dataset, strong_branching_limit, num_features, str(hidden_layers), epochs));
admipex1(dataset, strong_branching_limit, num_features, hidden_layers, epochs);
end = time.clock();
print("Runtime: %s" % str(end-start));



print ("end");



