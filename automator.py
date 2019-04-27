
from admipex1 import admipex1
import time
from datetime import datetime


def makeFileName(ds, sbl, nf, hl, e):
    return datetime.today().strftime('%Y-%m-%d_%H-%M-%S_') \
                    + ds + "_" \
                    + str(sbl) + "_" \
                    + str(nf) + "_" \
                    + "[" + ','.join(str(l) for l in hl) + "]_" \
                    + str(e) \
                    + ".csv";


# just for preallocating size of arrays, if you run more tests than this just increase this number
n = 1000;

# Data files location
dataDir = "data/";

# Results destination
resultsDir = "results/";

# List of things we can change in test runs
dataset                         = [None]*n;
strong_branching_limit          = [None]*n;
num_features                    = [None]*n;
hidden_layers                   = [None]*n; # for now must be 4 layers
epochs                          = [None]*n;
#num_random_seeds                = []; # Hold off for now

# IF YOU CHANGE THE ABOVE make sure to change this string to match:
header = "dataset,strong_branching_limit,num_features,hl1,hl2,hl3,hl4,epochs,num_branch_callbacks,runtime\n"; # runtime in s

# Holds output filenames
outputs = [None]*n;


# If you want to add another test run, put it here

run = 0;
dataset[run]                      = "air04.mps.gz";
strong_branching_limit[run]       = 10000000; # Set very high to be 100% (full?) strong branching
num_features[run]                 = 7; # How modify the set of features from here?
hidden_layers[run]                = [10, 20, 20, 10];
epochs[run]                       = 5;
outputs[run] = makeFileName(dataset[run], strong_branching_limit[run], num_features[run], \
                    hidden_layers[run], epochs[run]);
'''
outputs[run] = datetime.today().strftime('%Y-%M-%D_%H-%M-%S_') \
                    + dataset[run] + "_" \
                    + str(strong_branching_limit[run]) + "_" \
                    + num_features[run] + "_" \
                    + "[" + ','.join(str(l) for l in hidden_layers[run]) + "]_" \
                    + str(epochs[run]) \
                    + ".csv";
'''

run = run + 1;
dataset[run]                      = "air04.mps.gz";
strong_branching_limit[run]       = 45; # Set very high to be 100% (full?) strong branching
num_features[run]                 = 7; # How modify the set of features from here?
hidden_layers[run]                = [10, 20, 20, 10];
epochs[run]                       = 20;
outputs[run] = makeFileName(dataset[run], strong_branching_limit[run], num_features[run], \
                    hidden_layers[run], epochs[run]);


run = run + 1;
dataset[run]                      = "air04.mps.gz";
strong_branching_limit[run]       = 45; # Set very high to be 100% (full?) strong branching
num_features[run]                 = 7; # How modify the set of features from here?
hidden_layers[run]                = [10, 10, 10, 10];
epochs[run]                       = 5;
outputs[run] = makeFileName(dataset[run], strong_branching_limit[run], num_features[run], \
                    hidden_layers[run], epochs[run]);

run = run + 1;
dataset[run]                      = "air04.mps.gz";
strong_branching_limit[run]       = 45; # Set very high to be 100% (full?) strong branching
num_features[run]                 = 7; # How modify the set of features from here?
hidden_layers[run]                = [10, 10, 10, 10];
epochs[run]                       = 20;
outputs[run] = makeFileName(dataset[run], strong_branching_limit[run], num_features[run], \
                    hidden_layers[run], epochs[run]);

run = run + 1;
dataset[run]                      = "air04.mps.gz";
strong_branching_limit[run]       = 45; # Set very high to be 100% (full?) strong branching
num_features[run]                 = 7; # How modify the set of features from here?
hidden_layers[run]                = [30, 70, 70, 20];
epochs[run]                       = 5;
outputs[run] = makeFileName(dataset[run], strong_branching_limit[run], num_features[run], \
                    hidden_layers[run], epochs[run]);

run = run + 1;
dataset[run]                      = "air04.mps.gz";
strong_branching_limit[run]       = 45; # Set very high to be 100% (full?) strong branching
num_features[run]                 = 7; # How modify the set of features from here?
hidden_layers[run]                = [30, 70, 70, 20];
epochs[run]                       = 20;
outputs[run] = makeFileName(dataset[run], strong_branching_limit[run], num_features[run], \
                    hidden_layers[run], epochs[run]);



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

#print("Next try small sized network");


for r in range(run):

    # First open file
    op = open(resultsDir+outputs[run],"w+");

    # Then run the solver
    start=time.clock();
    print("%s, %d, %d, %s, %d" % (dataset[run], strong_branching_limit[run], num_features[run], str(hidden_layers[run]), epochs[run]));
    branch_times = admipex1(dataDir+dataset[run], strong_branching_limit[run], num_features[run], hidden_layers[run], epochs[run]);
    end = time.clock();
    print("Runtime: %s" % str(end-start));
    
    # Then save everything 
    op.write(header);
    op.write("%s,%d,%d,%s,%d,%f,%f,%d,%s\n" % dataset[run], strong_branching_limit[run], num_features[run], \
                        ','.join(str(l) for l in hidden_layers[run]), epochs[run], branch_times, str(end-start));

    #   WHERE GET OUTPUT FROM? PASS FILE HANDLE OR ADD RETURN VAL(S)? 
    # what about units for clock timing?
    op.close();
    


'''
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
'''



print ("end");



