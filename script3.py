
from admipex1 import admipex1
import time
from datetime import datetime
import subprocess

def makeFileName(ds, sbl, nf, hl, e):
    return datetime.today().strftime('%Y-%m-%d_%H-%M-%S_') \
                    + ds + "_" \
                    + str(sbl) + "_" \
                    + str(nf) + "_" \
                    + "[" + ','.join(str(l) for l in hl) + "]_" \
                    + str(e) \
                    + ".csv";

def pushToGit(filename):

    subprocess.call(["git", "add", "."]);
    print("Added");
    subprocess.call(["git", "commit", "-m", "AUTOMATED COMMIT: " + filename]);
    print("Commited");
    subprocess.call(["git", "push"]);
    print("Pushed!");



# just for preallocating size of arrays, if you run more tests than this just increase this number
n = 1000;

# Data files location
dataDir = "data/";

# Results destination
resultsDir = "results/";

# GPU? 0 or 1
gpu = 0

# Dataset if you want all of them to be same (None if you dont);
#ds = None;
#ds = "air04.mps.gz"; # For testing purposes, (make sure to set sbl to <12)
ds = "cov1075.mps.gz";

# SBL if you want all of them to be same (None if you dont);
#sbl = None;
sbl = 100;

# SBL if you want all of them to be same (None if you dont);
#nf = None;
nf = 7;

# List of things we can change in test runs
dataset                         = [ds]*n;
strong_branching_limit          = [sbl]*n;
num_features                    = [nf]*n;
hidden_layers                   = [None]*n; # for now must be 4 layers
epochs                          = [None]*n;
#num_random_seeds                = []; # Hold off for now

# IF YOU CHANGE THE ABOVE make sure to change this string to match:
header = "dataset,strong_branching_limit,num_features,hl1,hl2,hl3,hl4,epochs,num_branch_callbacks,runtime\n"; # runtime in s

# Holds output filenames
outputs = [None]*n;

# If you want to add another test run, put it here

run = 0;
#dataset[run]                      = "app1-2.mps.gz";
strong_branching_limit[run]       = 10000000; # Set very high to be 100% (full?) strong branching
#num_features[run]                 = 7; # How modify the set of features from here?
hidden_layers[run]                = [15, 30, 30, 15];
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
#dataset[run]                      = "air04.mps.gz";
#dataset[run]                      = "app1-2.mps.gz";
#strong_branching_limit[run]       = 45; # Set very high to be 100% (full?) strong branching
#num_features[run]                 = 7; # How modify the set of features from here?
hidden_layers[run]                = [15, 30, 30, 15];
epochs[run]                       = 20;
outputs[run] = makeFileName(dataset[run], strong_branching_limit[run], num_features[run], \
                    hidden_layers[run], epochs[run]);


run = run + 1;
#dataset[run]                      = "air04.mps.gz";
#dataset[run]                      = "app1-2.mps.gz";
#strong_branching_limit[run]       = 45; # Set very high to be 100% (full?) strong branching
#num_features[run]                 = 7; # How modify the set of features from here?
hidden_layers[run]                = [30, 100, 100, 30];
epochs[run]                       = 5;
outputs[run] = makeFileName(dataset[run], strong_branching_limit[run], num_features[run], \
                    hidden_layers[run], epochs[run]);

run = run + 1;
#dataset[run]                      = "air04.mps.gz";
#strong_branching_limit[run]       = 45; # Set very high to be 100% (full?) strong branching
#num_features[run]                 = 7; # How modify the set of features from here?
hidden_layers[run]                = [30, 100, 100, 30];
epochs[run]                       = 20;
outputs[run] = makeFileName(dataset[run], strong_branching_limit[run], num_features[run], \
                    hidden_layers[run], epochs[run]);

run = run + 1;
#dataset[run]                      = "air04.mps.gz";
#strong_branching_limit[run]       = 45; # Set very high to be 100% (full?) strong branching
#num_features[run]                 = 7; # How modify the set of features from here?
hidden_layers[run]                = [7, 10, 10, 7];
epochs[run]                       = 5;
outputs[run] = makeFileName(dataset[run], strong_branching_limit[run], num_features[run], \
                    hidden_layers[run], epochs[run]);

run = run + 1;
#dataset[run]                      = "air04.mps.gz";
#strong_branching_limit[run]       = 45; # Set very high to be 100% (full?) strong branching
#num_features[run]                 = 7; # How modify the set of features from here?
hidden_layers[run]                = [7, 10, 10, 7];
epochs[run]                       = 20;
outputs[run] = makeFileName(dataset[run], strong_branching_limit[run], num_features[run], \
                    hidden_layers[run], epochs[run]);


run = run + 1;
#dataset[run]                      = "air04.mps.gz";
#strong_branching_limit[run]       = 45; # Set very high to be 100% (full?) strong branching
#num_features[run]                 = 7; # How modify the set of features from here?
hidden_layers[run]                = [7, 7, 7, 7];
epochs[run]                       = 20;
outputs[run] = makeFileName(dataset[run], strong_branching_limit[run], num_features[run], \
                    hidden_layers[run], epochs[run]);


run = run + 1;
#dataset[run]                      = "air04.mps.gz";
#strong_branching_limit[run]       = 45; # Set very high to be 100% (full?) strong branching
#num_features[run]                 = 7; # How modify the set of features from here?
hidden_layers[run]                = [7, 7, 7, 7];
epochs[run]                       = 30;
outputs[run] = makeFileName(dataset[run], strong_branching_limit[run], num_features[run], \
                    hidden_layers[run], epochs[run]);


run = run + 1;
#dataset[run]                      = "air04.mps.gz";
#strong_branching_limit[run]       = 45; # Set very high to be 100% (full?) strong branching
#num_features[run]                 = 7; # How modify the set of features from here?
hidden_layers[run]                = [30, 70, 70, 20];
epochs[run]                       = 30;
outputs[run] = makeFileName(dataset[run], strong_branching_limit[run], num_features[run], \
                    hidden_layers[run], epochs[run]);


run = run + 1;
#dataset[run]                      = "air04.mps.gz";
#strong_branching_limit[run]       = 45; # Set very high to be 100% (full?) strong branching
#num_features[run]                 = 7; # How modify the set of features from here?
hidden_layers[run]                = [30, 70, 70, 20];
epochs[run]                       = 30;
outputs[run] = makeFileName(dataset[run], strong_branching_limit[run], num_features[run], \
                    hidden_layers[run], epochs[run]);


run = run + 1;
#dataset[run]                      = "air04.mps.gz";
#strong_branching_limit[run]       = 45; # Set very high to be 100% (full?) strong branching
#num_features[run]                 = 7; # How modify the set of features from here?
hidden_layers[run]                = [30, 70, 70, 20];
epochs[run]                       = 40;
outputs[run] = makeFileName(dataset[run], strong_branching_limit[run], num_features[run], \
                    hidden_layers[run], epochs[run]);


'''
run = run + 1;
dataset[run]                      = "air04.mps.gz";
strong_branching_limit[run]       = 45; # Set very high to be 100% (full?) strong branching
num_features[run]                 = 7; # How modify the set of features from here?
hidden_layers[run]                = [30, 70, 70, 20];
epochs[run]                       = 20;
outputs[run] = makeFileName(dataset[run], strong_branching_limit[run], num_features[run], \
                    hidden_layers[run], epochs[run]);
'''

'''
print(dataset);
print(strong_branching_limit);
print(num_features);
print(hidden_layers);
print(epochs);
print(outputs);
'''


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


for r in range(run+1):

    print(r);

    # First open file
    op = open(resultsDir+outputs[r],"w+");

    # Then run the solver
    start=time.clock();
    print("%s, %d, %d, %s, %d" % (dataset[r], strong_branching_limit[r], num_features[r], str(hidden_layers[r]), epochs[r]));
    (branch_times, predicts, sol_type, sol_obj_val) = admipex1(dataDir+dataset[r], strong_branching_limit[r], num_features[r], hidden_layers[r], epochs[r]);
    end = time.clock();
    print("Runtime: %s" % str(end-start));
    
    
    print("\n\n**************** end important values ****************\n\n")


    # Then save everything 
    op.write(header);
    op.write("%s,%d,%d,%s,%d,%d,%s\n" % (\
            dataset[r], \
            strong_branching_limit[r], \
            num_features[r], \
            ','.join(str(l) for l in hidden_layers[r]), \
            epochs[r], \
            branch_times, \
            end-start));

    op.write("\nsolution status,objective value\n%s,%s\n" % (sol_type, sol_obj_val));

    op.write("\nbranch iter,predicted var\n");
    for i in range(len(predicts)):
        # REMEMBER ONCE THIS WORKS UNCOMMENT AUTO GIT PUSHING
        op.write(str(predicts[i][0]) + ',' + str(predicts[i][1]) + "\n");
    

    '''
    op.write(dataset[run] + ',');
    op.write(str(strong_branching_limit[run]) + ',');
    op.write(str(num_features[run]) + ',');
    op.write(str(','.join(str(l) for l in hidden_layers[run])[run]) + ',');
    op.write(str(epochs[run]) + ',');
    op.write([run] + ',');
    op.write([run] + ',');
    '''


    #   WHERE GET OUTPUT FROM? PASS FILE HANDLE OR ADD RETURN VAL(S)? 
    # what about units for clock timing?
    op.close();
    

    # Push to git! 
    pushToGit(outputs[r]);


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



