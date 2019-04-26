#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: admipex1.py
# Version 12.9.0
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2009, 2019. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""
Use the node and branch callbacks for optimizing a MIP problem.

To run this example, the user must specify a problem file.

You can run this example at the command line by

    python admipex1.py <filename>
"""

from __future__ import print_function

from math import floor, fabs

import cplex as CPX
import cplex.callbacks as CPX_CB
import sys


# Adding stuff for pytorch
# Taken from https://github.com/pytorch/examples/blob/master/mnist/main.py
# and https://github.com/utkuozbulak/pytorch-custom-dataset-examples/blob/master/src/custom_datasets.py
# and https://www.kaggle.com/pinocookie/pytorch-dataset-and-dataloader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split # Will we need this?
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset  # For custom datasets

class MySolve(CPX_CB.SolveCallback):

    def __call__(self):
        self.times_called += 1
        if self.get_num_nodes() < 1:
            self.solve(self.method.primal)
        else:
            self.solve(self.method.dual)
        status = self.get_cplex_status()
        self.use_solution()


class MyBranch(CPX_CB.BranchCallback):

    def __call__(self):

        print("\n\n**************** Inside branch callback **************** (%d) \n\n" % self.times_called)


        self.times_called += 1
        br_type = self.get_branch_type()
        if (br_type == self.branch_type.SOS1 or
                br_type == self.branch_type.SOS2):
            return
	
        x = self.get_values() # returns solution values at current node

	
        objval = self.get_objective_value() # 
        # self.make_branch(
        #     objval,
        #     constraints=[([[bestj], [1.0]], "G", float(xj_lo + 1))],
        #     node_data=(bestj, xj_lo, "UP"))
        # self.make_branch(
        #     objval,
        #     constraints=[([[bestj], [1.0]], "L", float(xj_lo))],
        #     node_data=(bestj, xj_lo, "DOWN"))

	# Assuming rn that this is CPLEX's selection	
	for i in range(self.get_num_branches()):
		print("i is %d" % i);
		candidate = self.get_branch(i);
		print(str(candidate) + "\n");
		self.make_branch(candidate[0], candidate[1]); # leaving node_data blank for now 
	

	print("\n\n**************** Exiting branch callback ****************\n\n")


class MyNode(CPX_CB.NodeCallback):

    def __call__(self):
        self.times_called += 1
        bestnode = 0
        maxdepth = -1
        maxsiinf = 0.0
        for node in range(self.get_num_remaining_nodes()):
            depth = self.get_depth(node)
            siinf = self.get_infeasibility_sum(node)
            if depth >= maxdepth and (depth > maxdepth or siinf > maxsiinf):
                bestnode = node
                maxdepth = depth
                maxsiinf = siinf
        self.select_node(bestnode)
        # get_node_data retrieves the python object the node was created with
        # print "selected node with data", self.get_node_data(bestnode)


def admipex1(filename):
    c = CPX.Cplex(filename)

    # sys.stdout is the default output stream for log and results
    # so these lines may be omitted
    c.set_log_stream(sys.stdout)
    c.set_results_stream(sys.stdout)

    solve_instance = c.register_callback(MySolve)
    solve_instance.times_called = 0
    branch_instance = c.register_callback(MyBranch)
    branch_instance.times_called = 0
    node_instance = c.register_callback(MyNode)
    node_instance.times_called = 0

    c.parameters.mip.interval.set(1)
    c.parameters.preprocessing.linear.set(0)
    c.parameters.mip.strategy.search.set(
    c.parameters.mip.strategy.search.values.traditional)

    c.parameters.mip.display.set(0); # Reduce amount printed while solving

    # How to set branching strategy: use strong branching 
    c.parameters.mip.strategy.variableselect.set(3) # See table in this link for options https://www.ibm.com/support/knowledgecenter/es/SSSA5P_12.6.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/VarSel.html

    print("\n\n**************** Before running .solve() ****************\n\n")

    c.solve()

    print("\n\n**************** After running .solve() ****************\n\n")

    solution = c.solution

    # solution.get_status() returns an integer code
    print("Solution status = ", solution.get_status(), ":", end=' ')
    # the following line prints the corresponding string
    print(solution.status[solution.get_status()])
    print("Objective value = ", solution.get_objective_value())
    print()
    x = solution.get_values(0, c.variables.get_num() - 1)
    for j in range(c.variables.get_num()):
        if fabs(x[j]) > 1.0e-10:
            print("Column %d: Value = %17.10g" % (j, x[j]))

    print("Solve callback was called ", solve_instance.times_called, "times")
    print("Branch callback was called ", branch_instance.times_called, "times")
    print("Node callback was called ", node_instance.times_called, "times")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: admipex1.py filename")
        print("  filename   Name of a file, with .mps, .lp, or .sav")
        print("             extension, and a possible, additional .gz")
        print("             extension")
        sys.exit(-1)
    admipex1(sys.argv[1])
