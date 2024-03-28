import os
import json
import argparse
import time
import math
import numpy as np
from subprocess import check_call
def read_results(log_filename, max_trials):
    objs = []
    runtimes = []
    with open(log_filename, "r") as f:
        lines = f.readlines()
        for line in lines: # read the obj and runtime for each trial
            if line[:6] == "-Trial":
                line = line.strip().split(" ")
                assert len(objs) + 1 == int(line[-3])
                objs.append(int(line[-2]))
                runtimes.append(float(line[-1]))
        final_obj = int(lines[-6].split(",")[0].split(" ")[-1])
        if len(objs) == 0: # solved by subgradient optimization
            ascent_runtime = float(lines[69].split(" ")[-2])
            return [final_obj] * max_trials, [ascent_runtime]* max_trials
        else:
            assert objs[-1] == final_obj
            return objs, runtimes
args = {'dataset':'data/test/tsp2000_test.txt',
        'model_path':'logs/tsp50/best_val_checkpoint.tar',
        'n_samples':128,
        'batch_size':16,
        'lkh_trials':1000,
        'gcn_lkh_trials':1000
        }
args=argparse.Namespace(**args)
dataset_name = args.dataset[:-4].split("/")[-1]
method="LKH"
os.makedirs("result/" + dataset_name + "/" + method + "_para", exist_ok=True) 
os.makedirs("result/" + dataset_name + "/" + method + "_log", exist_ok=True) 
os.makedirs("result/" + dataset_name + "/tsp", exist_ok=True)
num_nodes = 2000
rerun=True
results=[]
for i in range(1):
    instance_name=str(i)
    para_filename = "result/" + dataset_name + "/LKH_para/" + instance_name + ".para"
    log_filename = "result/" + dataset_name + "/LKH_log/" + instance_name + ".log"
    instance_filename = "result/" + dataset_name + "/tsp/" + instance_name + ".tsp"
    with open(log_filename, "w") as f:
        check_call(["./LKH", para_filename], stdout=f)
    results.append(read_results(log_filename, 1000))
results = np.array(results)
dataset_objs = results[:, 0, :].mean(0)
dataset_runtimes = results[:, 1, :].sum(0)
trials = 1
while trials <= dataset_objs.shape[0]:
    print ("------experiments of trials: %d ------" % (trials))
    print ("GCNModel+LKH   %d %ds" % (dataset_objs[trials - 1], dataset_runtimes[trials - 1]))
    trials *= 10