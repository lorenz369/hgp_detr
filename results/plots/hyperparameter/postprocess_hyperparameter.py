import os
import re
import numpy as np
import pandas as pd
from plot_hyperparameter import plot_hyperparameter

datadir="."
files=[x for x in os.listdir(datadir) if x.endswith(".txt") and x.startswith("output_")]
files.sort()
files=[os.path.join(datadir,file) for file in files]
param_dic={}
for file in files:
    tag, ext = os.path.splitext(os.path.basename(file))
    tag = tag[len("output_"):]
    parameter = re.sub(r'\d+', '', tag)
    value = int(re.search(r'\d+$', tag).group())
    avg_processing_time, avg_loss_computation_time, avg_backward_pass_time, epochs  = 0, 0, 0, 0

    with open(file,"r") as f:
        cnt=0
        while True:
            ln=f.readline()
            if not ln:
                break
            cnt+=1
            if "processing time" in ln:
                epochs += 1
                match = re.search(r"(\d+\.\d+) seconds", ln)
                if match:
                    mseconds = float(match.group(1)) * 1000  # Convert the matched string to a float
                    avg_processing_time += mseconds
                else:
                    print("No seconds value found in the text.")

            if "loss computation time" in ln:
                match = re.search(r"(\d+\.\d+) seconds", ln)
                if match:
                    mseconds = float(match.group(1)) * 1000  # Convert the matched string to a float
                    avg_loss_computation_time += mseconds
                else:
                    print("No seconds value found in the text.")

            if "backward pass time" in ln:
                match = re.search(r"(\d+\.\d+) seconds", ln)
                if match:
                    mseconds = float(match.group(1)) * 1000  # Convert the matched string to a float
                    avg_backward_pass_time += mseconds
                else:
                    print("No seconds value found in the text.")

            if "Average Precision  (AP) @[ IoU=0.00:0.75 | area=   all | maxDets=100 ]" in ln:
                ap_iou_0 = float(ln.split('=')[-1].strip())

            if "Training time" in ln:
                match = re.search(r"(\d+):(\d+):(\d+)", ln)
                if match:
                    hours, minutes, seconds = map(int, match.groups())
                    total_minutes = hours * 60 + minutes + seconds / 60.0  # Convert everything to minutes
                else:
                    print("Time format not found in the text.")
    
    if parameter not in param_dic:
        param_dic[parameter] = {
            parameter: [],
            'Total Training time': [],
            'Average Precision [IoU=0.00:0.75]': [],
            'Average Processing time': [],
            'Average Loss computation time': [],
            'Average Backward pass time': []
        }
    param_dic[parameter][parameter].append(value)
    param_dic[parameter]['Total Training time'].append(total_minutes)
    param_dic[parameter]['Average Precision [IoU=0.00:0.75]'].append(ap_iou_0)
    param_dic[parameter]['Average Processing time'].append(avg_processing_time / epochs)
    param_dic[parameter]['Average Loss computation time'].append(avg_loss_computation_time / epochs)
    param_dic[parameter]['Average Backward pass time'].append(avg_backward_pass_time / epochs)

dfs = {}
for parameter in param_dic:
    df = pd.DataFrame(param_dic[parameter])
    df.sort_values(by=[parameter], inplace=True)
    dfs[parameter] = df


#         dfmetric.to_csv("pd_"+tag+".csv")
    
for parameter in dfs:
    print(dfs[parameter])
    plot_hyperparameter(dfs[parameter])
        

