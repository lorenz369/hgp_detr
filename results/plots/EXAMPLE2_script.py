#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: wes
Created: Thu Sep 30 05:51:28 PDT 2021

Description: this code generates a 2D "heatmap" style plot using sample data that
is hard-coded into the code.

Inputs: none, all problem parameters are hard-coded.

Outputs: a plot showing the heatmap, displayed to the screen

Dependencies: matplotlib, numpy

Assumptions: Developed and Tested with Python 3.8.8 on MacOS 11.6
'''

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

fname = "runtime_gpu.csv"
df = pd.read_csv(fname, comment="#")
print(df)

var_names = list(df.columns)

print("var names =", var_names)

# split the df into individual vars
# assumption: column order - 0=problem size, 1=blas time, 2=basic time

blocks = df[var_names[0]].values.tolist()
tpb32 = df[var_names[1]].values.tolist()
tpb64 = df[var_names[2]].values.tolist()
tpb128 = df[var_names[3]].values.tolist()
tpb256 = df[var_names[4]].values.tolist()
tpb512 = df[var_names[5]].values.tolist()
tpb1024 = df[var_names[6]].values.tolist()

# for i in range(len(blocks)):
#     tpb32[i] = round(tpb32[i], 2)
#     tpb64[i] = round(tpb64[i], 2)
#     tpb128[i] = round(tpb128[i], 2)
#     tpb256[i] = round(tpb256[i], 2)
#     tpb512[i] = round(tpb512[i], 2)
#     tpb1024[i] = round(tpb1024[i], 2)

for i in range(len(blocks)):
    tpb32[i] = round(tpb32[i], 1) if int(tpb32[i]) > 10 else round(tpb32[i], 2)
    tpb64[i] = round(tpb64[i], 1) if int(tpb64[i]) > 10 else round(tpb64[i], 2)
    tpb128[i] = round(tpb128[i], 1) if int(tpb128[i]) > 10 else round(tpb128[i], 2)
    tpb256[i] = round(tpb256[i], 1) if int(tpb256[i]) > 10 else round(tpb256[i], 2)
    tpb512[i] = round(tpb512[i], 1) if int(tpb512[i]) > 10 else round(tpb512[i], 2)
    tpb1024[i] = round(tpb1024[i], 1) if int(tpb1024[i]) > 10 else round(tpb1024[i], 2)


threads_per_block = ['32', '64', '128', '256', '512', '1024'] # y axis, 6 of them
thread_blocks = ["1", "4", "16", "64", "256", "1024", "4096"] # x axis, 7 of them

# runtime = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
#                     [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
#                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
#                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
#                     [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
#                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1]])

runtime = np.array([tpb32,
                    tpb64,
                    tpb128,
                    tpb256,
                    tpb512,
                    tpb1024])


fig, ax = plt.subplots()
im = ax.imshow(runtime, cmap="coolwarm")

# We want to show all ticks...
ax.set_xticks(np.arange(len(thread_blocks)))
ax.set_yticks(np.arange(len(threads_per_block)))
# ... and label them with the respective list entries
ax.set_xticklabels(thread_blocks)
ax.set_yticklabels(threads_per_block)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(threads_per_block)): # y axis
    for j in range(len(thread_blocks)): # x axis
        text = ax.text(j, i, runtime[i, j],
                       ha="center", va="center", color="k")

ax.set_title("Runtime in ms on GPU at Varying Block Size and Number of Blocks")
ax.set_ylabel('Threads per block')
ax.set_xlabel('Block Sizes')
fig.colorbar(im, ax=ax)
fig.tight_layout()
plt.show()

# EOF
