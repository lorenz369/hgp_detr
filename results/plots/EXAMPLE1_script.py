"""

E. Wes Bethel, Copyright (C) 2022

October 2022

Description: This code loads a .csv file and creates a 3-variable plot

Inputs: the named file "sample_data_3vars.csv"

Outputs: displays a chart with matplotlib

Dependencies: matplotlib, pandas modules

Assumptions: developed and tested using Python version 3.8.8 on macOS 11.6

"""

import pandas as pd
import matplotlib.pyplot as plt


fname = "runtime_cpu.csv"
df = pd.read_csv(fname, comment="#")
print(df)

var_names = list(df.columns)

print("var names =", var_names)

# split the df into individual vars
# assumption: column order - 0=problem size, 1=blas time, 2=basic time

concurrency = df[var_names[0]].values.tolist()
runtime = df[var_names[1]].values.tolist()


plt.title("Sobel Filter on CPU with OpenMP Parallelism: Runtime vs. Concurrency")

locs = [1, 2, 4, 8, 16]

xlocs = [i for i in range(len(concurrency))]

plt.xticks(xlocs, locs)

# here, we are plotting the raw values read from the input .csv file, which
# we interpret as being "time" that maps directly to the y-axis.
#
# what if we want to plot MFLOPS instead? How do we compute MFLOPS from
# time and problem size? You may need to add some code here to compute
# MFLOPS, then modify the plt.plot() lines below to plot MFLOPS rather than time.

# direct: number of operations: 1 arithmetic operation
# vector: number of operations: 1 arithmetic operation, 1 memory access
# Code 1: number of operations: 1 arithmetic operation, 1 memory access

# for i in range(len(problem_sizes)):
#     conc_4[i] = serial[i]/conc_4[i]
#     conc_16[i] = serial[i]/conc_16[i]
#     conc_64[i] = serial[i]/conc_64[i]

plt.plot(runtime, "r-x")

# print("runtime: \n")
# for i in range(len(concurrency)):
#     print(f"{conc_4[i]} \n")



#plt.xscale("log")
#plt.yscale("log")

plt.xlabel("level of concurrency")
plt.ylabel("runtime (seconds)")

varNames = ["runtime of openmp parallelized sobel filter"]
plt.legend(varNames, loc="best")

plt.grid(axis='both')

plt.show()

# EOF
