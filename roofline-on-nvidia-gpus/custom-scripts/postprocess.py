import sys
import os
import numpy as np
import pandas as pd
from roofline import roofline
from roofline_pu import roofline_pu

debug, flops = False, False
datadir="."
if len(sys.argv) > 1:
    debug = True if sys.argv[1]=='--debug' else False
    flops = True if sys.argv[1]=='--flops' else False
    datadir = sys.argv[1] if sys.argv[1].startswith('.') else "."


files=[x for x in os.listdir(datadir) if x.endswith(".csv") and x.startswith("output")]
files.sort()
files=[os.path.join(datadir,file) for file in files]
dfs={}
for file in files:
    tag, ext = os.path.splitext(os.path.basename(file))
    dfs[tag]=pd.DataFrame()
    with open(file,"r") as f:
        cnt=0
        while True:
            ln=f.readline()
            if not ln:
                break
            cnt+=1
            if "Host Name" in ln:
                break
        df = pd.read_csv(file, skiprows=cnt-1)
        df['Metric Value'] = pd.to_numeric(df['Metric Value'].str.replace(',', ''), errors='coerce')
        df['Metric Value'] = df['Metric Value'].astype(np.float64)

        if debug:
            print(df)

        dft=df.groupby(['Kernel Name','Metric Name']).sum()

        if debug:
            print()
            print("dft=df.groupby(['Kernel Name','Metric Name']).sum()")
            print(dft)

        dfmetric=pd.pivot_table(dft, index="Kernel Name", columns="Metric Name", values="Metric Value")

        if debug:
            print()
            print("dfmetric=pd.pivot_table(dft, index='Kernel Name', columns='Metric Name', values='Metric Value')")
            print(dfmetric)

        dfmetric["Count"]=df.groupby(["Kernel Name"]).count()["ID"].div(dfmetric.shape[1])

        dfmetric["Time"]=dfmetric["sm__cycles_elapsed.avg"] \
                        / (dfmetric["sm__cycles_elapsed.avg.per_second"] /dfmetric["Count"] )

        dfmetric["FP64 FLOPS"] = 2 * dfmetric["sm__sass_thread_inst_executed_op_dfma_pred_on.sum"] \
                            + dfmetric["sm__sass_thread_inst_executed_op_dmul_pred_on.sum"] \
                            + dfmetric["sm__sass_thread_inst_executed_op_dadd_pred_on.sum"]
        
        dfmetric["FP32 FLOPS"] = 2 * dfmetric["sm__sass_thread_inst_executed_op_ffma_pred_on.sum"] \
                            + dfmetric["sm__sass_thread_inst_executed_op_fmul_pred_on.sum"] \
                            + dfmetric["sm__sass_thread_inst_executed_op_fadd_pred_on.sum"]
        
        dfmetric["FP16 FLOPS"] = 2 * dfmetric["sm__sass_thread_inst_executed_op_hfma_pred_on.sum"] \
                            + dfmetric["sm__sass_thread_inst_executed_op_hmul_pred_on.sum"] \
                            + dfmetric["sm__sass_thread_inst_executed_op_hadd_pred_on.sum"]

        dfmetric["CC FLOPs"]= dfmetric["FP64 FLOPS"] + dfmetric["FP32 FLOPS"] + dfmetric["FP16 FLOPS"]

        dfmetric["TC FLOPs"]= 1024 * dfmetric["sm__inst_executed_pipe_tensor.sum"] # https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
        # The A100 SM diagram is shown in Figure 7. Volta and Turing have eight Tensor Cores per SM, with each Tensor Core performing 64 FP16/FP32 mixed-precision fused multiply-add (FMA) operations per clock. 
        # The A100 SM includes new third-generation Tensor Cores that each perform 256 FP16/FP32 FMA operations per clock. A100 has four Tensor Cores per SM, 
        # which together deliver 1024 dense FP16/FP32 FMA operations per clock, a 2x increase in computation horsepower per SM compared to Volta and Turing.
        dfmetric["all FLOPs"]= dfmetric["CC FLOPs"] + dfmetric["TC FLOPs"]

        conditions = [
            dfmetric["all FLOPs"] == 0,
            dfmetric["all FLOPs"] == dfmetric["FP16 FLOPS"],
            dfmetric["all FLOPs"] == dfmetric["FP32 FLOPS"],
            dfmetric["all FLOPs"] == dfmetric["FP64 FLOPS"],
            dfmetric["all FLOPs"] == dfmetric["TC FLOPs"],
            dfmetric["all FLOPs"] == dfmetric["CC FLOPs"]
        ]

        choices = ['MemOP', 'FP16', 'FP32', 'FP64', 'TC', 'CC']

        # Default to 'Mixed' if none of the above conditions are true
        dfmetric["ProcUnit"] = np.select(conditions, choices, default='TC/CC')

        dfmetric["bytes_moved"] = dfmetric["dram__bytes.sum"] + dfmetric["lts__t_bytes.sum"] + dfmetric["l1tex__t_bytes.sum"]
        
        dfmetric["AI HBM"] = dfmetric["all FLOPs"].div(dfmetric["dram__bytes.sum"])
        dfmetric["AI L2"] = dfmetric["all FLOPs"].div(dfmetric["lts__t_bytes.sum"])
        dfmetric["AI L1"] = dfmetric["all FLOPs"].div(dfmetric["l1tex__t_bytes.sum"])
        dfmetric["AI"] = dfmetric["all FLOPs"].div(dfmetric["bytes_moved"])


        dfmetric["GFLOP/s"] = dfmetric["all FLOPs"]/ dfmetric["Time"] /1024/1024/1024
        dfmetric["TC GFLOP/s"] = dfmetric["TC FLOPs"]/ dfmetric["Time"] /1024/1024/1024
        dfmetric["FP64 GFLOP/s"] = dfmetric["FP64 FLOPS"]/ dfmetric["Time"] /1024/1024/1024
        dfmetric["FP32 GFLOP/s"] = dfmetric["FP32 FLOPS"]/ dfmetric["Time"] /1024/1024/1024
        dfmetric["FP16 GFLOP/s"] = dfmetric["FP16 FLOPS"]/ dfmetric["Time"] /1024/1024/1024


        if debug or flops:
            print()
            print(dfmetric[["FP16 FLOPS",
                            "FP32 FLOPS",
                            "FP64 FLOPS",
                            "CC FLOPs",
                            "TC FLOPs",
                            "ProcUnit"]])

        # dfmetric.to_csv("pd_"+tag+".csv")
        dfs[tag]=dfmetric


tags=dfs.keys()
flags=["AI"] #"AI","HBM","L2","L1" or "all"
for tag in tags:
    for flag in flags:
        dfm=dfs[tag]
        LABELS = dfm.index.tolist()
        AI     = dfm["AI"].tolist()
        AIL1   = dfm["AI L1"].tolist()
        AIL2   = dfm["AI L2"].tolist()
        AIHBM  = dfm["AI HBM"].tolist()
        FLOPS  = dfm["GFLOP/s"].tolist()
        PU     = dfm["ProcUnit"].tolist()

        title = tag.split("output_")[1]
        roofline_pu(title, FLOPS, AI, AIHBM, AIL2, AIL1, LABELS, PU, flag)

