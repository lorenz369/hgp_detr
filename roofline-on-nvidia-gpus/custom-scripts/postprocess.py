import os
import numpy as np
import pandas as pd
from roofline import roofline

datadir="."
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

        dft=df.groupby(['Kernel Name','Metric Name']).sum()
        dfmetric=pd.pivot_table(df, index="Kernel Name", columns="Metric Name", values="Metric Value")

        dfmetric["Count"]=df.groupby(["Kernel Name"]).count()["ID"].div(dfmetric.shape[1])

        dfmetric["Time"]=dfmetric["sm__cycles_elapsed.avg"] \
                        / (dfmetric["sm__cycles_elapsed.avg.per_second"] /dfmetric["Count"] )

        dfmetric["CC FLOPs"]= 2 * dfmetric["sm__sass_thread_inst_executed_op_dfma_pred_on.sum"] \
                            + dfmetric["sm__sass_thread_inst_executed_op_dmul_pred_on.sum"] \
                            + dfmetric["sm__sass_thread_inst_executed_op_dadd_pred_on.sum"] \
                            + 2 * dfmetric["sm__sass_thread_inst_executed_op_ffma_pred_on.sum"] \
                            + dfmetric["sm__sass_thread_inst_executed_op_fmul_pred_on.sum"] \
                            + dfmetric["sm__sass_thread_inst_executed_op_fadd_pred_on.sum"] \
                            + 2 * dfmetric["sm__sass_thread_inst_executed_op_hfma_pred_on.sum"] \
                            + dfmetric["sm__sass_thread_inst_executed_op_hmul_pred_on.sum"] \
                            + dfmetric["sm__sass_thread_inst_executed_op_hadd_pred_on.sum"] 

        dfmetric["TC FLOPs"]= 1024 * dfmetric["sm__inst_executed_pipe_tensor.sum"] # https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
        # The A100 SM diagram is shown in Figure 7. Volta and Turing have eight Tensor Cores per SM, with each Tensor Core performing 64 FP16/FP32 mixed-precision fused multiply-add (FMA) operations per clock. 
        # The A100 SM includes new third-generation Tensor Cores that each perform 256 FP16/FP32 FMA operations per clock. A100 has four Tensor Cores per SM, 
        # which together deliver 1024 dense FP16/FP32 FMA operations per clock, a 2x increase in computation horsepower per SM compared to Volta and Turing.
        dfmetric["all FLOPs"]= dfmetric["CC FLOPs"] + dfmetric["TC FLOPs"]
        
        dfmetric["AI HBM"] = dfmetric["all FLOPs"].div(dfmetric["dram__bytes.sum"])
        dfmetric["AI L2"] = dfmetric["all FLOPs"].div(dfmetric["lts__t_bytes.sum"])
        dfmetric["AI L1"] = dfmetric["all FLOPs"].div(dfmetric["l1tex__t_bytes.sum"])

        dfmetric["GFLOP/s"] = dfmetric["all FLOPs"]/ dfmetric["Time"] /1024/1024/1024
        dfmetric["TC GFLOP/s"] = dfmetric["TC FLOPs"]/ dfmetric["Time"] /1024/1024/1024
#         dfmetric.to_csv("pd_"+tag+".csv")
        dfs[tag]=dfmetric


tags=dfs.keys()
flags=["all"] #"HBM","L2","L1" or "all"
for tag in tags:
    for flag in flags:
        dfm=dfs[tag]
        LABELS = dfm.index.tolist()
        AIL1   = dfm["AI L1"].tolist()
        AIL2   = dfm["AI L2"].tolist()
        AIHBM  = dfm["AI HBM"].tolist()
        FLOPS  = dfm["GFLOP/s"].tolist()

        roofline(tag, FLOPS, AIHBM, AIL2, AIL1, LABELS, flag)

