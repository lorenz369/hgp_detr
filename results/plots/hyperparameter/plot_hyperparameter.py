
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

font = { 'size'   : 15}
plt.rc('font', **font)

colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
styles = ['o','s','v','^','D',">","<","*","h","H","+","1","2","3","4","8","p","d","|","_",".",","]

markersize = 10
markerwidth = 2
maxchar = 25

def plot_hyperparameter(df):
    var_names = list(df.columns)
    parameter = df[var_names[0]].values.tolist()
    precision = df[var_names[2]].values.tolist()

    parameter = [str(p) for p in parameter]

    print(var_names)
    print(parameter)
    print(precision)

    fig = plt.figure(1,figsize=(10.67,6.6))
    plt.clf()
    ax = fig.gca()
    # plt.ylim(min(precision), max(precision))  # Set the limits from high to low

    if var_names[0] == 'batch-size':
        param_name = 'Batch Size'
    elif var_names[0] == 'num_queries':
        param_name = 'Number of Queries'
    elif var_names[0] == 'nheads':
        param_name = 'Number of Attention Heads'
    elif var_names[0] == 'dim_feedforward':
        param_name = 'Feedforward Dimension'
    elif var_names[0] == 'hidden_dim':
        param_name = 'Hidden Dimension'
    
    ax.set_xlabel(param_name)
    ax.set_ylabel('Average Precision [IoU=0.00:0.75]')

    ax.plot(parameter, precision, "r-x")

    ax.set_xticks(parameter)
    ax.set_xticklabels([p for p in parameter])

    ax.grid(True)

    plt.title("DETR Average Precision after 3 Epochs for varying " + param_name)

    # ax.text(xlim[0]*1.1,ylim[1]/1.1, '-'.join([filename,flag]), horizontalalignment='left',verticalalignment='top')
#     plt.title('-'.join([filename,flag]))

    plt.savefig('_'.join([param_name])+'.png')
#     plt.savefig('_'.join([filename,flag])+'.eps')

#    plt.show()
