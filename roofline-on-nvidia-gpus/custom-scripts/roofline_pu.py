
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

def roofline_pu(filename, FLOPS, AI, AIHBM, AIL2=None, AIL1=None, LABELS=None, UNIT=0, flag='HBM'):

    if not FLOPS:
        print('FLOPS can not be empty!')
        return
    if max(FLOPS)==0:
        print('FLOPS are all 0s!')
        return
    if (not AIHBM) and (not AIL2) and (not AIL1):
        print('AIHBM, AIL2 and AIL1 can not all be empty!')
        return
    if (len(FLOPS) != len(AIHBM)) or (len(FLOPS) != len(AIL2)) or (len(FLOPS) != len(AIL1)):
        print('FLOPS needs to have the same length as AI!')
        return
    if (flag != 'HBM') and (flag != 'L2') and (flag != 'L1') and (flag != 'all') and (flag != 'AI'):
        print('flag needs to be one of AI, HBM, L2, L1, and all!')
        return
    LABELS = [x[:maxchar] for x in LABELS]

    # Source: https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
    l2_roof = 2996.77 * 2.3
    memRoofs = [('L1', 54000.), ('L2', l2_roof),  ('HBM', 1555)] 
    cmpRoofs = [('TF32 Tensor', 155.9), ('FP64', 9.7), ('FP32', 19.5), ('FP16', 78)]

    fig = plt.figure(1,figsize=(10.67,6.6))
    plt.clf()
    ax = fig.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Arithmetic Intensity [FLOPs/Byte]')
    ax.set_ylabel('Performance [GFLOP/sec]')

    nx   = 10000
    xmin = -3 
    xmax = 3
    ymin = 1
    ymax = 165000

    ax.set_xlim(10**xmin, 10**xmax)
    ax.set_ylim(ymin, ymax)

    ixx = int(nx*0.02)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    scomp_x_elbow  = []
    scomp_ix_elbow = []
    smem_x_elbow   = []
    smem_ix_elbow  = []

    x = np.logspace(xmin,xmax,nx)
    for roof in cmpRoofs:
        for ix in range(1,nx):
            if float(memRoofs[0][1] * x[ix]) >= roof[1]*1024 and (memRoofs[0][1] * x[ix-1]) < roof[1]*1024:
                scomp_x_elbow.append(x[ix-1])
                scomp_ix_elbow.append(ix-1)
                break

    for roof in memRoofs:
        for ix in range(1,nx):
            if (cmpRoofs[0][1]*1024 <= roof[1] * x[ix] and cmpRoofs[0][1]*1024 > roof[1] * x[ix-1]):
                smem_x_elbow.append(x[ix-1])
                smem_ix_elbow.append(ix-1)
                break

    for i in range(len(cmpRoofs)):
        roof = cmpRoofs[i][1]*1024
        y = np.ones(len(x)) * roof
        ax.plot(x[scomp_ix_elbow[i]:],y[scomp_ix_elbow[i]:],c='k',ls='-',lw='2')

    for i in range(len(memRoofs)):
        roof = memRoofs[i][1]
        y = x * roof
        ax.plot(x[:smem_ix_elbow[i]+1],y[:smem_ix_elbow[i]+1],c='k',ls='-',lw='2')

    memops = 0
    tc = 0

    for i in range(len(AIHBM)):

        conditions = [
            UNIT[i] == 'FP16',
            UNIT[i] == 'FP32',
            UNIT[i] == 'FP64',
            UNIT[i] == 'TC',
            UNIT[i] == 'CC',
            UNIT[i] == 'TC/CC'
        ]

        choices = [0, 1, 2, 3, 4, 5]

        pu_index = int(np.select(conditions, choices, default='6'))

        if pu_index == 6:
            memops += 1
        elif pu_index == 3 or pu_index == 5:
            tc += 1

        if flag == 'AI':
            ax.plot(float(AI[i]),float(FLOPS[i]),c=colors[pu_index],marker=styles[pu_index],\
                    linestyle='None',ms=markersize,markerfacecolor='none',\
                    markeredgewidth=markerwidth,label=LABELS[i] if LABELS else "unknown")
        elif flag == 'L1':
            ax.plot(float(AIL1[i]),float(FLOPS[i]),c=colors[i%10],marker=styles[int(pu_index)],\
                    linestyle='None',ms=markersize,markerfacecolor='none',\
                    markeredgewidth=markerwidth,label=LABELS[i] if LABELS else "unknown")
        elif flag == 'L2':
            ax.plot(float(AIL2[i]),float(FLOPS[i]),c=colors[i%10],marker=styles[int(pu_index)],\
                    linestyle='None',ms=markersize,markerfacecolor='none',\
                    markeredgewidth=markerwidth,label=LABELS[i] if LABELS else "unknown")
        elif flag == 'HBM':
            ax.plot(float(AIHBM[i]),float(FLOPS[i]),c=colors[i%10],marker=styles[int(pu_index)],\
                    linestyle='None',ms=markersize,markerfacecolor='none',\
                    markeredgewidth=markerwidth,label=LABELS[i] if LABELS else "unknown")
        elif flag == 'all':
            ax.plot(float(AIL1[i]),float(FLOPS[i]),c=colors[i%10],marker=styles[int(pu_index)],\
                    linestyle='None',ms=markersize,markerfacecolor='none',\
                    markeredgewidth=markerwidth,label=LABELS[i] if LABELS else "unknown")
            ax.plot(float(AIL2[i]),float(FLOPS[i]),c=colors[i%10],marker=styles[int(pu_index)],\
                    linestyle='None',ms=markersize,markerfacecolor='none',\
                    markeredgewidth=markerwidth,label=LABELS[i] if LABELS else "unknown")
            ax.plot(float(AIHBM[i]),float(FLOPS[i]),c=colors[i%10],marker=styles[int(pu_index)],\
                    linestyle='None',ms=markersize,markerfacecolor='none',\
                    markeredgewidth=markerwidth,label=LABELS[i] if LABELS else "unknown")
            
    marker_handles = []  

    labels = ['FP16', 'FP32', 'FP64', 'TC', 'CC', 'TC/CC']
    for i in range(6):
        marker_handles.append(ax.plot([],[],c='k',marker=styles[i],linestyle='None',ms=markersize,\
                markerfacecolor='none',markeredgewidth=markerwidth,label=labels[i])[0])         


    for roof in cmpRoofs:
        ax.text(x[-ixx],roof[1]*1024,
              roof[0] + ': ' + '{0:.1f}'.format(roof[1]) + ' TFLOP/s',
              horizontalalignment='right',
              verticalalignment='bottom')

    for roof in memRoofs:
        ang = np.arctan(np.log10(xlim[1]/xlim[0]) / np.log10(ylim[1]/ylim[0])
                                   * fig.get_size_inches()[1]/fig.get_size_inches()[0] )
        if x[ixx]*roof[1] >ymin:
            ax.text(x[ixx],x[ixx]*roof[1]*(1+0.25*np.sin(ang)**2),
              roof[0] + ': ' + '{0:.1f}'.format(float(roof[1])) + ' GB/s',
              horizontalalignment='left',
              verticalalignment='bottom',
              rotation=180/np.pi*ang)
        else:
            ymin_ix_elbow=list()
            ymin_x_elbow=list()
            for ix in range(1,nx):
                if (ymin <= roof[1] * x[ix] and ymin > roof[1] * x[ix-1]):
                    ymin_x_elbow.append(x[ix-1])
                    ymin_ix_elbow.append(ix-1)
                    break
            ax.text(x[ixx+ymin_ix_elbow[0]],x[ixx+ymin_ix_elbow[0]]*roof[1]*(1+0.25*np.sin(ang)**2),
              roof[0] + ': ' + '{0:.1f}'.format(float(roof[1])) + ' GB/s',
              horizontalalignment='left',
              verticalalignment='bottom',
              rotation=180/np.pi*ang)


        
    leg1 = plt.legend(handles = marker_handles,loc='lower right', ncol=len(flag[0]) if 'all' not in flag else 3,bbox_to_anchor = (1,0))
    ax.add_artist(leg1)

    patch_handles = list()
    for i in range(0,len(AIHBM)):
        if FLOPS[i] > 0:
            patch_handles.append(mpatches.Patch(color=colors[i%10],label = LABELS[i] if LABELS else "unknown"))

    # leg2 = plt.legend(handles = patch_handles,loc=4,ncol=1,bbox_to_anchor = (1,0.1),scatterpoints = 1)

    # ax.text(xlim[0]*1.1,ylim[1]/1.1, '-'.join([filename,flag]), horizontalalignment='left',verticalalignment='top')
    # plt.title('-'.join([filename,flag]).replace('_',' '))

    vertical_offset = ylim[1] * 0.4
    kernel_count = len(UNIT) - 1
    tc_share = tc / kernel_count
    kernelinfo1 = "#Tensor Core Utilization: {} / {} ({:.2f})".format(tc, kernel_count, tc_share)
    kernelinfo2 = "#MemOp Kernels:" + " " + str(memops)
    ax.text(xlim[0]*1.1,ylim[1]/1.1, kernelinfo1, horizontalalignment='left',verticalalignment='top')
    ax.text(xlim[0]*1.1,ylim[1]/1.1 - vertical_offset, kernelinfo2, horizontalalignment='left',verticalalignment='top')

    plt.savefig('_'.join([filename,'PU',flag])+'.png')
#     plt.savefig('_'.join([filename,flag])+'.eps')

#    plt.show()
