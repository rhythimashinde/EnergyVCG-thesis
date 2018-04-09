import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from matplotlib import rcParams
# rcParams.update({'figure.autolayout': True})

def plot_trend(df,xname,filename,trends=None,yname=None,
               lstyles=['-','--',':','-.'],colors=None,font_size=4,ylab=None,xlab=None):
    if trends is None:
        trends=[d[:-5] for d in df.columns if ("_mean" in d)]
    fig,ax=plt.subplots()
    if len(trends)>len(lstyles):
        lstyles=['-']*len(trends)
    else:
        lstyles=lstyles[:len(trends)]
    lineArtists=[plt.Line2D((0,1),(0,0), color='k', marker='', linestyle=sty) for sty in lstyles]
    ax.set_xlabel(xlab or xname)
    ax.set_ylabel(ylab or yname)
    if yname is None:
        data=[(df,None)]
    else:
        data=[(df[df[yname]==i],i) for i in df[yname].unique()]
    if colors is None:
        cmap = plt.get_cmap('cubehelix_r')
        colors=[cmap(float(i+1)/(len(data)+1)) for i in range(len(data))]
    colorArtists = [plt.Line2D((0,1),(0,0), color=c) for c in colors]
    #fig.suptitle(title)
    # if ylim:
    #     ax.set_ylim(ylim)
    box = ax.get_position()
    # if yname and len(yname)>1 and len(trends)>1:
    #     # Shrink current axis's height by 10% on the bottom
    #     ax.set_position([box.x0, box.y0 + box.height * 0.2,
    #                   box.width, box.height * 0.8])
    if yname and len(df[yname].unique())>1:
        ax.add_artist(plt.legend(colorArtists,[l for d,l in data],
                                 loc='upper center', bbox_to_anchor=(0.5, 1.11),fancybox=True,
                                 shadow=True, ncol=len(data),fontsize=font_size))
    if len(trends)>1:
        ax.add_artist(plt.legend(lineArtists,trends,loc='upper center',
                                 bbox_to_anchor=(0.5, 1),fancybox=True, shadow=True,
                                 ncol=len(trends),fontsize=font_size))
    for (d,l),c in zip(data,colors):
        x=d[xname]
        for y,sty in zip(trends,lstyles):
            lab=(y if l is None else y+"; "+yname+"="+str(l))
            ax.plot(x,d[y+"_mean"],label=lab,linestyle=sty,color=c)
            ax.fill_between(x,np.asarray(d[y+"_mean"])-np.asarray(d[y+"_ci"]),
                            np.asarray(d[y+"_mean"])+np.asarray(d[y+"_ci"]),
                            alpha=0.2,linestyle=sty,facecolor=c)
    # plt.legend()
    fig.savefig(filename,format='png')
    plt.close(fig)

datadir="C:/Users/Rhytima/Dropbox/thesis_shinde/results/new_gl/discrimination_2"
plotdir="./results_rhy"
if not os.path.exists(plotdir):
    os.makedirs(plotdir)
treatments=[["base","B."],
            #["base_dis","B. Disc."],
            # ["exp3","M. Disc."],
            #["exp4","M. Disc. Bidsplit"],
            ["exp1","M."]
            ,["exp2","M. Bidsplit"]]
varnames=[["N","number of agents"]]#,["low_caste","proportion of agents in lower caste"]]
measures=[#["efficiency","Efficiency"],["gini","Inequality, number of sellers and buyers"],
          ["social_welfare","social welfare"],
          ["social_welfare_low","social welfare, low caste"],
          ["social_welfare_high","social welfare, high caste"],
          ["market_access","Market access"],
          ["market_access_low","Market access, low caste"],
          #["wealth_distribution","Inequality of profits"],
          #["wealth_distribution_low","Inequality of profits, low caste"],
          ["market_access_high","Market access, high caste"]]
          #,["wealth_distribution_high","Inequality of profits, high caste"]]
for v,vl in varnames:
    ret=[]
    for d,l in treatments:
        folder=os.path.join(datadir,d)
        tmp=pd.read_csv(os.path.join(folder,"evaluations_"+v+".csv"))
        tmp.drop("Unnamed: 0",axis=1,inplace=True)
        tmp["treatment"]=l
        ret.append(tmp)
    ret=pd.concat(ret)
    for m,l in measures:
        plot_trend(ret,v,os.path.join(plotdir,str(v)+"_dis_bid_base_"+str(m)+".png"),
                   yname="treatment",trends=[m],font_size=12,ylab=l,xlab=vl)
