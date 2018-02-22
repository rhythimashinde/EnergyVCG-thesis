import functools
import itertools
import pandas as pd
import numpy as np
from numpy.random import choice
from src.Supervisor import BaseSupervisor
from nego.src.Decisions import NegoDecisionLogicAgent
from nego.src.Decisions import NegoDecisionLogic
from nego.src.RewardLogic import NegoRewardLogic
from nego.src.MeasurementGen import NegoMeasurementGen
from nego.src.Evaluation import  NegoEvaluationLogic
from nego.src.utilsnego import *
from nego.src.Agent import *
from nego.src.Supervisor import *
import csv

def run_experiment(test,conf):
    log_tot=[]
    for r in range(conf["reps"]):
        for idx,p in expandgrid(conf["params"]).iterrows():
            params=p.to_dict()
            params.update({"repetition":r})
            f=functools.partial(conf["meas_fct"],**params)
            model=BaseSupervisor(N=int(params["N"]),measurement_fct=f,
                                 decision_fct=conf["dec_fct"],
                                 agent_decision_fct=conf["dec_fct_agent"],
                                 reward_fct=conf["rew_fct"],
                                 evaluation_fct=conf["eval_fct"],
                                 agent_type=NegoAgent)
            model.run(conf["T"],params=params)
            log_tot=log_tot+model.log
    varnames=[k for k,v in conf["params"].items() if len(v)>1] # keep vars for which there is more than one value
    for varname in varnames:
        # stats_rew=get_stats(log_tot,"reward",idx=[varname])
        # stats_perc=get_stats(log_tot,"perception",idx=[varname],cols=["production","consumption","tariff"])
        # stats_decs=get_stats(log_tot,"decisions",idx=[varname],cols=["action","cost"])
        stats_eval=get_stats(log_tot,"evaluation",idx=[varname],cols=["gini","efficiency",
                                                                      "wealth_distribution",
                                                                      "wealth_distribution_high",
                                                                      "wealth_distribution_low"])
        stats_eval1=get_stats(log_tot,"evaluation",idx=[varname],cols=["social_welfare_new","social_welfare_cost",
                                                                       "social_welfare_high_new",
                                                                       "social_welfare_low_new",
                                                                       "market_access","market_access_high",
                                                                       "market_access_low"])

        # nn = (pd.DataFrame([log_tot[i]["evaluation"]["efficiency"] for i in range(150)])[0]).tolist()
        # nn.append("efficiency")
        # rr = (pd.DataFrame([log_tot[i]["evaluation"]["gini"] for i in range(150)])[0]).tolist()
        # rr.append("gini")
        # q = (pd.DataFrame([log_tot[i]["evaluation"]["market_access"] for i in range(150)])[0]).tolist()
        # q.append("market_access")
        # r = (pd.DataFrame([log_tot[i]["evaluation"]["market_access_high"] for i in range(150)])[0]).tolist()
        # r.append("market_access_high")
        # p = (pd.DataFrame([log_tot[i]["evaluation"]["market_access_low"] for i in range(150)])[0]).tolist()
        # p.append("market_access_low")
        # m = (pd.DataFrame([log_tot[i]["evaluation"]["social_welfare"] for i in range(150)])[0]).tolist()
        # m.append("social_welfare")
        # n = (pd.DataFrame([log_tot[i]["evaluation"]["social_welfare_high"] for i in range(150)])[0]).tolist()
        # n.append("social_welfare_high")
        # o = (pd.DataFrame([log_tot[i]["evaluation"]["social_welfare_low"] for i in range(150)])[0]).tolist()
        # o.append("social_welfare_low")
        # tt = (pd.DataFrame([log_tot[i]["evaluation"]["wealth_distribution"] for i in range(150)])[0]).tolist()
        # tt.append("wealth_distribution")
        # pp = (pd.DataFrame([log_tot[i]["evaluation"]["wealth_distribution_high"] for i in range(150)])[0]).tolist()
        # pp.append("wealth_distribution_high")
        # qq = (pd.DataFrame([log_tot[i]["evaluation"]["wealth_distribution_low"] for i in range(150)])[0]).tolist()
        # qq.append("wealth_distribution_low")
        #
        # with open("log_"+str(varname)+".csv",'w') as myfile:
        #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        #     wr.writerow(nn)
        #     wr.writerow(rr)
        #     wr.writerow(q)
        #     wr.writerow(r)
        #     wr.writerow(p)
        #     wr.writerow(m)
        #     wr.writerow(n)
        #     wr.writerow(o)
        #     wr.writerow(tt)
        #     wr.writerow(pp)
        #     wr.writerow(qq)
        stats_all = pd.concat([stats_eval,stats_eval1],axis=1)
        stats_all.to_csv("evaluations_"+str(varname)+".csv")
        # plot_trend(stats_rew,varname,"./rewards_"+str(test)+"_"+str(varname)+"_nego.png")
        # plot_trend(stats_perc,varname,"./perceptions_"+str(test)+"_"+str(varname)+"_nego.png")
        # plot_trend(stats_decs,varname,"./decisions_"+str(test)+"_"+str(varname)+"_nego.png")
        plot_measures(stats_eval,varname,"./eval_"+str(test)+"_"+str(varname)+"_nego.png")
        plot_measures1(stats_eval1,varname,"./eval_1_"+str(test)+"_"+str(varname)+"_nego.png")

class RewardLogicFull(NegoRewardLogic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benefit=5
        self.damage=-10

    def get_rewards(self,decisions):
        """
        Almost full contribution is required
        """
        percs=np.sum([p["value"] for p in self.model.current_state["perception"]])
        thresh=np.random.uniform(percs*0.8,percs) # almost full contrib
        contribs=np.sum([d["contribution"] for d in decisions])
        outcome=success_nego(thresh,np.sum(contribs))
        if outcome==1:
            costs=np.array([d["cost"] for d in decisions])
            ret=-costs+self.benefit
            ret=[{"reward":r} for r in ret]
        else:
            ret=[{"reward":self.damage}]*self.model.N
        return ret

class RewardLogicUniform(NegoRewardLogic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benefit=5
        self.damage=-10

    def get_rewards(self,decisions):
        """
        The threshold is randomly generated around the average contribution
        """
        percs=[p["production"] for p in self.model.current_state["perception"]]
        thresh=np.random.normal(loc=np.mean(percs),scale=1)
        thresh=max(1,thresh)
        contribs=np.sum([d["contribution"] for d in decisions])
        outcome=success_nego(thresh,np.sum(contribs))
        if outcome==1:
            costs=np.array([d["cost"] for d in decisions])
            ret=-costs+self.benefit
            ret=[{"reward":r} for r in ret]
        else:
            ret=[{"reward":self.damage}]*self.model.N
        return ret

class DecisionLogicEmpty(NegoDecisionLogic):
    """
    Returns a constant decision
    """
    def get_decision(self,perceptions):
        pass

    def get_feedback(self,perceptions,reward):
        pass

class DecisionLogicSupervisorMandatory(NegoDecisionLogic):
    """
    Returns a constant decision
    """
    def get_decision(self,perceptions):
        self.last_actions=[{"contribution":a["value"],"cost":a["cost"],"agentID":a["agentID"],
                            "contributed":True,"timestep":a["timestep"]} for a in perceptions]
        return self.last_actions

class DecisionLogicSupervisorProbabilistic(NegoDecisionLogic):
    """
    Returns a constant decision
    """
    def get_decision(self,perceptions):
        self.last_actions=[{"contribution":a["value"],"cost":a["cost"],"agentID":a["agentID"],
                            "contributed":(True if np.random.uniform()<=0.5 else False),"timestep":a["timestep"]}
                           for a in perceptions]
        return self.last_actions

class MeasurementGenUniform(NegoMeasurementGen):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.n1=kwargs["n1"]
        self.n2=kwargs["n2"]

    def get_measurements(self,population,timestep):
        """
        Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
        """
        ret=[{"value":np.random.uniform(self.n1,self.n2),"cost":0,"timestep":timestep,"agentID":i}
             for i in range(len(population))]
        return ret

class MeasurementGenNormal(NegoMeasurementGen):
    def __init__(self,*args, **kwargs):
        super().__init__()
        self.mu=kwargs["mu"]
        self.s=3

    def get_measurements(self,population,timestep):
        """
        Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
        """
        ret=[{"production":np.random.normal(loc=self.mu,scale=self.s),
              "consumption":np.random.normal(loc=self.mu,scale=self.s),
              "timestep":timestep,"agentID":i,"tariff":np.random.uniform(low=0,high=5)}
             for i in range(len(population))]
        return ret

class MeasurementGenBinomial(NegoMeasurementGen):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mu1=kwargs["mu1"]
        self.s1=1
        self.mu2=kwargs["mu2"]
        self.s2=1
        self.sep=kwargs["rich"]
        self.produce_low = kwargs["buy_low"] # proportion of agents who can produce in lower caste
        self.produce_high = kwargs["buy_high"] # proportion of agents who can produce in higher caste
        self.caste=kwargs["low_caste"] # proportion of agents in low caste
        self.biased_low=kwargs["bias_low"]  # proportion of biased agents among low caste
        self.biased_high = kwargs["bias_high"] # proportion of biased agents among low caste
        self.bias_mediator = kwargs["bias_degree"] # proportion of agents being biased by the mediator

    def get_measurements(self,population,timestep):
        """
        Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
        """
        ret=[{"production":(np.random.normal(loc=self.mu1,scale=self.s1)
                       if i>len(population)*self.caste else
                       np.random.normal(loc=self.mu2,scale=self.s2)),
              "consumption":(np.random.normal(loc=self.mu1,scale=self.s1)
                       if i>len(population)*self.caste else
                       np.random.normal(loc=self.mu2,scale=self.s2)),
              "tariff":np.random.uniform(1,5),"main_cost":0.1,
              "social_type":(2 if i>len(population)*self.caste else 1),
              "biased":(0 if i<len(population)*(1-self.caste)*(1-self.biased_high)
                                else(1 if i<len(population)*(1-self.caste)
                                     else(0 if i<len(population)*((1-self.caste)+self.caste*(1-self.biased_low))
                                          else 1))),
              "bias_degree":(choice((True,False),1,p=(self.bias_mediator,(1-self.bias_mediator))))[0],
              "cost":0,"timestep":timestep,"agentID":i}
             for i in range(len(population))]
        return ret


class MeasurementGenReal(NegoMeasurementGen):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mu1=kwargs["mu1"]
        self.s1=1
        self.mu2=kwargs["mu2"]
        self.s2=0.5
        self.produce_low = kwargs["buy_low"] # proportion of agents who can produce in lower caste
        self.produce_high = kwargs["buy_high"] # proportion of agents who can produce in higher caste
        self.caste=kwargs["low_caste"] # proportion of agents in low caste
        self.biased_low=kwargs["bias_low"]  # proportion of biased agents among low caste
        self.biased_high = kwargs["bias_high"] # proportion of biased agents among low caste
        self.bias_mediator = kwargs["bias_degree"] # proportion of agents being biased by the mediator
        self.tariff_avg = kwargs["tariff_avg"]
        self.produce_avg = kwargs["produce_avg"]

    def get_measurements(self,population,timestep):
        """
        Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
        """
        with open('tariff.csv') as csvfile:
            has_header = csv.Sniffer().sniff(csvfile.readline())
            csvfile.seek(0)
            readCSV = csv.DictReader(csvfile)
            if has_header:
                next(readCSV)
            data = [row for row in readCSV]
            tariff = data[timestep]["inrpriceperkwh"+str(int(self.tariff_avg))]
            tariff_new = abs(np.random.normal(loc=float(tariff),scale=self.s2))
            production = self.produce_avg*np.random.uniform(20000,100000)*8/24/20000
            ret=[{"consumption":int(abs(np.random.normal(loc=self.mu2,scale=self.s1)))
                           if i>len(population)*self.caste else
                           int(abs(np.random.normal(loc=self.mu1,scale=self.s1))),
                  "tariff":tariff_new,
                  "social_type":(2 if i>len(population)*self.caste else 1),
                  "old_production":0,"old_consumption":0,
                  "production":(0 if i<len(population)*(1-self.caste)*(1-self.produce_high)
                                else(int(abs(np.random.normal(production,self.s2))) if i<len(population)*(1-self.caste)
                                     else(0 if i<len(population)*((1-self.caste)+self.caste*(1-self.produce_low))
                                          else abs(np.random.normal(production,self.s2))))),
                  "biased":(0 if i<len(population)*(1-self.caste)*(1-self.biased_high)
                                else(1 if i<len(population)*(1-self.caste)
                                     else(0 if i<len(population)*((1-self.caste)+self.caste*(1-self.biased_low))
                                          else 1))),
                  "bias_degree":(0 if i>len(population)*self.bias_mediator else 1),"agentID":0,
                  "main_cost":0.1,"cost":0,"timestep":timestep,"type":None}
                 for i in range(len(population))]  # high class is 2, low class is 1, main_cost is maintenance cost
            return ret

if __name__ == '__main__':

    tests_N={"real":{"T":23,"reps":50,"dec_fct":NegoDecisionLogic,"dec_fct_agent":NegoDecisionLogicAgent,
                       "rew_fct":NegoRewardLogic, "eval_fct":NegoEvaluationLogic,
                       "params":{"N":[20,50,100],"mu1":[1.01],"mu2":[1.37],"bias_low":[0.02],
                                 "bias_high":[0.8],"low_caste":[0.36],"tariff_avg":[1],
                                 "produce_avg":[1],"buy_low":[0.25],"buy_high":[0.48],"bias_degree":[0.5]},
                       "meas_fct":MeasurementGenReal}}

    tests_low_caste={"real":{"T":23,"reps":50,"dec_fct":NegoDecisionLogic,"dec_fct_agent":NegoDecisionLogicAgent,
                       "rew_fct":NegoRewardLogic, "eval_fct":NegoEvaluationLogic,
                       "params":{"N":[50],"mu1":[1.01],"mu2":[1.37],"bias_low":[0.02],
                                 "bias_high":[0.8],"low_caste":[0.2,0.36,0.8],"tariff_avg":[1],
                                 "produce_avg":[1],"buy_low":[0.25],"buy_high":[0.48],"bias_degree":[0.5]},
                       "meas_fct":MeasurementGenReal}}

    tests_buy_low={"real":{"T":23,"reps":50,"dec_fct":NegoDecisionLogic,"dec_fct_agent":NegoDecisionLogicAgent,
                       "rew_fct":NegoRewardLogic, "eval_fct":NegoEvaluationLogic,
                       "params":{"N":[50],"mu1":[1.01],"mu2":[1.37],"bias_low":[0.02],
                                 "bias_high":[0.8],"low_caste":[0.36],"tariff_avg":[1],
                                 "produce_avg":[1],"buy_low":[0.25,0.5,0.8],"buy_high":[0.48],
                                 "bias_degree":[0.5]},"meas_fct":MeasurementGenReal}}

    tests_bias_high={"real":{"T":23,"reps":50,"dec_fct":NegoDecisionLogic,"dec_fct_agent":NegoDecisionLogicAgent,
                       "rew_fct":NegoRewardLogic, "eval_fct":NegoEvaluationLogic,
                       "params":{"N":[50],"mu1":[1.01],"mu2":[1.37],"bias_low":[0.02],
                                 "bias_high":[0.2,0.5,0.8],"low_caste":[0.36],"tariff_avg":[1],
                                 "produce_avg":[1],"buy_low":[0.25],"buy_high":[0.48],
                                 "bias_degree":[0.5]},"meas_fct":MeasurementGenReal}}

    tests_bias_degree={"real":{"T":23,"reps":50,"dec_fct":NegoDecisionLogic,"dec_fct_agent":NegoDecisionLogicAgent,
                       "rew_fct":NegoRewardLogic, "eval_fct":NegoEvaluationLogic,
                       "params":{"N":[50],"mu1":[1.01],"mu2":[1.37],"bias_low":[0.02],
                                 "bias_high":[0.5],"low_caste":[0.36],"tariff_avg":[1],
                                 "produce_avg":[1],"buy_low":[0.25],"buy_high":[0.48],
                                 "bias_degree":[0.2,0.5,0.8]},"meas_fct":MeasurementGenReal}}

    tests_consumption={"real":{"T":23,"reps":50,"dec_fct":NegoDecisionLogic,"dec_fct_agent":NegoDecisionLogicAgent,
                       "rew_fct":NegoRewardLogic, "eval_fct":NegoEvaluationLogic,
                       "params":{"N":[50],"mu1":[1.01,2.02,3.03],"mu2":[1.37],"bias_low":[0.5],
                                 "bias_high":[0.5],"low_caste":[0.36],"tariff_avg":[1],"produce_avg":[1],
                                 "buy_low":[0.25],"buy_high":[0.48],"bias_degree":[0.5]},
                       "meas_fct":MeasurementGenReal}}

    tests_tariff={"real":{"T":23,"reps":50,"dec_fct":NegoDecisionLogic,"dec_fct_agent":NegoDecisionLogicAgent,
                       "rew_fct":NegoRewardLogic, "eval_fct":NegoEvaluationLogic,
                       "params":{"N":[50],"mu1":[1.01],"mu2":[1.37],"bias_low":[0.5],
                                 "bias_high":[0.5],"low_caste":[0.36],"tariff_avg":[1,2,3],"produce_avg":[1],
                                 "buy_low":[0.25],"buy_high":[0.48],"bias_degree":[0.5]},
                       "meas_fct":MeasurementGenReal}}

    tests_production={"real":{"T":23,"reps":50,"dec_fct":NegoDecisionLogic,"dec_fct_agent":NegoDecisionLogicAgent,
                       "rew_fct":NegoRewardLogic, "eval_fct":NegoEvaluationLogic,
                       "params":{"N":[50],"mu1":[1.01],"mu2":[1.37],"bias_low":[0.5],
                                 "bias_high":[0.5],"low_caste":[0.36],"tariff_avg":[1],"produce_avg":[1,2,3],
                                 "buy_low":[0.25],"buy_high":[0.48],"bias_degree":[0.5]},
                       "meas_fct":MeasurementGenReal}}

    for test,conf in tests_N.items():
        run_experiment(test,conf)

    for test,conf in tests_bias_degree.items():
        run_experiment(test,conf)

    for test,conf in tests_bias_high.items():
        run_experiment(test,conf)

    for test,conf in tests_buy_low.items():
        run_experiment(test,conf)

    for test,conf in tests_consumption.items():
        run_experiment(test,conf)

    for test,conf in tests_low_caste.items():
        run_experiment(test,conf)

    for test,conf in tests_production.items():
        run_experiment(test,conf)

    for test,conf in tests_tariff.items():
        run_experiment(test,conf)

    # tests0={"real":{"T":10,"reps":50,"dec_fct":NegoDecisionLogic,"dec_fct_agent":NegoDecisionLogicAgent,
    #                    "rew_fct":NegoRewardLogic, "eval_fct":NegoEvaluationLogic,
    #                    "params":{"N":[5,10],"mu1":[1.01],"mu2":[1.37],"bias_low":[0.02],
    #                              "bias_high":[0.8],"low_caste":[0.36],"tariff_avg":[1],
    #                              "produce_avg":[1],"buy_low":[0.25],"buy_high":[0.48],"bias_degree":[0.5]},
    #                    "meas_fct":MeasurementGenReal}}
    #
    # for test,conf in tests0.items():
    #     run_experiment(test,conf)