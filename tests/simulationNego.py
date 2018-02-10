import functools
import itertools
import pandas as pd
import numpy as np
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
            log_tot=log_tot+model.log # concatenate lists
    # compute statistics for all tables in log file
    varnames=[k for k,v in conf["params"].items() if len(v)>1] # keep vars for which there is more than one value
    for varname in varnames:
        #stats_rew=get_stats(log_tot,"reward",idx=[varname])
        #stats_perc=get_stats(log_tot,"perception",idx=[varname],cols=["production","consumption","tariff"])
        #stats_decs=get_stats(log_tot,"decisions",idx=[varname],cols=["action","cost"])
        stats_eval=get_stats(log_tot,"evaluation",idx=[varname],cols=["social_welfare_high","social_welfare_low",
                                                                      "gini","efficiency","market_access_high",
                                                                      "market_access_low","wealth_distribution_high",
                                                                      "wealth_distribution_low"])
        #plot_trend(stats_rew,varname,"./rewards_"+str(test)+"_"+str(varname)+"_nego.png")
        #plot_trend(stats_perc,varname,"./perceptions_"+str(test)+"_"+str(varname)+"_nego.png")
        #plot_trend(stats_decs,varname,"./decisions_"+str(test)+"_"+str(varname)+"_nego.png")
        plot_measures(stats_eval,varname,"./eval_"+str(test)+"_"+str(varname)+"_nego.png")

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
              "bias_degree":(0 if i>len(population)*self.bias_mediator else 1),
              "cost":0,"timestep":timestep,"agentID":i}
             for i in range(len(population))]
        return ret


class MeasurementGenReal(NegoMeasurementGen):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mu1=kwargs["mu1"]
        self.s1=1
        self.mu2=kwargs["mu2"]
        self.s2=1
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
            ret=[{"consumption":abs(np.random.normal(loc=self.mu2,scale=self.s1)
                           if i>len(population)*self.caste else
                           np.random.normal(loc=self.mu1,scale=self.s2)),
                  "tariff":tariff_new,
                  "social_type":(2 if i>len(population)*self.caste else 1),
                  "production":(0 if i<len(population)*(1-self.caste)*(1-self.produce_high)
                                else(production if i<len(population)*(1-self.caste)
                                     else(0 if i<len(population)*((1-self.caste)+self.caste*(1-self.produce_low))
                                          else production))),
                  "biased":(0 if i<len(population)*(1-self.caste)*(1-self.biased_high)
                                else(1 if i<len(population)*(1-self.caste)
                                     else(0 if i<len(population)*((1-self.caste)+self.caste*(1-self.biased_low))
                                          else 1))),
                  "bias_degree":(0 if i>len(population)*self.bias_mediator else 1),
                  "main_cost":0.1,"cost":0,"timestep":timestep,"agentID":i,"type":None}
                 for i in range(len(population))]  # high class is 2, low class is 1, main_cost is maintenance cost
            return ret

if __name__ == '__main__':
    # tests={"uniform":{"N":10,"rep":10,"params":{"mu":[5,20,50]},"meas_fct":MeasurementGenNormal}
    # tests={"uniform":{"N":10,"rep":1,"params":{"mu":[2,5,8]},"meas_fct":MeasurementGenNormal}}
    tests={"binomial":{"T":5,"reps":50,"dec_fct":NegoDecisionLogic,"dec_fct_agent":NegoDecisionLogicAgent,
                       "rew_fct":NegoRewardLogic, "eval_fct":NegoEvaluationLogic,
                       "params":{"N":[20,50,100],"mu1":[1],"mu2":[5],"rich":[0.5],"bias_low":[0.2],
                                  "bias_high":[0.2,0.5,0.8],"low_caste":[0.36],"tariff_avg":[1],"produce_avg":[1],
                                  "buy_low":[0.25],"buy_high":[0.48],"bias_degree":[0.5]},
                       "meas_fct":MeasurementGenBinomial}}

    # for base, exp 1, 2, 3 implement snippet below only
    # tests={"real":{"T":23,"reps":50,"dec_fct":NegoDecisionLogic,"dec_fct_agent":NegoDecisionLogicAgent,
    #                    "rew_fct":NegoRewardLogic, "eval_fct":NegoEvaluationLogic,
    #                    "params":{"N":[20,50,100],"mu1":[1.01],"mu2":[1.37],"bias_low":[0.5],
    #                              "bias_high":[0.2,0.5,0.8],"low_caste":[0.36],"tariff_avg":[1],
    #                              "produce_avg":[1],"buy_low":[0.25],"buy_high":[0.48],"bias_degree":[0.5]},
    #                    "meas_fct":MeasurementGenReal}}

    # for exp 4, 5 implement snippet below only
    # tests={"real":{"T":23,"reps":50,"dec_fct":NegoDecisionLogic,"dec_fct_agent":NegoDecisionLogicAgent,
    #                    "rew_fct":NegoRewardLogic, "eval_fct":NegoEvaluationLogic,
    #                    "params":{"N":[20,50,100],"mu1":[1.01],"mu2":[1.37],"bias_low":[0.5],
    #                              "bias_high":[0.5],"low_caste":[0.36],"tariff_avg":[1],"produce_avg":[1],
    #                              "buy_low":[0.25],"buy_high":[0.48],"bias_degree":[0.2,0.5,0.8]},
    #                    "meas_fct":MeasurementGenReal}}

    # for sensitivity analysis with consumption upgrading
    # tests={"real":{"T":23,"reps":50,"dec_fct":NegoDecisionLogic,"dec_fct_agent":NegoDecisionLogicAgent,
    #                    "rew_fct":NegoRewardLogic, "eval_fct":NegoEvaluationLogic,
    #                    "params":{"N":[50],"mu1":[1.01,2.02,3.03],"mu2":[1.37],"bias_low":[0.5],
    #                              "bias_high":[0.5],"low_caste":[0.36],"tariff_avg":[1],"produce_avg":[1],
    #                              "buy_low":[0.25],"buy_high":[0.48],"bias_degree":[0.5]},
    #                    "meas_fct":MeasurementGenReal}}

    # for sensitivity analysis for tariff scaling
    # tests={"real":{"T":23,"reps":50,"dec_fct":NegoDecisionLogic,"dec_fct_agent":NegoDecisionLogicAgent,
    #                    "rew_fct":NegoRewardLogic, "eval_fct":NegoEvaluationLogic,
    #                    "params":{"N":[50],"mu1":[1.01],"mu2":[1.37],"bias_low":[0.5],
    #                              "bias_high":[0.5],"low_caste":[0.36],"tariff_avg":[1,2,3],"produce_avg":[1],
    #                              "buy_low":[0.25],"buy_high":[0.48],"bias_degree":[0.5]},
    #                    "meas_fct":MeasurementGenReal}}

    # for sensitivity analysis for production scaling
    # tests={"real":{"T":23,"reps":50,"dec_fct":NegoDecisionLogic,"dec_fct_agent":NegoDecisionLogicAgent,
    #                    "rew_fct":NegoRewardLogic, "eval_fct":NegoEvaluationLogic,
    #                    "params":{"N":[50],"mu1":[1.01],"mu2":[1.37],"bias_low":[0.5],
    #                              "bias_high":[0.5],"low_caste":[0.36],"tariff_avg":[1],"produce_avg":[1,2,3],
    #                              "buy_low":[0.25],"buy_high":[0.48],"bias_degree":[0.5]},
    #                    "meas_fct":MeasurementGenReal}}

    for test,conf in tests.items():
        run_experiment(test,conf)