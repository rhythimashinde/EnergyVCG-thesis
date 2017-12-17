from nego.mediated.Agents_Supervisor import NegoModel
from nego.bilateral.Agents_Supervisor import NegoModel
import matplotlib.pyplot as plt
import pandas as pd
import csv

#### LOG ####
N = 100
model = NegoModel(N)
timestep = 4
for i in range(timestep):
    m=model.perception()
    decisions = model.decision_fct()
    rewards = model.feedback()
    perceptions = model.perception()
    social_measurements = model.social_measurements(perceptions)
    costs = model.transactions_all()
    model.create_agents(m,decisions,rewards)
    model.step(decisions,rewards,perceptions,timestep)
    full_log = model.log_all()
    agents_total = full_log.shape[0]
    ratio = model.log(full_log)['ratio_seller'].sum()
    total = model.log(full_log).shape[0]
    if ratio != 0:
        model.evaluate(decisions,social_measurements,agents_total,ratio,total,rewards,costs,timestep)
    model.log(full_log).to_csv("out_log.csv",index=False)

#### PLOT ####
# TODO define better plots with clear differentiating results
# with open('out_log.csv', newline='') as csvfile:
#     reader = csv.reader(csvfile,delimiter=',', quotechar='|')
#     label = []
#     y = []
#     x = []
#     for row in reader:
#         if row[1] != 'id':
#             label_row = row[5]
#             production = row[2]
#             id = row[1]
#             y.append(production)
#             x.append(id)
#             label.append(label_row)
#     colors = ['red' if l == "buyer" else 'green' for l in label]
#     plt.scatter(x,y,color=colors)
#     plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
#     plt.show()

#### PLOT THE EVALUATION ####
# TODO change the evaluation here by including decisions
# s=NegoModel(5)
# s.threshold=3
# measures_poor = s.evaluate([1,0,0,0,0],0)
# measures_uniform = s.evaluate([1,1,1,0,0],0)
# measures_rich = s.evaluate([1,1,1,1,1],0)
#
# x = ["1_poor","2_uniform","3_rich"]
#
# y_gini = [measures_poor["gini"],measures_uniform["gini"],measures_rich["gini"]]
# plt.plot(x,y_gini,label="gini")
#
# y_success = [measures_poor["success"],measures_uniform["success"],measures_rich["success"]]
# plt.plot(x,y_success,label="success")
#
# y_efficiency = [measures_poor["efficiency"], measures_uniform["efficiency"], measures_rich["efficiency"]]
# plt.plot(x,y_efficiency,label="efficiency")
#
# plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
# plt.show()