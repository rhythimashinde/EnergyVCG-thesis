from Nego.bilateral.Agents_Supervisor import NegoModel
import matplotlib.pyplot as plt
import pandas as pd
import csv

#### LOG ####
N = 100
model = NegoModel(N)
m=model.perception()
decisions = model.decision_fct()
rewards = model.feedback()
agents=model.init_agents(m,decisions,rewards)
partner = [a.partner for a in agents]
partner_id = [a.unique_id for a in partner]
d = [[a.unique_id,a.production,a.consumption,a.tariff,a.type,a.reward,a.state,a.action] for a in agents]
agents_dataframe = pd.DataFrame(data=d,columns=['id','production','consumption','tariff','type','reward','state','action'])
agents_dataframe_new = agents_dataframe.assign(partner_id = partner_id)
#agents_dataframe_new.to_csv('out_log.csv',sep=",")

#### PLOT ####
with open('out_log.csv', newline='') as csvfile:
    reader = csv.reader(csvfile,delimiter=',', quotechar='|')
    label = []
    y = []
    x = []
    for row in reader:
        if row[1] != 'id':
            label_row = row[5]
            production = row[2]
            id = row[1]
            y.append(production)
            x.append(id)
            label.append(label_row)
    colors = ['red' if l == "buyer" else 'green' for l in label]
    plt.scatter(x,y,color=colors)
    plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
    plt.show()

#### TEST THE EVALUATION ####
s=NegoModel(5)
s.threshold=3
measures_poor = s.evaluate([1,0,0,0,0],0)
measures_uniform = s.evaluate([1,1,1,0,0],0)
measures_rich = s.evaluate([1,1,1,1,1],0)

x = ["1_poor","2_uniform","3_rich"]

y_gini = [measures_poor["gini"],measures_uniform["gini"],measures_rich["gini"]]
plt.plot(x,y_gini,label="gini")

y_success = [measures_poor["success"],measures_uniform["success"],measures_rich["success"]]
plt.plot(x,y_success,label="success")

y_efficiency = [measures_poor["efficiency"], measures_uniform["efficiency"], measures_rich["efficiency"]]
plt.plot(x,y_efficiency,label="efficiency")

plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
plt.show()