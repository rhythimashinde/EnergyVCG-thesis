from Nego.bilateral.Agents_Supervisor import NegoModel
import matplotlib.pyplot as plt

model = NegoModel(100)
m=model.perception()
decisions = model.chose_action()
agents=model.init_agents(m,decisions)
measures = [[a.production,a.consumption,a.tariff,a.energy,a.type,a.partner,a.reward] for a in agents]
print(measures)

x = [a.unique_id for a in model.schedule.agents]
y = [a.energy for a in model.schedule.agents]
label = [a.type for a in model.schedule.agents]
colors = ['red' if l == "buyer" else 'green' for l in label]
plt.scatter(x,y,color=colors)
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)

plt.show()

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

