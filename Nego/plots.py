from Agents_Supervisor_bilateral import NegoModel
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

y_poor = round(measures_poor["gini"],2)
y_uniform = round(measures_uniform["gini"],2)
y_rich = round(measures_rich["gini"],2)
y_gini = [y_poor,y_uniform,y_rich]
plt.plot(x,y_gini,label="gini")

y_poor_success = round(measures_poor["success"],2)
y_uniform_success = round(measures_uniform["success"],2)
y_rich_success = round(measures_rich["success"],2)
y_success = [y_poor_success, y_uniform_success,y_rich_success]
plt.plot(x,y_success,label="success")

y_poor_efficiency = measures_poor["efficiency"]
y_uniform_efficiency = measures_uniform["efficiency"]
y_rich_efficiency = measures_rich["efficiency"]
y_efficiency = [y_poor_efficiency, y_uniform_efficiency,y_rich_efficiency]
plt.plot(x,y_efficiency,label="efficiency")

plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.show()

