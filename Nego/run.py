from Agents_Supervisor import NegoModel
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

plt.show()
