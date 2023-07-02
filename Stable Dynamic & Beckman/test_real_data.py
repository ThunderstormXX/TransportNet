import pandas as pd
import numpy as np
import data_handler as dh
import model as md
import time
import matplotlib.pyplot as plt
from matplotlib import rc
from platform import python_version
import graph_tool
import pickle





# ТУТ ДОЛЖНЫ БЫТЬ ДАННЫЕ ПОЛУЧЕННЫЕ ИЗ OpenstreetMap
beckmann_save = 'beckmann_results/'
cities_data = 'cities_data/'
net_name = cities_data + 'Anaheim_net.tntp'
trips_name = cities_data + 'Anaheim_trips.tntp'
handler = dh.DataHandler()
graph_data = handler.GetGraphData(net_name, columns = ['init_node', 'term_node', 'capacity', 'free_flow_time'])
graph_correspondences, total_od_flow = handler.GetGraphCorrespondences(trips_name)
model = md.Model(graph_data, graph_correspondences, 
                    total_od_flow, mu = 0.25, rho = 0.15)



# print(graph_data['graph_table'].head())
print(type(graph_correspondences), graph_correspondences)
print(graph_data)
print(type(total_od_flow),total_od_flow)


max_iter = 10

# # Conjugate FWM test

# assert(model.mu == 0.25)

# print('Conjugate Frank-Wolfe without stopping criteria')
# solver_kwargs = {'max_iter' : max_iter, 'stop_crit': 'max_iter',
#                  'verbose' : True, 'verbose_step': 2000, 'save_history' : True , "alpha_default" : 0.6}
# tic = time.time()
# result_cfwm = model.find_equilibrium(solver_name = 'cfwm', solver_kwargs = solver_kwargs)
# toc = time.time()
# print('Elapsed time: {:.0f} sec'.format(toc - tic))
# print('Time ratio =', np.max(result_cfwm['times'] / graph_data['graph_table']['free_flow_time']))
# print('Flow excess =', np.max(result_cfwm['flows'] / graph_data['graph_table']['capacity']) - 1, end = '\n\n')



# # FWM test
# assert(model.mu == 0.25)

# print('Frank-Wolfe without stopping criteria')
# solver_kwargs = {'max_iter' : max_iter, 'stop_crit': 'max_iter',
#                  'verbose' : True, 'verbose_step': 2000, 'save_history' : True}
# tic = time.time()
# result_fwm = model.find_equilibrium(solver_name = 'fwm', solver_kwargs = solver_kwargs)
# toc = time.time()
# print('Elapsed time: {:.0f} sec'.format(toc - tic))
# print('Time ratio =', np.max(result_fwm['times'] / graph_data['graph_table']['free_flow_time']))
# print('Flow excess =', np.max(result_fwm['flows'] / graph_data['graph_table']['capacity']) - 1, end = '\n\n')


#RESULT 

# dual_gaps_fwm = result_fwm['duality_gaps']
# dual_gaps_cfwm = result_cfwm['duality_gaps']
# iters = np.arange(len(dual_gaps_cfwm))
# plt.figure(figsize = (10, 5))

# plt.plot(iters, dual_gaps_fwm , color ='red' ,label = 'Frank Wolf')
# plt.plot(iters, dual_gaps_cfwm , color = 'green' ,label = 'Conjugate Frank Wolf')
# plt.ylabel('duality gap', fontsize = 12)
# plt.xlabel('iterations', fontsize = 12)
# plt.yscale('log')
# plt.show()

