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




# # EXAMPLE DATA (SIMPLY)
# beckmann_save = 'beckmann_results/'
# cities_data = 'cities_data/'
# net_name = cities_data + 'Example_net.tntp'
# trips_name = cities_data + 'Example_trips.tntp'
# handler = dh.DataHandler()
# graph_data = handler.GetGraphData(net_name, columns = ['init_node', 'term_node', 'capacity', 'free_flow_time'])
# graph_correspondences, total_od_flow = handler.GetGraphCorrespondences(trips_name)
# model = md.Model(graph_data, graph_correspondences, 
#                     total_od_flow, mu = 0.25, rho = 0.15)
# print(graph_data['graph_table'].head())

# #EXAMPLE DATA (REAL DATA) (Chicago))
# beckmann_save = 'beckmann_results/'
# cities_data = 'cities_data/'
# net_name = cities_data + 'ChicagoSketch_net.tntp'
# trips_name = cities_data + 'ChicagoSketch_trips.tntp'
# handler = dh.DataHandler()
# graph_data = handler.GetGraphData(net_name, columns = ['init_node', 'term_node', 'capacity', 'free_flow_time'])
# graph_correspondences, total_od_flow = handler.GetGraphCorrespondences(trips_name)
# model = md.Model(graph_data, graph_correspondences, 
#                     total_od_flow, mu = 0.25, rho = 0.15)
# graph_data['graph_table'].head()

#EXAMPLE DATA (REAL DATA) (ANAHEIM)
beckmann_save = 'beckmann_results/'
cities_data = 'cities_data/'
net_name = cities_data + 'Anaheim_net.tntp'
trips_name = cities_data + 'Anaheim_trips.tntp'
handler = dh.DataHandler()
graph_data = handler.GetGraphData(net_name, columns = ['init_node', 'term_node', 'capacity', 'free_flow_time'])
graph_correspondences, total_od_flow = handler.GetGraphCorrespondences(trips_name)
model = md.Model(graph_data, graph_correspondences, 
                    total_od_flow, mu = 0.25, rho = 0.15)
graph_data['graph_table'].head()


#  Подбор alpha

alpha_range = np.arange(0,1.0,0.01)

max_iter = 25
duality_by_alpha = []
for alpha in alpha_range :
    print("alpha :",alpha)
    assert(model.mu == 0.25)
    # print('Conjugate Frank-Wolfe without stopping criteria')
    solver_kwargs = {'max_iter' : max_iter, 'stop_crit': 'max_iter',
                    'verbose' : True, 'verbose_step': 2000, 'save_history' : True , 'alpha_default' : alpha}
    tic = time.time()
    result_cfwm = model.find_equilibrium(solver_name = 'cfwm', solver_kwargs = solver_kwargs)
    toc = time.time()
    # print('Elapsed time: {:.0f} sec'.format(toc - tic))
    # print('Time ratio =', np.max(result_cfwm['times'] / graph_data['graph_table']['free_flow_time']))
    # print('Flow excess =', np.max(result_cfwm['flows'] / graph_data['graph_table']['capacity']) - 1, end = '\n\n')
    duality_by_alpha.append(result_cfwm['duality_gaps'][-1])
# print(duality_by_alpha , alpha_range)



# #RESULT 

plt.figure(figsize = (10, 5))

plt.plot(alpha_range, duality_by_alpha , color ='red' ,label = '')

plt.ylabel('50 iters dual_gaps', fontsize = 12)
plt.xlabel('alpha', fontsize = 12)
plt.yscale('log')
plt.show()

