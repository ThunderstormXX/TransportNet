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




max_iter = 1000

# Conjugate FWM test

assert(model.mu == 0.25)

print('Conjugate Frank-Wolfe without stopping criteria')
solver_kwargs = {'max_iter' : max_iter, 'stop_crit': 'max_iter',
                 'verbose' : True, 'verbose_step': 2000, 'save_history' : True , "alpha_default" : 0.6}
tic = time.time()
result_cfwm = model.find_equilibrium(solver_name = 'cfwm', solver_kwargs = solver_kwargs)
toc = time.time()
print('Elapsed time: {:.0f} sec'.format(toc - tic))
print('Time ratio =', np.max(result_cfwm['times'] / graph_data['graph_table']['free_flow_time']))
print('Flow excess =', np.max(result_cfwm['flows'] / graph_data['graph_table']['capacity']) - 1, end = '\n\n')

# Conjugate FWM (With linesearch) test

assert(model.mu == 0.25)

print('Conjugate Frank-Wolfe without stopping criteria')
solver_kwargs = {'max_iter' : max_iter, 'stop_crit': 'max_iter',
                 'verbose' : True, 'verbose_step': 2000, 'save_history' : True , "alpha_default" : 0.6 , "linesearch" :True}
tic = time.time()
result_cfwm_linesearch = model.find_equilibrium(solver_name = 'cfwm', solver_kwargs = solver_kwargs)
toc = time.time()
print('Elapsed time: {:.0f} sec'.format(toc - tic))
print('Time ratio =', np.max(result_cfwm_linesearch['times'] / graph_data['graph_table']['free_flow_time']))
print('Flow excess =', np.max(result_cfwm_linesearch['flows'] / graph_data['graph_table']['capacity']) - 1, end = '\n\n')


# FWM test
assert(model.mu == 0.25)

print('Frank-Wolfe without stopping criteria')
solver_kwargs = {'max_iter' : max_iter, 'stop_crit': 'max_iter',
                 'verbose' : True, 'verbose_step': 2000, 'save_history' : True}
tic = time.time()
result_fwm = model.find_equilibrium(solver_name = 'fwm', solver_kwargs = solver_kwargs)
toc = time.time()
print('Elapsed time: {:.0f} sec'.format(toc - tic))
print('Time ratio =', np.max(result_fwm['times'] / graph_data['graph_table']['free_flow_time']))
print('Flow excess =', np.max(result_fwm['flows'] / graph_data['graph_table']['capacity']) - 1, end = '\n\n')

# FWM (With linesearch) test
assert(model.mu == 0.25)

print('Frank-Wolfe without stopping criteria')
solver_kwargs = {'max_iter' : max_iter, 'stop_crit': 'max_iter',
                 'verbose' : True, 'verbose_step': 2000, 'save_history' : True , "linesearch" : True}
tic = time.time()
result_fwm_linesearch = model.find_equilibrium(solver_name = 'fwm', solver_kwargs = solver_kwargs)
toc = time.time()
print('Elapsed time: {:.0f} sec'.format(toc - tic))
print('Time ratio =', np.max(result_fwm_linesearch['times'] / graph_data['graph_table']['free_flow_time']))
print('Flow excess =', np.max(result_fwm_linesearch['flows'] / graph_data['graph_table']['capacity']) - 1, end = '\n\n')


#USTM test

# eps_abs = 31
# print('eps_abs =', eps_abs)
# print(eps_abs)
# solver_kwargs = {'eps_abs': eps_abs,
#                     'max_iter': max_iter, 'stop_crit': 'dual_gap',
#                     'verbose' : True, 'verbose_step': 2000, 'save_history' : True}
# tic = time.time()
# result_ustm = model.find_equilibrium(solver_name = 'ustm', composite = True, solver_kwargs = solver_kwargs)
# toc = time.time()
# print('Elapsed time: {:.0f} sec'.format(toc - tic))
# print('Time ratio =', np.max(result_ustm['times'] / graph_data['graph_table']['free_flow_time']))
# print('Flow excess =', np.max(result_ustm['flows'] / graph_data['graph_table']['capacity']) - 1, end = '\n\n')

# result_ustm['eps_abs'] = eps_abs
# result_ustm['elapsed_time'] = toc - tic

#UGD test 

# assert(model.mu == 0.25)
# print('eps_abs =', eps_abs)
# solver_kwargs = {'eps_abs': eps_abs,
#                     'max_iter': max_iter, 'stop_crit': 'dual_gap',
#                     'verbose' : True, 'verbose_step': 4000, 'save_history' : True}
# tic = time.time()
# result_ugd = model.find_equilibrium(solver_name = 'ugd', composite = True, solver_kwargs = solver_kwargs)
# toc = time.time()
# print('Elapsed time: {:.0f} sec'.format(toc - tic))
# print('Time ratio =', np.max(result_ugd['times'] / graph_data['graph_table']['free_flow_time']))
# print('Flow excess =', np.max(result_ugd['flows'] / graph_data['graph_table']['capacity']) - 1, end = '\n\n')

# result_ugd['eps_abs'] = eps_abs
# result_ugd['elapsed_time'] = toc - tic


# WDA test

# solver_kwargs = {'max_iter' : max_iter, 'stop_crit': 'max_iter',
#                  'verbose': True, 'verbose_step': 4000, 'save_history': True}
# tic = time.time()
# result_wda = model.find_equilibrium(solver_name = 'wda', composite = True, solver_kwargs = solver_kwargs)
# toc = time.time()
# print('Elapsed time: {:.0f} sec'.format(toc - tic))
# print('Time ratio =', np.max(result_wda['times'] / graph_data['graph_table']['free_flow_time']))
# print('Flow excess =', np.max(result_wda['flows'] / graph_data['graph_table']['capacity']) - 1, end = '\n\n')

# result_wda['elapsed_time'] = toc - tic

# WDA-noncomposite test

# solver_kwargs = {'max_iter' : max_iter, 'stop_crit': 'max_iter',
#                  'verbose': True, 'verbose_step': 4000, 'save_history' : True}
# tic = time.time()
# result_wdaNon = model.find_equilibrium(solver_name = 'wda', composite = False, solver_kwargs = solver_kwargs)
# toc = time.time()
# print('Elapsed time: {:.0f} sec'.format(toc - tic))
# print('Time ratio =', np.max(result_wdaNon['times'] / graph_data['graph_table']['free_flow_time']))
# print('Flow excess =', np.max(result_wdaNon['flows'] / graph_data['graph_table']['capacity']) - 1, end = '\n\n')

# result_wdaNon['elapsed_time'] = toc - tic

#RESULT 

dual_gaps_fwm = result_fwm['duality_gaps']
dual_gaps_cfwm = result_cfwm['duality_gaps']
dual_gaps_cfwm_linesearch = result_cfwm_linesearch['duality_gaps']
dual_gaps_fwm_linesearch = result_fwm_linesearch['duality_gaps']
# dual_gaps_ustm = result_ustm['duality_gaps']
# dual_gaps_ugd = result_ugd['duality_gaps']
# dual_gaps_wda = result_wda['duality_gaps']
# dual_gaps_wdaNon = result_wdaNon['duality_gaps']
iters = np.arange(len(dual_gaps_cfwm))
plt.figure(figsize = (10, 5))

plt.plot(iters, dual_gaps_fwm , color ='red' ,label = 'Frank Wolf')
plt.plot(iters, dual_gaps_fwm_linesearch , color = 'grey' ,label = 'Frank Wolf (linesearch)')
plt.plot(iters, dual_gaps_cfwm , color = 'green' ,label = 'Conjugate Frank Wolf')
plt.plot(iters, dual_gaps_cfwm_linesearch , color = 'brown' ,label = 'Conjugate Frank Wolf (linesearch)')
# plt.plot(iters, dual_gaps_ustm , color ='blue' ,label = 'USTM')
# plt.plot(iters, dual_gaps_ugd , color ='black' ,label = 'UGD')
# plt.plot(iters, dual_gaps_wda , color ='orange' ,label = 'WDA')
# plt.plot(iters, dual_gaps_wdaNon , color ='red' ,label = 'WDA-noncomposite')

plt.ylabel('duality gap', fontsize = 12)
plt.xlabel('iterations', fontsize = 12)
plt.yscale('log')
plt.show()

