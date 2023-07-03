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

import osmnx as ox
import networkx as nx 
import pandas as pd 
import copy

graph = ox.graph_from_place('Dolgoprudny, Russia')
result_graph = copy.deepcopy(graph)
# Какие типы дорог хотим оставить
need_keys = ['length','oneway','reversed']
highway_types = ["motorway", "trunk", "primary", "secondary",'service']
edges_to_remove = []
for u, v, key, edge_data in result_graph.edges(keys=True, data=True):
    # KEYS FILTER
    for need_key in need_keys :
        if need_key not in edge_data.keys() :
            edges_to_remove.append((u, v, key))
            break
    # HIGHWAY FILTER
    if 'highway' not in edge_data.keys() :
        edges_to_remove.append((u, v, key))
    else :
        if type(edge_data['highway']) == type(""):
            if edge_data['highway'] not in highway_types :
                edges_to_remove.append((u, v, key))
        else :
            for highway_type in edge_data['highway'] :
                if highway_type not in highway_types :
                    edges_to_remove.append((u, v, key))
                    break
for u, v, key in set(edges_to_remove):
    result_graph.remove_edge(u, v, key)
# Получение списка вершин без ребер
isolated_nodes = [node for node in result_graph.nodes() if result_graph.degree(node) == 0]
# Удаление вершин без ребер
result_graph.remove_nodes_from(isolated_nodes)


graph_table = pd.DataFrame(result_graph.edges(data=True),columns=['init_node' , 'term_node' , 'data'])
graph_table['init_node_thru'] = graph_table['data'].apply(lambda x: True)
graph_table['term_node_thru'] = graph_table['data'].apply(lambda x: True)
# малые дороги : lanes = 1 
graph_table['capacity'] = graph_table['data'].apply(lambda x: x['lanes'] if 'lanes' in x.keys() and type(x['lanes']) != type([])  else 1.0 )
#maxspeed пока по дефолту 32 км/ч = 10 м/c --> Итоговая величина: сек
def getfreeflowtime(x) :
    return int(x['length'])/int(x['maxspeed'])*10/32 if 'maxspeed' in x.keys() and type(x['maxspeed']) == type("") and x['maxspeed'].isdigit() else int(x['length'])/10 
graph_table['free_flow_time'] = graph_table['data'].apply(getfreeflowtime)
graph_table.drop('data', axis=1, inplace=True)

nodes = result_graph.nodes(data=True)
nodes = np.array(nodes)
graph_correspondences = dict()
graph_correspondences[nodes[0][0]] = {'targets' : [nodes[4][0]],'corrs' : [100]}
for i in range(1,30) :
    graph_correspondences[nodes[i][0]] = {'targets' : [nodes[i-1][0]],'corrs' : [100]}
print(graph_correspondences)

graph_data = {
    'nodes number': len(result_graph.nodes(data=False))  , 
    'links number' : len(result_graph.edges(data=False)) , 
    'zones number' :  len(graph_correspondences),
    'graph_table': graph_table
}
total_od_flow = 10000



model = md.Model(graph_data, graph_correspondences, 
                    total_od_flow, mu = 0.25, rho = 0.15)

max_iter = 10

# # Conjugate FWM test

assert(model.mu == 0.25)

print('Conjugate Frank-Wolfe without stopping criteria')
solver_kwargs = {'max_iter' : max_iter, 'stop_crit': 'max_iter',
                 'verbose' : True, 'verbose_step': 2000, 'save_history' : True , "alpha_default" : 0.6}
tic = time.time()
result_cfwm = model.find_equilibrium(solver_name = 'cfwm', solver_kwargs = solver_kwargs)
toc = time.time()
print('Elapsed time: {:.0f} sec'.format(toc - tic))
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

