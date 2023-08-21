import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
np.set_printoptions(suppress=True)

import data_handler as dh
import sinkhorn as skh
import model as md
import csv

import pandas as pd

net_name = './data/vl_links.txt'
trips_name = './data/vl_trips_test.txt'
parsers = 'vladik'

graph_table_name = './data/graph_table_test.csv'

# net_name = '../data/SiouxFalls_net.tntp'
# trips_name = '../data/SiouxFalls_trips.tntp'
# parsers = 'tntp'

best_sink_beta = 0.005
sink_num_iter, sink_eps = 2500, 10**(-8)
INF_COST = 100
INF_TIME = 1e10


def get_times_inverse_func(capacity, times, rho = 0.15, mu=0.25):
    capacities = capacity.to_numpy()
    # FIXME  IS IT A BUG???
    freeflowtimes = times #graph_table['free_flow_time'].to_numpy()
    # print('hm: ', np.power(times / freeflowtimes, mu))
    return np.transpose( (capacities / rho) * (np.power(times / freeflowtimes, mu) - 1.0))

def get_LW(L_dict, W_dict, new_to_old):
    L = np.array([L_dict[new_to_old[i]] for i in range(len(L_dict))], dtype=np.double)
    W = np.array([W_dict[new_to_old[i]] for i in range(len(W_dict))], dtype=np.double)
    L = handler.distributor_L_W(L)
    W = handler.distributor_L_W(W)
    people_num = L.sum()
    print(type(L))
    L /= np.nansum(L)
    W /= np.nansum(W)
    return L, W, people_num


if __name__ == '__main__':

    handler = dh.DataHandler()
    
    # graph_data = handler.GetGraphData(net_name, eval(f'handler.{parsers}_net_parser'), columns=['init_node', 'term_node', 'capacity', 'free_flow_time'])
    graph_table = pd.read_csv(graph_table_name, low_memory=False , sep = ' ')
    graph_data = {}
    graph_data['graph_table'] = graph_table
    graph_data['nodes number'] = len(set(graph_table.init_node.values) | set(graph_table.term_node.values))
    graph_data['links number'] = graph_table.shape[0]
    print('nUMBER OF NODES, LINKS: ', graph_data['nodes number'], 
            graph_data['links number'])
    # print(graph_data)
    
    L_dict, W_dict = handler.GetLW_dicts(trips_name, eval(f'handler.{parsers}_corr_parser'))

    handler = dh.DataHandler()

    max_iter = 2
    alpha = 0.9

    # no "corrs" (d_ij) in empty corr dict - no need for them in the problem input, but it's convenient to create
    # corr_dict to use existing functions
    empty_corr_dict = {source: {'targets': list(W_dict.keys())} for source in L_dict.keys()}
    empty_corr_matrix, old_to_new, new_to_old = handler.reindexed_empty_corr_matrix(empty_corr_dict)
    print('fill correspondence_matrix')

    print('init LW')
    L, W, people_num = get_LW(L_dict, W_dict, new_to_old)
    total_od_flow = people_num


    model = md.Model(graph_data, empty_corr_dict, total_od_flow, mu=0.25)

    T_dict = handler.get_T_from_t(graph_data['graph_table']['free_flow_time'],
                                             graph_data, model)
    T = handler.T_matrix_from_dict(T_dict, empty_corr_matrix.shape, old_to_new)

    if parsers == 'vladik':
        best_sink_beta = T.shape[0] / np.nansum(T)

    for ms_i in range(4):

        print('iteration: ', ms_i)

        s = skh.Sinkhorn(L, W, people_num, sink_num_iter, sink_eps)
        T = np.nan_to_num(T, nan=0, posinf=0, neginf=0)

        best_sink_beta = T.shape[0] / np.nansum(T)

        # зачем тут nan_to_num если Т уже через него пропущена с другим nan=
        cost_matrix = np.nan_to_num(T * best_sink_beta, nan=INF_COST)
        rec, _, _ = s.iterate(cost_matrix)
        sink_correspondences_dict = handler.corr_matrix_to_dict(rec, new_to_old)

        # L, W, people_num = get_LW(rec)
        L_new = np.nansum(rec, axis=1)
        L_new /= np.nansum(L_new)
        W_new = np.nansum(rec, axis=0)
        W_new /= np.nansum(W_new)
        # print('L:', np.isclose(L, L_new) , sep = '\n')
        # print('W:', W == W_new, sep = '\n')
        # print('cost matrix : ',cost_matrix)
        # print('rec ; ' ,rec)
        # print('L :', L)
        # print('Lnew :', L_new)
        # print('W :', W)
        # print('Wnew :', W_new)
        assert(np.allclose(L, L_new))
        assert(np.allclose(W, W_new))

        model.refresh_correspondences(graph_data, sink_correspondences_dict)

        for i, eps_abs in enumerate(np.logspace(1, 3, 1)):
            solver_kwargs = {'eps_abs': eps_abs,
                             'max_iter': max_iter}

            result = model.find_equilibrium(solver_name='ustm', composite=True,
                                            solver_kwargs=solver_kwargs,
                                            base_flows=alpha * graph_data['graph_table']['capacity'])

        model.graph.update_flow_times(result['times'])
        T_dict = result['zone travel times']
        T = handler.T_matrix_from_dict(T_dict, rec.shape, old_to_new)

        # no impact on iterations from code below
        flows_inverse_func = get_times_inverse_func(graph_data['graph_table']['capacity'], result['times'], rho=0.15, mu=0.25)
        subg = result['subg']
        # for key in result['subg'].keys():
        #     subg[key[0] - 1][key[1] - 1] = result['subg'][key]
        print('subg shape: ', np.shape(subg), 'flows_inv shape: ',  np.shape(flows_inverse_func))

        if max_iter == 2:

            np.savetxt('KEV_res/multi/flows/' + str(ms_i) + '_flows.txt', result['flows'], delimiter=' ')
            np.savetxt('KEV_res/multi/times/' + str(ms_i) + '_time.txt', result['times'], delimiter=' ')
            np.savetxt('KEV_res/multi/corr_matrix/' + str(ms_i) + '_corr_matrix.txt', rec, delimiter=' ')
            np.savetxt('KEV_res/multi/inverse_func/' + str(ms_i) + '_inverse_func.txt', flows_inverse_func,
                       delimiter=' ')
            np.savetxt('KEV_res/multi/subg/' + str(ms_i) + '_nabla_func.txt', subg, delimiter=' ')

        elif max_iter == 1:
            print('Mistake, should be 2! Counter in ustm starts from 1!')

        else:
            np.savetxt('KEV_res/iter/flows/' + str(ms_i) + '_flows.txt', result['flows'], delimiter=' ')
            np.savetxt('KEV_res/iter/times/' + str(ms_i) + '_time.txt', result['times'], delimiter=' ')
            np.savetxt('KEV_res/iter/corr_matrix/' + str(ms_i) + '_corr_matrix.txt', rec, delimiter=' ')
            np.savetxt('KEV_res/multi/inverse_func/' + str(ms_i) + '_inverse_func.txt', flows_inverse_func,
                       delimiter=' ')
            np.savetxt('KEV_res/multi/subg/' + str(ms_i) + '_nabla_func.txt', subg, delimiter=' ')

    with open('KEV_res/' + 'result.csv', 'w') as f:
        w = csv.DictWriter(f, result.keys())
        w.writeheader()
        w.writerow(result)
