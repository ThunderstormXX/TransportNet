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
import datetime


# по названию города ==> модель и таблицу графа
def init_city(beckmann_save , net_name , trips_name) :
    handler = dh.DataHandler()
    graph_data = handler.GetGraphData(net_name, columns = ['init_node', 'term_node', 'capacity', 'free_flow_time'],parser= eval(f'handler.tntp_net_parser'))
    graph_correspondences, total_od_flow = handler.GetGraphCorrespondences(trips_name)
    model = md.Model(graph_data, graph_correspondences, 
                        total_od_flow, mu = 0.25, rho = 0.15)
    graph_data['graph_table'].head()
    return model ,graph_data

# Запуск моделей с выбором числа итераций и города и аргументов метода и кастомный нейминг
def run_method(method , name , solver_kwargs , model ,graph_data ,city_name = '' , max_iter = 100) :
    assert(model.mu == 0.25)
    print('----------------------Running ' ,name ,'in ', max_iter ,' iters , on city ', city_name,'--------------------------------')
    tic = time.time()
    result_fwm = model.find_equilibrium(solver_name = method, solver_kwargs = solver_kwargs)
    toc = time.time()
    print('Elapsed time: {:.0f} sec'.format(toc - tic))
    print('----------------------END--------------------------------------------------------')
    return result_fwm , method , max_iter , city_name
def run_experiment(list_methods , model , graph_data , city_name , max_iter) :
    experiments = []
    for method , name , solver_kwargs in list_methods :
        result ,_,_,_ = run_method(method , name , solver_kwargs , model ,graph_data ,city_name = city_name , max_iter = max_iter)
        experiments.append((result , name , max_iter , city_name))
    return experiments

# список графиков ==>> вывод и сохранение
def plot( experiments , d_gap_display = True , primal_display = False , time_display = False , loglog =False , save = False) :
    color_generator = plt.cm.get_cmap('tab20', len(experiments))
    colors = [color_generator(i) for i in np.linspace(0, 1, len(experiments))]
    time = datetime.datetime.now().time().strftime("%H:%M")
    date = datetime.datetime.now().date()
    experiments_folder = './experiments_results/'
    if d_gap_display :       
        plt.figure(figsize = (10, 5))
        for col_id , experiment in enumerate(experiments) :
            result , name , max_iter , city_name = experiment
            dual_gaps = result['duality_gaps']
            iters = np.arange(max_iter)
            if loglog == False :
                plt.plot(iters, dual_gaps , color =colors[col_id] ,label = name)
            else :
                plt.loglog(iters, dual_gaps , color =colors[col_id] ,label = name)
            experiment_path = experiments_folder +'iterations/dgaps_'+ name + '_' + city_name + '_' + str(max_iter) + 'iters_datetime_' + str(date)+'_' + str(time)+ '.csv'
            if save :
                df_dual_gaps = pd.DataFrame(dual_gaps)
                df_dual_gaps.to_csv( experiment_path  , index=False)
        
        plt.ylabel('duality gap', fontsize = 12)
        plt.xlabel('iterations', fontsize = 12)
        plt.title('Сходимость duality gap на городе ' + city_name)
        plt.legend()
        if loglog == False :
            plt.yscale('log')
        if save :
            plt.savefig(experiments_folder +'pictures/'+ 'Experiment_dualgap_date_'+ str(date) +'_time_'+str(time)+ '_city_' + city_name + '_' + str(max_iter) + 'iters' + '.png')
        plt.show()
    elif primal_display :
        plt.figure(figsize = (10, 5))
        for col_id , experiment in enumerate(experiments) :
            result , name , max_iter , city_name = experiment
            primals = result['duality_gaps']
            iters = np.arange(max_iter)
            if loglog == False :
                plt.plot(iters, primals , color =colors[col_id] ,label = name)
            else :
                plt.loglog(iters, primals , color =colors[col_id] ,label = name)
            experiment_path = experiments_folder+ 'iterations/primals_' + name + '_' + city_name + '_' + str(max_iter) + 'iters' + '.csv'
            if save :
                df_primals = pd.DataFrame(primals)
                df_primals.to_csv( experiment_path  , index=False)
        plt.ylabel('primals', fontsize = 12)
        plt.xlabel('iterations', fontsize = 12)
        plt.title('Сходимость primal на городе ' + city_name)
        plt.legend()
        if loglog == False :
            plt.yscale('log')
        if save :
            plt.savefig(experiments_folder +'pictures/'+'Experiments_primal_date_'+ str(date) +'_time_'+str(time)+ '_city_' + city_name + '_' + str(max_iter) + 'iters' + '.png')
        plt.show()
    elif time_display :
        plt.figure(figsize = (10, 5))
        for col_id , experiment in enumerate(experiments) :
            result , name , max_iter , city_name = experiment
            times = result['time_list']
            dual_gaps = result['duality_gaps']
            if loglog == False :
                plt.plot(times, dual_gaps , color =colors[col_id] ,label = name)
            else :
                plt.loglog(times, dual_gaps , color =colors[col_id] ,label = name)
            experiment_path = experiments_folder+ 'iterations/dgaps_by_times_' + name + '_' + city_name + '_' + str(max_iter) + 'iters' + '.csv'
            if save :
                df_times = pd.DataFrame(times)
                df_times.to_csv( experiment_path  , index=False)
        plt.ylabel('d_gaps', fontsize = 12)
        plt.xlabel('times (sec)', fontsize = 12)
        plt.title('Сходимость dgap на городе ' + city_name)
        plt.legend()
        if loglog == False :
            plt.yscale('log')
        if save :
            plt.savefig(experiments_folder +'pictures/'+'Experiments_dgaps_by_time_date_'+ str(date) +'_time_'+str(time)+ '_city_' + city_name + '_' + str(max_iter) + 'iters' + '.png')
        plt.show()
# Кастомно прочекать графики по значениям целевой функции в file.csv
def display(filename , last_iters = 0 ) :
    values = pd.read_csv(filename ).values
    
    plt.figure(figsize = (10, 5))

    iters = np.arange(len(values) if last_iters == 0 else last_iters)
    n = len(iters)
    values = values[-n:] 
    plt.plot(iters, values  ,label = filename)
    plt.ylabel('values', fontsize = 12)
    plt.xlabel('iterations', fontsize = 12)
    plt.title('Сходимость values ')
    plt.legend()
    plt.yscale('log')
    plt.show()


# display('./experiments_results/Frank Wolfe_SiouxFalls_100iters.csv' , 10)


