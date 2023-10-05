from math import sqrt
import numpy as np
from history import History
from scipy.optimize import minimize_scalar

def fukushima_frank_wolfe_method(oracle, primal_dual_oracle,
                       t_start, max_iter = 1000,
                       eps = 1e-5, eps_abs = None, stop_crit = 'dual_gap_rel',
                       verbose_step = 100, verbose = False, save_history = False , linesearch = False , l_parameter = 1 ,weight_parameter = 0 ,
                       lambda_k=0):
    if stop_crit == 'dual_gap_rel':
        def crit():
            return duality_gap <= eps * duality_gap_init
    elif stop_crit == 'dual_gap':
        def crit():
            return duality_gap <= eps_abs
    elif stop_crit == 'max_iter':
        def crit():
            return it_counter == max_iter
    elif callable(stop_crit):
        crit = stop_crit
    else:
        raise ValueError("stop_crit should be callable or one of the following names: \
                         'dual_gap', 'dual_gap_rel', 'max iter'")
    t = None
    flows = - oracle.grad(t_start)
    t_weighted = np.copy(t_start)
    primal, dual, duality_gap_init, state_msg = primal_dual_oracle(flows, t_weighted) 
    if save_history:
        history = History('iter', 'primal_func', 'dual_func', 'dual_gap')
        history.update(0, primal, dual, duality_gap_init)
    if verbose:
        print(state_msg) 
    if eps_abs is None:
        eps_abs = eps * duality_gap_init
    
    duality_gap_list = []
    relative_gaps = []
    primal_list = []
    y_parameter_list = []    
    time_list = []
    curr_time = oracle.time
    success = False
    gamma = 1 
    
    for it_counter in range(1, max_iter+1):

        t = primal_dual_oracle.get_times(flows)
        y_parameter = primal_dual_oracle.get_flows(t) 
        # primal, dual, duality_gap, state_msg  = primal_dual_oracle(flows, t_weighted)
        # primal_list.append(primal)

        if it_counter == 1 :
            Q = y_parameter

        # Непрерывный случай
        if weight_parameter != 0 :
            Q = Q*(1-weight_parameter)+y_parameter*weight_parameter    
            d_k = Q-flows
        else :    #Дисретный случай
            if len(y_parameter_list) < l_parameter :    
                y_parameter_list.append(y_parameter)
            else :
                y_parameter_list.pop(0)
                y_parameter_list.append(y_parameter)
            nu = np.sum(y_parameter_list,axis=0)/len(y_parameter_list) - flows

            w = y_parameter - flows
            d_k = 0
            if np.sum(t*nu)/np.linalg.norm(nu ,ord=2) < np.sum(t*w)/np.linalg.norm(w ,ord=2) :
                d_k = nu
            else :
                d_k = w
            
        if linesearch :
            # Точно ли надо брать primal_dual_oracle от  (1-gamma*t_weighted +gamma*t  может там просто t_weighted ??????
            # Ответ: да надо , ведь ниже всегда считаются flows и t_weighted одновременно а потом загоняются в функцию
            res = minimize_scalar( lambda y : primal_dual_oracle( flows + y*d_k , (1.0 - gamma) * t_weighted + gamma * t)[0] , bounds = (0.0,1.0) , tol = 1e-12 )
            gamma = res.x
        else :
            gamma = 2.0/(it_counter + 1)

        # lambda модификация aka FWlambda
        if lambda_k != 0 :
            gamma_mod = lambda_k*gamma 
            betta = min(gamma_mod,1)
            if primal_dual_oracle(flows + betta*d_k , (1.0 - betta) * t_weighted + betta * t)[0] < primal_dual_oracle(flows,t_weighted)[0] :
                gamma = betta
        

        flows = flows + gamma*d_k        #по сути то же : flows = (1.0 - gamma) * flows + gamma *
        t_weighted = (1.0 - gamma) * t_weighted + gamma * t
        
        primal, dual, duality_gap, state_msg  = primal_dual_oracle(flows, t_weighted)

        duality_gap_list.append(duality_gap)
        time_list.append(oracle.time - curr_time)
        if save_history:
            history.update(it_counter, primal, dual, duality_gap)
        if verbose and (it_counter % verbose_step == 0):
            print('\nIterations number: {:d}'.format(it_counter))
            print(state_msg, flush = True)
        if crit():
            success = True
            break
     
    result = {'times': t_weighted, 'flows': flows,
              'iter_num': it_counter,
              'res_msg' : 'success' if success else 'iterations number exceeded',
              'duality_gaps' : duality_gap_list , 'relative_gaps': relative_gaps ,
              'primal_list' : primal_list ,
              'time_list': time_list}
    if save_history:
        result['history'] = history.dict
    if verbose:
        print('\nResult: ' + result['res_msg'])
        print('Total iters: ' + str(it_counter))
        print(state_msg)
        print('Oracle elapsed time: {:.0f} sec'.format(oracle.time))
    return result