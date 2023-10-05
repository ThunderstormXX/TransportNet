from math import sqrt
import numpy as np
from history import History
from scipy.optimize import minimize_scalar

def frank_wolfe_method(oracle, primal_dual_oracle,
                       t_start, max_iter = 1000,
                       eps = 1e-5, eps_abs = None, stop_crit = 'dual_gap_rel',
                       verbose_step = 100, verbose = False, save_history = False , linesearch = False , lambda_k = 0):
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

    # print('start values : primal =' , primal , '  dual = ' , dual )

    if save_history:
        history = History('iter', 'primal_func', 'dual_func', 'dual_gap')
        history.update(0, primal, dual, duality_gap_init)
    if verbose:
        print(state_msg) 
    if eps_abs is None:
        eps_abs = eps * duality_gap_init
    
    duality_gap_list = []
    relative_gap_list = []
    primal_list = []
    time_list = []
    curr_time = oracle.time
    success = False
    gamma = 1 
    for it_counter in range(1, max_iter+1):
        t = primal_dual_oracle.get_times(flows)
        yk_FW = primal_dual_oracle.get_flows(t) 
        primal, dual, duality_gap, state_msg  = primal_dual_oracle(flows, t)
        primal_list.append(primal)
        
        if linesearch :
            res = minimize_scalar( lambda y : primal_dual_oracle(( 1.0 - y ) * flows + y * yk_FW , (1.0 - gamma) * t_weighted + gamma * t)[0] , bounds = (0.0,1.0) , tol = 1e-12 )
            gamma = res.x
        else :
            gamma = 2.0/(it_counter + 1)

        # lambda модификация aka FWlambda
        if lambda_k != 0 :
            gamma_mod = lambda_k*gamma 
            betta = min(gamma_mod,1)
            if primal_dual_oracle(( 1.0 -  betta) * flows + betta * yk_FW , (1.0 - betta) * t_weighted + betta * t)[2] < primal_dual_oracle(flows,t_weighted)[2] :
                gamma = betta
        dk_FW = yk_FW - flows
        flows = (1.0 - gamma) * flows + gamma * yk_FW
        t_weighted = (1.0 - gamma) * t_weighted + gamma * t
        
        primal, dual, duality_gap, state_msg  = primal_dual_oracle(flows, t_weighted)
        

        # print(duality_gap ,  - np.dot())

        duality_gap_list.append(duality_gap)
        time_list.append(oracle.time- curr_time)
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
              'duality_gaps' : duality_gap_list ,
              'relative_gaps' : relative_gap_list ,
              'primal_list': primal_list ,
              'time_list' : time_list}
    if save_history:
        result['history'] = history.dict
    if verbose:
        print('\nResult: ' + result['res_msg'])
        print('Total iters: ' + str(it_counter))
        print(state_msg)
        print('Oracle elapsed time: {:.0f} sec'.format(oracle.time))
    return result