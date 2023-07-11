from math import sqrt
import numpy as np
from history import History
from scipy.optimize import minimize_scalar

def conjugate_frank_wolfe_method(oracle, primal_dual_oracle,
                       t_start, max_iter = 1000,
                       eps = 1e-5, eps_abs = None, stop_crit = 'dual_gap_rel',
                       verbose_step = 100, verbose = False, save_history = False ,alpha_default = 0.95 , linesearch = False , restarts = False):
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
    success = False

    duality_gap_list = []
    if restarts :
        all_it_counter = 0
        while all_it_counter < max_iter :
            gamma = 1.0
            t = primal_dual_oracle.get_times(flows) #На самом деле не важно что тут, это чтобы инициализировать t просто
            x_star = primal_dual_oracle.get_flows(t) #На самом деле не важно что тут, это чтобы инициализировать x_star просто
            alpha = 1
            for it_counter in range(1, max_iter+1):
                all_it_counter+=1
                if all_it_counter == max_iter + 1 : 
                    break
                t = primal_dual_oracle.get_times(flows)
                x_aon = primal_dual_oracle.get_flows(t)

                if it_counter > 1 :
                    hessian = primal_dual_oracle.get_diff_times(flows)
                    denom = np.sum(( x_star - flows ) * hessian * ( x_aon - flows_old ))
                    if denom == 0 :
                        alpha = 0
                    else :
                        alpha = np.sum(( x_star - flows ) * hessian * ( x_aon - flows )) / np.sum(( x_star - flows ) * hessian * ( x_aon - flows_old )) 
                    
                    # print(alpha)
                    if alpha < 0 :
                        alpha = 0 
                    if alpha > 1 :
                        alpha = alpha_default


                x_star = x_star*alpha + (1-alpha)*x_aon
                

                res = minimize_scalar( lambda y : primal_dual_oracle(( 1.0 - y ) * flows + y * x_star , (1.0 - gamma) * t_weighted + gamma * t)[2] , bounds = (0.0,1.0) , tol = 1e-12 )
                gamma = res.x

                
                # print(gamma)

                flows_old = flows
                flows = (1.0 - gamma) * flows + gamma * x_star
                t_weighted = (1.0 - gamma) * t_weighted + gamma * t


                primal, dual, duality_gap, state_msg  = primal_dual_oracle(flows, t_weighted)
                duality_gap_list.append(duality_gap)
                if save_history:
                    history.update(it_counter, primal, dual, duality_gap)
                if verbose and (it_counter % verbose_step == 0):
                    print('\nIterations number: {:d}'.format(it_counter))
                    print(state_msg, flush = True)
                if crit():
                    success = True
                    break
                
                if gamma < 1e-7 :
                    break
    elif linesearch :
        gamma = 1.0
        t = primal_dual_oracle.get_times(flows) #На самом деле не важно что тут, это чтобы инициализировать x_star просто
        x_star = primal_dual_oracle.get_flows(t) #На самом деле не важно что тут, это чтобы инициализировать x_star просто
        alpha = 1
        for it_counter in range(1, max_iter+1):
            t = primal_dual_oracle.get_times(flows)
            x_aon = primal_dual_oracle.get_flows(t)

            if it_counter > 1 :
                hessian = primal_dual_oracle.get_diff_times(flows)
                denom = np.sum(( x_star - flows ) * hessian * ( x_aon - flows_old ))
                if denom == 0 :
                    alpha = 0
                else :
                    alpha = np.sum(( x_star - flows ) * hessian * ( x_aon - flows )) / np.sum(( x_star - flows ) * hessian * ( x_aon - flows_old )) 
                
                # print(np.max(flows - flows_old) ,np.sum(( x_star - flows ) * hessian * ( x_aon - flows )) ,np.sum(( x_star - flows ) * hessian * ( x_aon - flows_old )) , alpha)
                if alpha < 0 :
                    alpha = 0 
                if alpha > 0.99 :
                    alpha = alpha_default


            x_star = x_star*alpha + (1-alpha)*x_aon
            # print(gamma)

            res = minimize_scalar( lambda y : primal_dual_oracle(( 1.0 - y ) * flows + y * x_star , (1.0 - gamma) * t_weighted + gamma * t)[2] , bounds = (0.0, 1.0) , tol = 1e-16 )
            gamma = res.x



            flows_old = flows
            flows = (1.0 - gamma) * flows + gamma * x_star
            t_weighted = (1.0 - gamma) * t_weighted + gamma * t


            primal, dual, duality_gap, state_msg  = primal_dual_oracle(flows, t_weighted)
            duality_gap_list.append(duality_gap)
            if save_history:
                history.update(it_counter, primal, dual, duality_gap)
            if verbose and (it_counter % verbose_step == 0):
                print('\nIterations number: {:d}'.format(it_counter))
                print(state_msg, flush = True)
            if crit():
                success = True
                break
    else :
        gamma = 1.0
        t = primal_dual_oracle.get_times(flows) #На самом деле не важно что тут, это чтобы инициализировать t просто
        x_star = primal_dual_oracle.get_flows(t) #На самом деле не важно что тут, это чтобы инициализировать x_star просто
        alpha = 1
        for it_counter in range(1, max_iter+1):
            t = primal_dual_oracle.get_times(flows)
            x_aon = primal_dual_oracle.get_flows(t)

            if it_counter > 1 :
                hessian = primal_dual_oracle.get_diff_times(flows)
                denom = np.sum(( x_star - flows ) * hessian * ( x_aon - flows_old ))
                if denom == 0 :
                    alpha = 0
                else :
                    alpha = np.sum(( x_star - flows ) * hessian * ( x_aon - flows )) / np.sum(( x_star - flows ) * hessian * ( x_aon - flows_old )) 
                
                # print(alpha)
                if alpha < 0 :
                    alpha = 0 
                if alpha > 1 :
                    alpha = alpha_default


            x_star = x_star*alpha + (1-alpha)*x_aon
            gamma = 2/(it_counter+1)



            flows_old = flows
            flows = (1.0 - gamma) * flows + gamma * x_star
            t_weighted = (1.0 - gamma) * t_weighted + gamma * t


            primal, dual, duality_gap, state_msg  = primal_dual_oracle(flows, t_weighted)
            duality_gap_list.append(duality_gap)
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
              'duality_gaps' : duality_gap_list}
    if save_history:
        result['history'] = history.dict
    if verbose:
        print('\nResult: ' + result['res_msg'])
        print('Total iters: ' + str(it_counter))
        print(state_msg)
        print('Oracle elapsed time: {:.0f} sec'.format(oracle.time))
    return result