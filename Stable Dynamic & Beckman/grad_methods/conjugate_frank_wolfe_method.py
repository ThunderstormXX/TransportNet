from math import sqrt
import numpy as np
from history import History
from scipy.optimize import minimize_scalar

def conjugate_frank_wolfe_method(oracle, primal_dual_oracle,
                       t_start, max_iter = 1000,
                       eps = 1e-5, eps_abs = None, stop_crit = 'dual_gap_rel',
                       verbose_step = 100, verbose = False, save_history = False ,alpha_default = 0.95 , linesearch = False ,
                       biconjugate = False):
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
    primal_list = []
   
    if biconjugate :
        gamma = 1.0
        t = primal_dual_oracle.get_times(flows) #На самом деле не важно что тут, это чтобы инициализировать t просто
        flows = primal_dual_oracle.get_flows(t) #На самом деле не важно что тут, это чтобы инициализировать x_star просто
        
        for it_counter in range(1, max_iter+1):
            if it_counter == 1 :
                t = primal_dual_oracle.get_times(flows)
                sk_FW = primal_dual_oracle.get_flows(t)    
                dk = sk_FW - flows
                sk_BFW_old = sk_FW
            elif it_counter == 2 :
                t = primal_dual_oracle.get_times(flows)
                sk_FW = primal_dual_oracle.get_flows(t)    
                dk = sk_FW - flows
                sk_BFW = sk_FW
            else :
                t = primal_dual_oracle.get_times(flows)
                sk_FW = primal_dual_oracle.get_flows(t)
                hessian = primal_dual_oracle.get_diff_times(flows)
                dk_FW = sk_FW - flows

                dk_bar  = sk_BFW - flows
                dk_bbar = gamma * sk_BFW - flows + (1-gamma) * sk_BFW_old
                muk = - np.sum( dk_bbar* hessian * dk_FW ) / np.sum( dk_bbar* hessian * (sk_BFW_old - sk_BFW) )
                nuk = - np.sum( dk_bar* hessian * dk_FW ) / np.sum( dk_bar* hessian * dk_bar) + muk*gamma/(1-gamma)
                muk = max(0, muk)
                nuk = max(0, nuk)
                betta_0 = 1 / ( 1 + muk + nuk )
                betta_1 = nuk * betta_0
                betta_2 = muk * betta_0

                sk_BFW_new = betta_0*sk_FW + betta_1*sk_BFW + betta_2*sk_BFW_old
                sk_BFW_old = sk_BFW
                sk_BWF = sk_BFW_new
                dk_BFW =  sk_BFW - flows
                dk = dk_BFW
            
            if linesearch :
                res = minimize_scalar( lambda y : primal_dual_oracle(flows + y*dk , (1.0 - gamma) * t_weighted + gamma * t)[2] , bounds = (0.0,1.0) , tol = 1e-12 )
                gamma = res.x
            else :
                gamma = 2.0/(it_counter + 1)

            flows = flows + gamma*dk
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
            yk_FW = primal_dual_oracle.get_flows(t)

            if it_counter > 1 :
                hessian = primal_dual_oracle.get_diff_times(flows)
                denom = np.sum(( x_star - flows ) * hessian * ( yk_FW - flows_old ))
                if denom == 0 :
                    alpha = 0
                else :
                    alpha = np.sum(( x_star - flows ) * hessian * ( yk_FW- flows )) / np.sum(( x_star - flows ) * hessian * ( yk_FW- flows_old )) 
                
                # print(alpha)
                if alpha < 0 :
                    alpha = 0 
                if alpha > alpha_default :
                    alpha = alpha_default


            x_star = x_star*alpha + (1-alpha)*yk_FW            
            if linesearch :
                res = minimize_scalar( lambda y : primal_dual_oracle((1-y)*flows + y*x_star , (1.0 - gamma) * t_weighted + gamma * t)[2] , bounds = (0.1,1.0) , tol = 1e-12 )
                gamma = res.x
            else :
                gamma = 2.0/(it_counter + 1)
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




    result = {'times': t_weighted, 'flows': flows,
              'iter_num': it_counter,
              'res_msg' : 'success' if success else 'iterations number exceeded',
              'duality_gaps' : duality_gap_list ,
              'primal_list' : primal_list}
    if save_history:
        result['history'] = history.dict
    if verbose:
        print('\nResult: ' + result['res_msg'])
        print('Total iters: ' + str(it_counter))
        print(state_msg)
        print('Oracle elapsed time: {:.0f} sec'.format(oracle.time))
    return result