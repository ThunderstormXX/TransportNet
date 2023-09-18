#EXPERIMENTS
import test_method as test

#INIT CITY and model
beckmann_save = 'beckmann_results/'
cities_data = 'cities_data/'


### Simple Example
# net_name = cities_data + 'Example_net.tntp'
# trips_name = cities_data + 'Example_trips.tntp'
###Chicago
# net_name = cities_data + 'ChicagoSketch_net.tntp'
# trips_name = cities_data + 'ChicagoSketch_trips.tntp'
### ANAHEIM
# net_name = cities_data + 'Anaheim_net.tntp'
# trips_name = cities_data + 'Anaheim_trips.tntp'
# city_name = 'Anaheim'

###SIOUX FALLS
net_name = cities_data + 'SiouxFalls_net.tntp'
trips_name = cities_data + 'SiouxFalls_trips.tntp'
city_name = 'SiouxFalls'

model,graph_data = test.init_city(beckmann_save, cities_data , net_name , trips_name)

#INIT MAX ITER
max_iter = 100

#INIT METHODS
list_methods = []

### FWM
list_methods.append(( 'fwm', 'Frank Wolfe' , 
    {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose' : True, 'verbose_step': 2000, 'save_history' : True} ))
### FWM linesearch
# list_methods.append(( 'fwm', 'Frank Wolfe linesearch' , 
#     {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose' : True, 'verbose_step': 2000, 'save_history' : True , "linesearch" : True} ))
### FWM lambda
# list_methods.append(( 'fwm', 'Frank Wolfe lambda_k ='+str(1.5) , 
#     {'max_iter' : max_iter, 'stop_crit': 'max_iter',
#                  'verbose' : True, 'verbose_step': 2000, 'save_history' : True , 'lambda_k': 1.5} ))


### CFWM
list_methods.append(( 'cfwm', 'Conjugate Frank Wolfe' , 
    {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose' : True, 'verbose_step': 3000, 'save_history' : True , "alpha_default" : 0.6 } ))
### CFWM linesearch
# list_methods.append(( 'cfwm', 'Conjugate Frank Wolfe linesearch' , 
#     {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose' : True, 'verbose_step': 2000, 'save_history' : True , "alpha_default" : 0.6 , "linesearch" :True} ))

### FWF weighted
# weights = [ 0.5 , 0.2 ]
# for w in weights :
#     list_methods.append(( 'fwf' , 'Fukushima Frank-Wolfe weighted =' + str(w) ,
#         {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose' : True, 'verbose_step': 2000, 'save_history' : True , 'weight_parameter' : w ,  'linesearch': True }))

### FWF l_param with lambda mod
# l_parameters = [2,5]
# lambda_k = 1.5
# for l in l_parameters:
#     list_methods.append(( 'fwf' , 'Fukushima Frank-Wolfe lambda='+str(lambda_k)+' with l_param =' + str(l) ,
#         {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose' : True, 'verbose_step': 2000, 'save_history' : True ,
#                       'l_parameter' : l , 'linesearch' : True , 'lambda_k': lambda_k}))
    
### Biconjugate Frank Wolfe 
list_methods.append(( 'cfwm', 'Biconjugate Frank Wolfe' , 
    {'max_iter' : max_iter, 'stop_crit': 'max_iter',
                 'verbose' : True, 'verbose_step': 2000, 'save_history' : True , 'biconjugate' : True } ))

### USTM
# eps_abs = 31
# list_methods.append(( 'ustm', 'USTM with eps_abs ='+ str(eps_abs) , 
#     {'eps_abs': eps_abs,'max_iter': max_iter, 'stop_crit': 'dual_gap', 'verbose' : True, 'verbose_step': 2000, 'save_history' : True} ))
### UGD 
# eps_abs = 31
# list_methods.append(( 'ugd', 'UGD with eps_abs ='+ str(eps_abs) , 
#     {'eps_abs': eps_abs,'max_iter': max_iter, 'stop_crit': 'dual_gap','verbose' : True, 'verbose_step': 4000, 'save_history' : True} ))
### WDA 
# list_methods.append(( 'wda', 'WDA' , 
#     {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose': True, 'verbose_step': 4000, 'save_history': True} ))
### WDA-noncomposite
# list_methods.append(( 'wda', 'WDA noncomposote' , 
#     {'max_iter' : max_iter, 'stop_crit': 'max_iter','verbose': True, 'verbose_step': 4000, 'save_history' : True} ))



#RUN EXPERIMENTS
experiments = []
for method , name , solver_kwargs in list_methods :
    result ,_,_,_ = test.run_method(method , name , solver_kwargs , model ,graph_data ,city_name = city_name , max_iter = max_iter)
    experiments.append((result , name , max_iter , city_name))

#DISPLAY RESULTS
test.save_and_plot( experiments , d_gap_display = True , primal_display = False )

