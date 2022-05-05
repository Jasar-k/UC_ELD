import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import os


# generate population_size number of matrices of size N*T ( rows: number of units, cols: time periods)

def getRandomUC(N, T, population_size):
    return np.random.randint(0,2,(population_size,N,T))

# function to calculate the uptime (+) or downtime (-) of all the units at all times

def getRunTimeMatrix(unit_commitment_matrix, initial_state=0):
    N,T = np.shape(unit_commitment_matrix)
    run_time_matrix = np.empty((N,T), dtype=int)

    prev_state = initial_state
    
    for t in range(T):
        for n in range(N):
            uc = unit_commitment_matrix[n,t]
            if prev_state[n]<=0 and uc==0:
                run_time_matrix[n,t] = prev_state[n]-1
            elif prev_state[n]<=0 and uc==1:
                run_time_matrix[n,t] = 1
            elif prev_state[n]>=0 and uc==0:
                run_time_matrix[n,t] = -1
            elif prev_state[n]>=0 and uc==1:
                run_time_matrix[n,t] = prev_state[n]+1
        prev_state = run_time_matrix[:,t]

    return run_time_matrix

# Repairing Spinning Reserve Constraints

def repairSpinningReserve(uc, unit_max_power, load, spinning_reserve, avg_full_load_cost):
    N,T = np.shape(uc)
    for t in range(T):
        uc_t = uc[:,t]
        power_supp = np.sum(unit_max_power*uc_t)
        load_t = load[t]
        sr_t = spinning_reserve[t]
        excess_sr_t = power_supp - load_t - sr_t

        priority_list_index = np.argsort(avg_full_load_cost)

        for i in range(N):
            pl_index = priority_list_index[i]
            if excess_sr_t<0 and uc_t[pl_index]==0:
                uc_t[pl_index] = 1
                excess_sr_t = excess_sr_t + unit_max_power[pl_index]
            elif excess_sr_t>=0:
                break
        
        uc[:,t] = uc_t
    
    return uc

# Repairing the constraints related to minimum up and down time

def repairUpDownTimes(uc, min_up_times, min_down_times, run_time_matrix, initial_status):
    N,T = np.shape(uc)
    prev_run_time = initial_status

    for t in range(T):
        for i in range(N):
            prev_rt_i = int(prev_run_time[i])
            current_rt_i = int(run_time_matrix[i,t])
            min_down_time_i = int(min_down_times[i])
            min_up_time_i = int(min_up_times[i])
            t_checkpoint = int(t+min_down_time_i-1)
            if current_rt_i==-1:
                # this means that uc[i,t-1]=1 and uc[i,t]=0 
                if prev_rt_i>0 and prev_rt_i<min_up_time_i:
                    uc[i,t]=1
                elif t_checkpoint<T and run_time_matrix[i,t_checkpoint]>(-1*min_down_time_i):
                    uc[i,t]=1
                elif t_checkpoint>=T:
                    sum_uc = np.sum(uc[i,t:])
                    if sum_uc:
                       uc[i,t]=1
        
        run_time_matrix = getRunTimeMatrix(uc, initial_status)
        prev_run_time = run_time_matrix[:,t]
    
    return [uc, run_time_matrix]


# decommiting the extra units committed because of runtime repairing

def decommitExcessUnits(uc, all_units_max_power, load, spinning_reserve, avg_full_load_cost, run_time_matrix,initial_status, min_up_times, min_down_times):
    N,T = np.shape(uc)

    prev_runtime = initial_status

    for t in range(T):
        # for all time periods
        uc_t = uc[:,t] # all units status at time t
        power_supp = np.sum(all_units_max_power*uc_t) # maximum power available for supply
        load_t = load[t]
        sr_t = spinning_reserve[t]
        excess_sr_t = power_supp - load_t - sr_t # extra/excess spinning reserve committed
        avg_cost_committed_units = avg_full_load_cost*uc_t # this cost is zero for units which are off
        desc_priority_list_index = np.argsort(-1*avg_cost_committed_units)
        
        for i in range(N):
            # iterate through all units at time t
            unit_index = desc_priority_list_index[i] # actual index of the unit; selecting max priced unit first to check for decommitment possibility
            #unit_all_run_times = run_time_matrix[unit_index, :] # runtime vector of the considered unit
            unit_max_power = all_units_max_power[unit_index]

            if avg_cost_committed_units[unit_index] == 0:
                # all the committed units have been iterated through so break the loop
                break
            
            elif unit_max_power<=excess_sr_t:
                min_up_t = int(min_up_times[unit_index])
                min_down_t =int( min_down_times[unit_index])
                prev_rt_i = int(prev_runtime[unit_index])

                if t==(T-1):
                    # decommit if constraint satisfied at t-1, no need to check for next time because it is not there
                    if prev_runtime[unit_index]<=0 or prev_runtime[unit_index]>=min_up_t:
                        uc[unit_index,t]=0
                        excess_sr_t = excess_sr_t - unit_max_power

                elif prev_rt_i>=min_up_t:    
                    next_rt_i = run_time_matrix[unit_index,t+1]
                    if next_rt_i<=0:
                        # if unit is off at next time... [t-1, t, t+1] = [1,1,0]
                        uc[unit_index,t] = 0
                        excess_sr_t = excess_sr_t - unit_max_power
                    elif next_rt_i>0:
                        # if unit is on at next time... [t-1, t, t+1] = [1,1,1] ==> set to zero only if minimum downtime is 1
                        if min_down_t==1:
                            uc[unit_index,t] = 0
                            excess_sr_t = excess_sr_t - unit_max_power

                elif prev_rt_i<=0:
                    next_rt_i = run_time_matrix[unit_index,t+1]
                    if next_rt_i<=0:
                        # if unit is off at next time... [t-1, t, t+1] = [0,1,0]
                        uc[unit_index,t] = 0
                        excess_sr_t = excess_sr_t - unit_max_power
                    elif next_rt_i>0:
                        # if unit is on at next time... [t-1, t, t+1] = [0,1,1]
                        min_up_t = min_up_times[unit_index]
                        check_time_index = int(t + min_up_t)
                        if check_time_index>=T:
                            uc[unit_index,t] = 0
                            excess_sr_t = excess_sr_t - unit_max_power
                        elif run_time_matrix[unit_index, check_time_index]>=min_up_t:
                            uc[unit_index,t] = 0
                            excess_sr_t = excess_sr_t - unit_max_power
        #print("iteration number:", t, "\n", uc)
        run_time_matrix = getRunTimeMatrix(uc, initial_status)
        prev_runtime = run_time_matrix[:,t]

    return [uc, run_time_matrix]        
            

# solving the ELD problem using the lambda iteration method

def lambdaIterationELD(uc, units_max_power, units_min_power, load, tolerance, start_lambda, a, b, c):
    N,T = np.shape(uc)
    power_generation = np.zeros((N,T))

    for t in range(T):
        error = tolerance
        ELDlambda = start_lambda

        while error>=tolerance:
            sum_power = 0
            for i in range(N):
                if uc[i,t]==1:
                    power_i = (ELDlambda-b[i])/[2*c[i]]
                    if power_i>units_max_power[i]:
                        power_i = units_max_power[i]
                    elif power_i<units_min_power[i]:
                        power_i = units_min_power[i]
                    power_generation[i,t] = power_i
                    sum_power = sum_power + power_i
            
            error = load[t]-sum_power
            if error>=0:
                ELDlambda = 1.1*ELDlambda
            else:
                ELDlambda = 0.9*ELDlambda
            
            error = abs(error)
        # print(ELDlambda)

    return power_generation


# returns the value of cost function (objective)

def cost(uc, power_generation, a, b, c, cold_start_hours, hot_start_cost, cold_start_cost, run_time_matrix, initial_state):
    # cost = a + b*power + c*(power^2)
    N,T = np.shape(power_generation)
    A = np.transpose(np.tile(a, (T,1)))
    B = np.transpose(np.tile(b, (T,1)))
    C = np.transpose(np.tile(c, (T,1)))
    power_cost_arr = A*uc + B*power_generation + C*power_generation*power_generation
    power_cost = np.sum(power_cost_arr)
    start_up_cost = 0

    prev_run_time = initial_state
    current_run_time = run_time_matrix[:,0]
    for t in range(T):
        for i in range(N):
            prev_rt_i = prev_run_time[i]
            current_rt_i = current_run_time[i]
            if prev_rt_i<0 and current_rt_i>0:
                # start up of the unit
                cold_start_hr_i = cold_start_hours[i]
                if prev_rt_i<=(-1*cold_start_hr_i):
                    # cold start
                    start_up_cost = start_up_cost + cold_start_cost[i]
                else:
                    # hot start
                    start_up_cost = start_up_cost + hot_start_cost[i]
        if t<(T-1):
            prev_run_time = run_time_matrix[:,t]
            current_run_time = run_time_matrix[:,t+1]

    total_cost = power_cost + start_up_cost
    return total_cost


# Wind generation calculation from wind speed forecast (MULTIDIMENSIONAL) 

def P_wind(forecast_wind_speed,v1,v2,v3,Pwn):
    NW,T = np.shape(forecast_wind_speed)
    P = np.zeros((NW, T))
    for unit_i in range(NW):
        v1_i = v1[unit_i]
        v2_i = v2[unit_i]
        v3_i = v3[unit_i]
        Pwn_i = Pwn[unit_i]
        for t in range (T):  
            fws = forecast_wind_speed[unit_i,t]
            if fws>v1_i and fws<=v2_i:
                P[unit_i,t]=Pwn_i*(fws-v1_i)/(v2_i-v1_i)
            elif fws>v2 and fws<v3_i:
                P[unit_i,t]=Pwn_i
    return P

# Solar PV generation calculation from wind solar irradiance forecast (MULTIDIMENSIONAL) 

def P_solar(Psn,Gt,Gstd,Rc):
    NS,T = np.shape(Gt)
    P = np.zeros((NS, T))
    for unit_i in range(NS):
        Psn_i = Psn[unit_i]
        Rc_i = Rc[unit_i]
        Gstd_i = Gstd[unit_i]
        for t in range (T):  
            Gt_it = Gt[unit_i,t]
            if Gt_it>0 and Gt_it<Rc_i:
                P[unit_i,t]=Psn_i*Gt_it*Gt_it/(Gstd_i*Rc_i)
            if Gt_it>Rc_i:
                P[unit_i,t]=Psn_i*Gt_it/(Gstd_i)
    return P


# Enhanced PSO main function

def uc_epso(input_dict,progress_var,root):
    # get data from dictionary

    # Pnow = input_dict["Pnow"]
    initial_state=input_dict["initial_run_time"]
    P_min = input_dict["Pmin"]
    P_max = input_dict["Pmax"]
    # RU = input_dict["RU"]
    # RD = input_dict["RD"]
    # PZL1 = input_dict["PZL1"]
    # PZU1 = input_dict["PZU1"]
    # PZL2 = input_dict["PZL2"]
    # PZU2 = input_dict["PZU2"]
    cost_hot_start = input_dict["cost_hot_start"]
    cost_cold_start = input_dict["cost_cold_start"]
    cold_start_hrs = input_dict["cold_start_hrs"]
    min_up = input_dict["min_up_time"]
    min_down = input_dict["min_down_time"]
    a = input_dict["a"]
    b = input_dict["b"]
    c = input_dict["c"]
    # d = input_dict["d"]
    # e = input_dict["e"]
    # B00 = input_dict["B00"]
    # B0i = input_dict["B0i"]
    # Bij = input_dict["Bij"]
    c1 = input_dict["c1"]
    c2 = input_dict["c2"]
    w_min = input_dict["w_min"]
    w_max = input_dict["w_max"]
    max_iterations = int(input_dict["max_iters"])
    pop_size = int(input_dict["pop_size"])
    # Load = input_dict["Load"]
    demands=input_dict["Load"]

    bool_wind = input_dict["bool_wind"]
    bool_solar = input_dict["bool_solar"]

    # getting renewable data
    Pow_wind=0
    Pow_solar=0

    if bool_wind:
        # True => wind present
        forecast_wind_speed = input_dict["forecast_wind_speed"]
        v1 = input_dict["v1"]
        v2 = input_dict["v2"]
        v3 = input_dict["v3"]
        Pwn = input_dict["Pwn"]
        Pow_wind = P_wind(forecast_wind_speed,v1,v2,v3,Pwn)
        
    if bool_solar:
        # True => solar present
        Psn = input_dict["Psn"]
        Gt = input_dict["Gt"]
        Gstd = input_dict["Gstd"]
        Rc = input_dict["Rc"]
        Pow_solar = P_solar(Psn,Gt,Gstd,Rc)
        
    # initialization
    N = np.size(P_min)
    T = np.size(demands)

    avg_full_load_cost = a/P_max + b + c*P_max
    priority_list_index = np.argsort(avg_full_load_cost)

    population_uc = getRandomUC(N, T, pop_size)
    population_power_generation = np.empty((pop_size,N,T))
    population_cost = np.empty(pop_size)

    max_cost = np.sum(P_max)*T*np.max(avg_full_load_cost) # all units generating at all times

    gbest_uc = np.empty((N,T), dtype=int)
    gbest_power_generation = np.empty((N,T))
    gbest_cost = max_cost

    pbest_uc = np.empty((pop_size,N,T), dtype=int)
    pbest_power_generation = np.empty((pop_size,N,T))
    pbest_cost = max_cost*np.ones(pop_size)

    tolerance = 1
    start_lambda = 16

    v = np.random.random_sample((pop_size,N,T))

    pop_cost_all_iterations = np.empty((pop_size, max_iterations))

    dw_diter = (w_max-w_min)/max_iterations

    # Dealing with renewables
    Pow_solar_at_T = np.sum(Pow_solar, 0)
    Pow_wind_at_T = np.sum(Pow_wind, 0)
    P_renewable_at_T = (Pow_solar_at_T+Pow_wind_at_T).reshape((-1))
  
    spinning_reserves = 0.1*demands + 0.2*P_renewable_at_T
    net_demands = demands-P_renewable_at_T


    # Main code : Loops and performs PSO

    start_time = time.time()

    print("Starting iteration...")
    for iter_num in range(max_iterations):

        for pop_index in range(pop_size):
            
            uc = population_uc[pop_index,:,:]
            uc = repairSpinningReserve(uc, P_max, net_demands, spinning_reserves, avg_full_load_cost)
            run_time_matrix = getRunTimeMatrix(uc, initial_state)
            [uc, run_time_matrix] = repairUpDownTimes(uc, min_up, min_down, run_time_matrix, initial_state)
            [uc, run_time_matrix] = decommitExcessUnits(uc, P_max, net_demands, spinning_reserves, avg_full_load_cost, run_time_matrix,initial_state, min_up, min_down)
            power_generation = lambdaIterationELD(uc, P_max, P_min, net_demands, tolerance, start_lambda, a, b, c)
            total_cost = cost(uc, power_generation, a, b, c, cold_start_hrs, cost_hot_start, cost_cold_start, run_time_matrix, initial_state)
            
            population_uc[pop_index,:,:] = uc
            population_power_generation[pop_index,:,:] = power_generation
            population_cost[pop_index] = total_cost

            pop_cost_all_iterations[pop_index, iter_num] = total_cost

            #print(population_uc)

            if total_cost<pbest_cost[pop_index]:
                # update personal best
                pbest_cost[pop_index] = total_cost 
                pbest_uc[pop_index,:,:] = uc
                pbest_power_generation[pop_index,:,:] = power_generation

        min_pbest_cost_index = np.argmin(pbest_cost)
        min_pbest_cost = pbest_cost[min_pbest_cost_index]

        if min_pbest_cost<gbest_cost:
            # update global best
            gbest_cost = min_pbest_cost 
            gbest_uc = pbest_uc[min_pbest_cost_index,:,:]
            gbest_power_generation = pbest_power_generation[min_pbest_cost_index,:,:]

        print("Iteration: ", iter_num)
        print("Global best cost: ", gbest_cost)
        print("Execution time: ",time.time()-start_time, " seconds")

        # calculate PSO velocity 

        gbest_uc_3d = np.repeat(gbest_uc[np.newaxis,:, :], pop_size, axis=0)
        rand_pbest = np.random.random_sample((pop_size,N,T))
        rand_gbest = np.random.random_sample((pop_size,N,T))
        v_global = c1*rand_pbest*(gbest_uc_3d - population_uc)
        v_pbest = c2*rand_gbest*(pbest_uc - population_uc)
        w = w_max - dw_diter*iter_num
        v = w*v + v_global + v_pbest

        # update PSO position
        rand_position_update = np.random.random_sample((pop_size,N,T))
        sigmoid_v = 1/(1+np.exp(-v))
        population_uc = np.array(rand_position_update<sigmoid_v, dtype=int)

        progress_var.set((iter_num+1)/max_iterations*100)
        root.update_idletasks()

    print("\n\n")
    print("pop_cost_all_iterations (popultion size X no. of iterations)\n", pop_cost_all_iterations)
    print("\n\n")
    print("gbest_uc\n",gbest_uc)
    print("\n\n")
    print("gbest_power_generation\n",gbest_power_generation)
    print("\n\n")
    print("gbest_power_generation total for each time\n",np.sum(gbest_power_generation,0))
    print("\n\n")
    print("gbest_cost\n",gbest_cost)
    print("\n\n")
    print("Total execution time: ",time.time()-start_time, " seconds")

    output_dict = {
        "best_total_cost":gbest_cost,
        "best_uc":gbest_uc,
        "best_power_gen":gbest_power_generation,
        "time_taken":time.time()-start_time,
    }

    return output_dict


# IBPSO main function

def uc_ibpso(input_dict,progress_var,root):
    # get data from dictionary

    # Pnow = input_dict["Pnow"]
    initial_state=input_dict["initial_run_time"]
    P_min = input_dict["Pmin"]
    P_max = input_dict["Pmax"]
    # RU = input_dict["RU"]
    # RD = input_dict["RD"]
    # PZL1 = input_dict["PZL1"]
    # PZU1 = input_dict["PZU1"]
    # PZL2 = input_dict["PZL2"]
    # PZU2 = input_dict["PZU2"]
    cost_hot_start = input_dict["cost_hot_start"]
    cost_cold_start = input_dict["cost_cold_start"]
    cold_start_hrs = input_dict["cold_start_hrs"]
    min_up = input_dict["min_up_time"]
    min_down = input_dict["min_down_time"]
    a = input_dict["a"]
    b = input_dict["b"]
    c = input_dict["c"]
    # d = input_dict["d"]
    # e = input_dict["e"]
    # B00 = input_dict["B00"]
    # B0i = input_dict["B0i"]
    # Bij = input_dict["Bij"]
    # c1 = input_dict["c1"]
    # c2 = input_dict["c2"]
    # w_min = input_dict["w_min"]
    # w_max = input_dict["w_max"]
    max_iterations = int(input_dict["max_iters"])
    pop_size = int(input_dict["pop_size"])
    # Load = input_dict["Load"]
    demands=input_dict["Load"]

    bool_wind = input_dict["bool_wind"]
    bool_solar = input_dict["bool_solar"]

    # getting renewable data
    Pow_wind=0
    Pow_solar=0

    if bool_wind:
        # True => wind present
        forecast_wind_speed = input_dict["forecast_wind_speed"]
        v1 = input_dict["v1"]
        v2 = input_dict["v2"]
        v3 = input_dict["v3"]
        Pwn = input_dict["Pwn"]
        Pow_wind = P_wind(forecast_wind_speed,v1,v2,v3,Pwn)
        
    if bool_solar:
        # True => solar present
        Psn = input_dict["Psn"]
        Gt = input_dict["Gt"]
        Gstd = input_dict["Gstd"]
        Rc = input_dict["Rc"]
        Pow_solar = P_solar(Psn,Gt,Gstd,Rc)
        
    # initialization
    N = np.size(P_min)
    T = np.size(demands)

    avg_full_load_cost = a/P_max + b + c*P_max
    priority_list_index = np.argsort(avg_full_load_cost)

    population_uc = getRandomUC(N, T, pop_size)
    population_power_generation = np.empty((pop_size,N,T))
    population_cost = np.empty(pop_size)

    max_cost = np.sum(P_max)*T*np.max(avg_full_load_cost) # all units generating at all times

    gbest_uc = np.empty((N,T), dtype=int)
    gbest_power_generation = np.empty((N,T))
    gbest_cost = max_cost

    pbest_uc = np.empty((pop_size,N,T), dtype=int)
    pbest_power_generation = np.empty((pop_size,N,T))
    pbest_cost = max_cost*np.ones(pop_size)

    tolerance = 1
    start_lambda = 16

    v = np.random.random_sample((pop_size,N,T))

    pop_cost_all_iterations = np.empty((pop_size, max_iterations))

    # dw_diter = (w_max-w_min)/max_iterations

    # Dealing with renewables
    Pow_solar_at_T = np.sum(Pow_solar, 0)
    Pow_wind_at_T = np.sum(Pow_wind, 0)
    P_renewable_at_T = (Pow_solar_at_T+Pow_wind_at_T).reshape((-1))
  
    spinning_reserves = 0.1*demands + 0.2*P_renewable_at_T
    net_demands = demands-P_renewable_at_T


    # Main code : Loops and performs PSO

    start_time = time.time()

    print("Starting iteration...")
    for iter_num in range(max_iterations):

        for pop_index in range(pop_size):
            
            uc = population_uc[pop_index,:,:]
            uc = repairSpinningReserve(uc, P_max, net_demands, spinning_reserves, avg_full_load_cost)
            run_time_matrix = getRunTimeMatrix(uc, initial_state)
            [uc, run_time_matrix] = repairUpDownTimes(uc, min_up, min_down, run_time_matrix, initial_state)
            [uc, run_time_matrix] = decommitExcessUnits(uc, P_max, net_demands, spinning_reserves, avg_full_load_cost, run_time_matrix,initial_state, min_up, min_down)
            power_generation = lambdaIterationELD(uc, P_max, P_min, net_demands, tolerance, start_lambda, a, b, c)
            total_cost = cost(uc, power_generation, a, b, c, cold_start_hrs, cost_hot_start, cost_cold_start, run_time_matrix, initial_state)
            
            population_uc[pop_index,:,:] = uc
            population_power_generation[pop_index,:,:] = power_generation
            population_cost[pop_index] = total_cost

            pop_cost_all_iterations[pop_index, iter_num] = total_cost

            #print(population_uc)

            if total_cost<pbest_cost[pop_index]:
                # update personal best
                pbest_cost[pop_index] = total_cost 
                pbest_uc[pop_index,:,:] = uc
                pbest_power_generation[pop_index,:,:] = power_generation

        min_pbest_cost_index = np.argmin(pbest_cost)
        min_pbest_cost = pbest_cost[min_pbest_cost_index]

        if min_pbest_cost<gbest_cost:
            # update global best
            gbest_cost = min_pbest_cost 
            gbest_uc = pbest_uc[min_pbest_cost_index,:,:]
            gbest_power_generation = pbest_power_generation[min_pbest_cost_index,:,:]

        print("Iteration: ", iter_num)
        print("Global best cost: ", gbest_cost)
        print("Execution time: ",time.time()-start_time, " seconds")

        # calculate PSO velocity 
        gbest_uc_3d = np.repeat(gbest_uc[np.newaxis,:, :], pop_size, axis=0)
        w1 = np.random.randint(0,2)
        w2 = np.random.randint(0,2)
        v_pbest = np.bitwise_and(w1, np.bitwise_xor(pbest_uc, population_uc))
        v_global = np.bitwise_and(w2, np.bitwise_xor(gbest_uc_3d, population_uc))
        v = np.bitwise_or(v_pbest, v_global)

        # update PSO position
        population_uc = np.bitwise_xor(population_uc, v)

        progress_var.set((iter_num+1)/max_iterations*100)
        root.update_idletasks()

    print("\n\n")
    print("pop_cost_all_iterations (popultion size X no. of iterations)\n", pop_cost_all_iterations)
    print("\n\n")
    print("gbest_uc\n",gbest_uc)
    print("\n\n")
    print("gbest_power_generation\n",gbest_power_generation)
    print("\n\n")
    print("gbest_power_generation total for each time\n",np.sum(gbest_power_generation,0))
    print("\n\n")
    print("gbest_cost\n",gbest_cost)
    print("\n\n")
    print("Total execution time: ",time.time()-start_time, " seconds")

    output_dict = {
        "best_total_cost":gbest_cost,
        "best_uc":gbest_uc,
        "best_power_gen":gbest_power_generation,
        "time_taken":time.time()-start_time,
    }

    return output_dict



# BGSA main function

def uc_bgsa(input_dict,progress_var,root):
    # get data from dictionary

    # Pnow = input_dict["Pnow"]
    initial_state=input_dict["initial_run_time"]
    P_min = input_dict["Pmin"]
    P_max = input_dict["Pmax"]
    # RU = input_dict["RU"]
    # RD = input_dict["RD"]
    # PZL1 = input_dict["PZL1"]
    # PZU1 = input_dict["PZU1"]
    # PZL2 = input_dict["PZL2"]
    # PZU2 = input_dict["PZU2"]
    cost_hot_start = input_dict["cost_hot_start"]
    cost_cold_start = input_dict["cost_cold_start"]
    cold_start_hrs = input_dict["cold_start_hrs"]
    min_up = input_dict["min_up_time"]
    min_down = input_dict["min_down_time"]
    a = input_dict["a"]
    b = input_dict["b"]
    c = input_dict["c"]
    # d = input_dict["d"]
    # e = input_dict["e"]
    # B00 = input_dict["B00"]
    # B0i = input_dict["B0i"]
    # Bij = input_dict["Bij"]
    # c1 = input_dict["c1"]
    # c2 = input_dict["c2"]
    # w_min = input_dict["w_min"]
    # w_max = input_dict["w_max"]
    # Load = input_dict["Load"]
    demands=input_dict["Load"]

    bool_wind = input_dict["bool_wind"]
    bool_solar = input_dict["bool_solar"]

    # getting renewable data
    Pow_wind=0
    Pow_solar=0

    if bool_wind:
        # True => wind present
        forecast_wind_speed = input_dict["forecast_wind_speed"]
        v1 = input_dict["v1"]
        v2 = input_dict["v2"]
        v3 = input_dict["v3"]
        Pwn = input_dict["Pwn"]
        Pow_wind = P_wind(forecast_wind_speed,v1,v2,v3,Pwn)
        
    if bool_solar:
        # True => solar present
        Psn = input_dict["Psn"]
        Gt = input_dict["Gt"]
        Gstd = input_dict["Gstd"]
        Rc = input_dict["Rc"]
        Pow_solar = P_solar(Psn,Gt,Gstd,Rc)
        
    # initialization

    N = np.size(P_min)
    T = np.size(demands)

    avg_full_load_cost = a/P_max + b + c*P_max
    priority_list_index = np.argsort(avg_full_load_cost)

    max_iterations = int(input_dict["max_iters"])

    pop_size = int(input_dict["G0"])
    K_best = int(input_dict["K_best"])

    population_uc = getRandomUC(N, T, pop_size)
    population_power_generation = np.empty((pop_size,N,T))
    population_cost = np.empty(pop_size)

    max_cost = np.sum(P_max)*T*np.max(avg_full_load_cost) # all units generating at all times

    gbest_uc = np.empty((N,T), dtype=int)
    gbest_power_generation = np.empty((N,T))
    gbest_cost = max_cost

    tolerance = 1
    start_lambda = 16

    # v = np.random.random_sample((pop_size,N,T))
    G0 = input_dict["G0"]
    epsilon = input_dict["epsilon"]
    population_acceleration = np.empty((pop_size,N,T))
    population_velocity = np.random.random_sample((pop_size,N,T))

    pop_cost_all_iterations = np.empty((pop_size, max_iterations))

    # Dealing with renewables
    Pow_solar_at_T = np.sum(Pow_solar, 0)
    Pow_wind_at_T = np.sum(Pow_wind, 0)
    P_renewable_at_T = (Pow_solar_at_T+Pow_wind_at_T).reshape((-1))
  
    spinning_reserves = 0.1*demands + 0.2*P_renewable_at_T
    net_demands = demands-P_renewable_at_T


    # Main code : Loops and performs PSO

    start_time = time.time()

    print("Starting iteration...")
    for iter_num in range(max_iterations):

        for pop_index in range(pop_size):
            
            uc = population_uc[pop_index,:,:]
            uc = repairSpinningReserve(uc, P_max, net_demands, spinning_reserves, avg_full_load_cost)
            run_time_matrix = getRunTimeMatrix(uc, initial_state)
            [uc, run_time_matrix] = repairUpDownTimes(uc, min_up, min_down, run_time_matrix, initial_state)
            [uc, run_time_matrix] = decommitExcessUnits(uc, P_max, net_demands, spinning_reserves, avg_full_load_cost, run_time_matrix,initial_state, min_up, min_down)
            power_generation = lambdaIterationELD(uc, P_max, P_min, net_demands, tolerance, start_lambda, a, b, c)
            total_cost = cost(uc, power_generation, a, b, c, cold_start_hrs, cost_hot_start, cost_cold_start, run_time_matrix, initial_state)
            
            population_uc[pop_index,:,:] = uc
            population_power_generation[pop_index,:,:] = power_generation
            population_cost[pop_index] = total_cost

            pop_cost_all_iterations[pop_index, iter_num] = total_cost

        # Binary gravitational search algorithm ===>
    
        G = G0*(1-iter_num/max_iterations)

        # mass calculation
        pop_cost_asc_sorted_index = np.argsort(population_cost)
        best_index = pop_cost_asc_sorted_index[0]
        worst_index = pop_cost_asc_sorted_index[-1]
        best_cost = population_cost[best_index]
        worst_cost = population_cost[worst_index]
        m = (population_cost-worst_cost)/(best_cost-worst_cost)
        sum_m = np.sum(m)
        M = m/sum_m

        # update g best
        if best_cost<gbest_cost:
            gbest_cost = best_cost
            gbest_power_generation = population_power_generation[best_index,:,:]
            gbest_uc = population_uc[best_index,:,:]


        #print("\n Masses :", M)

        # print best solution 
        best_uc = population_uc[best_index,:,:]
        best_power_generation = population_power_generation[best_index,:,:]
        print("\nIteration: ", iter_num)
        print("Best cost: ", best_cost)
        print("Global best cost: ", gbest_cost)
        print("Execution time: ",time.time()-start_time, " seconds")

        # force and acceleration calculation
        # population_total_force = np.zeros((pop_size,N,T))
        for i_index in range(pop_size):
            # M_i = M[i_index]
            X_i = population_uc[i_index]
            #total_force = 0
            total_acceleration = 0
            for k in range(K_best):
                j_index = pop_cost_asc_sorted_index[k]
                M_j = M[j_index]
                X_j = population_uc[j_index,:,:]
                diff = X_j-X_i
                eucledian_distance = np.sqrt(np.sum(diff*diff))
                #force =(G*(M_i*M_j)*(X_j-X_i))/(eucledian_distance+epsilon)
                #total_force += force
                acceleration =  (G*M_j*(X_j-X_i))/(eucledian_distance+epsilon)
                total_acceleration += acceleration
            
            #population_total_force[i_index,:,:] = total_force
            #population_acceleration[i_index,:,:] = total_force/(M_i+epsilon) # --------------------- divide by zero for worst case
            population_acceleration[i_index,:,:] = total_acceleration
        
        #print(population_total_force)
        #print(population_acceleration)

        # velocity calculation
        random_velocity = np.random.random_sample((pop_size,N,T))
        random_acceleration = np.random.random_sample((pop_size,N,T))
        population_velocity = random_velocity*population_velocity + random_acceleration*population_acceleration

        # position update
        population_abs_tanh_v = abs(np.tanh(population_velocity))
        random_position = np.random.random_sample((pop_size,N,T))
        logic_position = np.array(random_position<population_abs_tanh_v, dtype=int)
        population_uc = abs(logic_position-population_uc)

        progress_var.set((iter_num+1)/max_iterations*100)
        root.update_idletasks()



    print("\n\n")
    print("pop_cost_all_iterations (popultion size X no. of iterations)\n", pop_cost_all_iterations)
    print("\n\n")
    print("gbest_uc\n",gbest_uc)
    print("\n\n")
    print("gbest_power_generation\n",gbest_power_generation)
    print("\n\n")
    print("gbest_power_generation total for each time\n",np.sum(gbest_power_generation,0))
    print("\n\n")
    print("gbest_cost\n",gbest_cost)
    print("\n\n")
    total_time = time.time()-start_time
    print("Total execution time: ",total_time, " seconds")

    output_dict = {
        "best_total_cost":gbest_cost,
        "best_uc":gbest_uc,
        "best_power_gen":gbest_power_generation,
        "time_taken":total_time,
    }

    return output_dict




def uc_gar(input_dict,progressbar,root):

    initial_state=input_dict["initial_run_time"]
    P_min = input_dict["Pmin"]
    P_max = input_dict["Pmax"]
    r_mut=0.008
    r_cross=0.9
    cost_hot_start = input_dict["cost_hot_start"]
    cost_cold_start = input_dict["cost_cold_start"]
    cold_start_hrs = input_dict["cold_start_hrs"]
    min_up = input_dict["min_up_time"]
    min_down = input_dict["min_down_time"]
    a = input_dict["a"]
    b = input_dict["b"]
    c = input_dict["c"]


    generations = int(input_dict["max_iters"])
    pop_size = int(input_dict["pop_size"])
    # Load = input_dict["Load"]
    demands=input_dict["Load"]

    bool_wind = input_dict["bool_wind"]
    bool_solar = input_dict["bool_solar"]

    # getting renewable data
    Pow_wind=0
    Pow_solar=0
    def P_wind(forecast_wind_speed,v1,v2,v3,Pwn):
        NW,T = np.shape(forecast_wind_speed)
        P = np.zeros((NW, T))
        for unit_i in range(NW):
            v1_i = v1[unit_i]
            v2_i = v2[unit_i]
            v3_i = v3[unit_i]
            Pwn_i = Pwn[unit_i]
            for t in range (T):  
                fws = forecast_wind_speed[unit_i,t]
                if fws>v1_i and fws<=v2_i:
                    P[unit_i,t]=Pwn_i*(fws-v1_i)/(v2_i-v1_i)
                elif fws>v2 and fws<v3_i:
                    P[unit_i,t]=Pwn_i
        return P
    
    def P_solar(Psn,Gt,Gstd,Rc):
        NS,T = np.shape(Gt)
        P = np.zeros((NS, T))
        for unit_i in range(NS):
            Psn_i = Psn[unit_i]
            Rc_i = Rc[unit_i]
            Gstd_i = Gstd[unit_i]
            for t in range (T):  
                Gt_it = Gt[unit_i,t]
                if Gt_it>0 and Gt_it<Rc_i:
                    P[unit_i,t]=Psn_i*Gt_it*Gt_it/(Gstd_i*Rc_i)
                if Gt_it>Rc_i:
                    P[unit_i,t]=Psn_i*Gt_it/(Gstd_i)
        return P
    if bool_wind:
        # True => wind present
        forecast_wind_speed = input_dict["forecast_wind_speed"]
        v1 = input_dict["v1"]
        v2 = input_dict["v2"]
        v3 = input_dict["v3"]
        Pwn = input_dict["Pwn"]
        Pow_wind = P_wind(forecast_wind_speed,v1,v2,v3,Pwn)
        
    if bool_solar:
        # True => solar present
        Psn = input_dict["Psn"]
        Gt = input_dict["Gt"]
        Gstd = input_dict["Gstd"]
        Rc = input_dict["Rc"]
        Pow_solar = P_solar(Psn,Gt,Gstd,Rc)    
   
    population_size=pop_size
    tolerance=5
    start_lambda=16
    

    #solar parameters
    
    #
    

    
    

    avg_full_load_cost = a/P_max + b + c*P_max
    Pow_solar_at_T = np.sum(Pow_solar, 0)
    Pow_wind_at_T = np.sum(Pow_wind, 0)
    P_renewable_at_T = (Pow_solar_at_T+Pow_wind_at_T).reshape((-1))
      
    spinning_reserves = 0.1*demands + 0.2*P_renewable_at_T
    net_demands = demands-P_renewable_at_T
    
    demands = net_demands
    
    T = np.size(demands)
    N = np.size(P_max)
    
    
    
    
    #cost should be probabilty=
    # f(i)=1/(1+actualcost(i))
    #p=f(i)/sigma(f(i))
    #
    def costtoprob(cost):
        cost=1/(1+cost)
        cost=cost/sum(cost)
        return cost
    def getRandomUC(N, T, population_size):
        return np.random.randint(0,2,(N,T,population_size))
    
    def getRunTimeMatrix(unit_commitment_matrix, initial_state=0):
        
        N,T = np.shape(unit_commitment_matrix)
        run_time_matrix = np.empty((N,T), dtype=int)
    
        prev_state = initial_state
        
        for t in range(T):
            for n in range(N):
                uc = unit_commitment_matrix[n,t]
                if prev_state[n]<=0 and uc==0:
                    run_time_matrix[n,t] = prev_state[n]-1
                elif prev_state[n]<=0 and uc==1:
                    run_time_matrix[n,t] = 1
                elif prev_state[n]>=0 and uc==0:
                    run_time_matrix[n,t] = -1
                elif prev_state[n]>=0 and uc==1:
                    run_time_matrix[n,t] = prev_state[n]+1
            prev_state = run_time_matrix[:,t]
    
        return run_time_matrix
    
    
    
    def repairSpinningReserve(uc, unit_max_power, load, spinning_reserve, avg_full_load_cost):
        if np.shape(np.shape(uc))[0]==1:
            uc=GenometoUC(uc)
        N,T = np.shape(uc)
        for t in range(T):
            uc_t = uc[:,t]
            power_supp = np.sum(unit_max_power*uc_t)
            load_t = load[t]
            sr_t = spinning_reserve[t]
            excess_sr_t = power_supp - load_t - sr_t
    
            priority_list_index = np.argsort(avg_full_load_cost)
    
            for i in range(N):
                pl_index = priority_list_index[i]
                if excess_sr_t<0 and uc_t[pl_index]==0:
                    uc_t[pl_index] = 1
                    excess_sr_t = excess_sr_t + unit_max_power[pl_index]
                elif excess_sr_t>=0:
                    break
            
            uc[:,t] = uc_t
        
        return uc
    def repairUpDownTimes(uc, min_up_times, min_down_times, run_time_matrix, initial_status):
        if np.shape(np.shape(uc))[0]==1:
            uc=GenometoUC(uc)
        N,T = np.shape(uc)
        prev_run_time = initial_status
    
        for t in range(T):
            for i in range(N):
                prev_rt_i = prev_run_time[i]
                current_rt_i = run_time_matrix[i,t]
                min_down_time_i = min_down_times[i]
                min_up_time_i = min_up_times[i]
                t_checkpoint = int(t+min_down_time_i-1)
                if current_rt_i==-1:
                    # this means that uc[i,t-1]=1 and uc[i,t]=0 
                    if prev_rt_i>0 and prev_rt_i<min_up_time_i:
                        uc[i,t]=1
                    elif t_checkpoint<T and run_time_matrix[i,t_checkpoint]>(-1*min_down_time_i):
                        uc[i,t]=1
                    elif t_checkpoint>=T:
                        sum_uc = np.sum(uc[i,t:])
                        if sum_uc:
                           uc[i,t]=1
            
            run_time_matrix = getRunTimeMatrix(uc, initial_status)
            prev_run_time = run_time_matrix[:,t]
        
        return [uc, run_time_matrix]

    def decommitExcessUnits(uc, all_units_max_power, load, spinning_reserve, avg_full_load_cost, run_time_matrix,initial_status, min_up_times, min_down_times):
        if np.shape(np.shape(uc))[0]==1:
            uc=GenometoUC(uc)
        N,T = np.shape(uc)
        
        prev_runtime = initial_status
    
        for t in range(T):
            # for all time periods
            uc_t = uc[:,t] # all units status at time t
            power_supp = np.sum(all_units_max_power*uc_t) # maximum power available for supply
            load_t = load[t]
            sr_t = spinning_reserve[t]
            excess_sr_t = power_supp - load_t - sr_t # extra/excess spinning reserve committed
            avg_cost_committed_units = avg_full_load_cost*uc_t # this cost is zero for units which are off
            desc_priority_list_index = np.argsort(-1*avg_cost_committed_units)
            
            for i in range(N):
                # iterate through all units at time t
                unit_index = desc_priority_list_index[i] # actual index of the unit; selecting max priced unit first to check for decommitment possibility
                unit_all_run_times = run_time_matrix[unit_index, :] # runtime vector of the considered unit
                unit_max_power = all_units_max_power[unit_index]
    
                if avg_cost_committed_units[unit_index] == 0:
                    # all the committed units have been iterated through so break the loop
                    break
                
                elif unit_max_power<=excess_sr_t:
                    min_up_t = min_up_times[unit_index]
                    min_down_t = min_down_times[unit_index]
                    prev_rt_i = prev_runtime[unit_index]
    
                    if t==(T-1):
                        # decommit if constraint satisfied at t-1, no need to check for next time because it is not there
                        if prev_runtime[unit_index]<=0 or prev_runtime[unit_index]>=min_up_t:
                            uc[unit_index,t]=0
                            excess_sr_t = excess_sr_t - unit_max_power
    
                    elif prev_rt_i>=min_up_t:    
                        next_rt_i = run_time_matrix[unit_index,t+1]
                        if next_rt_i<=0:
                            # if unit is off at next time... [t-1, t, t+1] = [1,1,0]
                            uc[unit_index,t] = 0
                            excess_sr_t = excess_sr_t - unit_max_power
                        elif next_rt_i>0:
                            # if unit is on at next time... [t-1, t, t+1] = [1,1,1] ==> set to zero only if minimum downtime is 1
                            if min_down_t==1:
                                uc[unit_index,t] = 0
                                excess_sr_t = excess_sr_t - unit_max_power
    
                    elif prev_rt_i<=0:
                        next_rt_i = run_time_matrix[unit_index,t+1]
                        if next_rt_i<=0:
                            # if unit is off at next time... [t-1, t, t+1] = [0,1,0]
                            uc[unit_index,t] = 0
                            excess_sr_t = excess_sr_t - unit_max_power
                        elif next_rt_i>0:
                            # if unit is on at next time... [t-1, t, t+1] = [0,1,1]
                            min_up_t = int(min_up_times[unit_index])
                            check_time_index =int( t + min_up_t)
                            if check_time_index>=T:
                                uc[unit_index,t] = 0
                                excess_sr_t = excess_sr_t - unit_max_power
                            elif run_time_matrix[unit_index, check_time_index]>=min_up_t:
                                uc[unit_index,t] = 0
                                excess_sr_t = excess_sr_t - unit_max_power
            #print("iteration number:", t, "\n", uc)
            run_time_matrix = getRunTimeMatrix(uc, initial_status)
            prev_runtime = run_time_matrix[:,t]
    
        return [uc, run_time_matrix]
    def lambdaIterationELD(uc, units_max_power, units_min_power, load, tolerance, start_lambda, a, b, c):
        N,T = np.shape(uc)
        power_generation = np.zeros((N,T))
    
        for t in range(T):
            error = tolerance
            ELDlambda = start_lambda
    
            while error>=tolerance:
                sum_power = 0
                for i in range(N):
                    if uc[i,t]==1:
                        power_i = (ELDlambda-b[i])/[2*c[i]]
                        if power_i>units_max_power[i]:
                            power_i = units_max_power[i]
                        elif power_i<units_min_power[i]:
                            power_i = units_min_power[i]
                        power_generation[i,t] = power_i
                        sum_power = sum_power + power_i
                
                error = load[t]-sum_power
                if error>=0:
                    ELDlambda = 1.1*ELDlambda
                else:
                    ELDlambda = 0.9*ELDlambda
                
                error = abs(error)
            # print(ELDlambda)
    
        return power_generation
    def P_cost(uc, power_generation, a, b, c, cold_start_hours, hot_start_cost, cold_start_cost, run_time_matrix, initial_state):
        N,T = np.shape(power_generation)
        A = np.transpose(np.tile(a, (T,1)))
        B = np.transpose(np.tile(b, (T,1)))
        C = np.transpose(np.tile(c, (T,1)))
        power_cost_arr = A*uc + B*power_generation + C*power_generation*power_generation
        power_cost = np.sum(power_cost_arr)
        start_up_cost = 0
    
        prev_run_time = initial_state
        current_run_time = run_time_matrix[:,0]
        for t in range(T):
            for i in range(N):
                prev_rt_i = prev_run_time[i]
                current_rt_i = current_run_time[i]
                if prev_rt_i<0 and current_rt_i>0:
                    # start up of the unit
                    cold_start_hr_i = cold_start_hours[i]
                    if prev_rt_i<=(-1*cold_start_hr_i):
                        # cold start
                        start_up_cost = start_up_cost + cold_start_cost[i]
                    else:
                        # hot start
                        start_up_cost = start_up_cost + hot_start_cost[i]
            if t<(T-1):
                prev_run_time = run_time_matrix[:,t]
                current_run_time = run_time_matrix[:,t+1]
    
        total_cost = power_cost + start_up_cost
        return total_cost
    def UpDownTimes_Violation(uc, min_up_times, min_down_times, run_time_matrix, initial_status):
        N,T = np.shape(uc)
        prev_run_time = initial_status
        Z=0
        for t in range(T):
            for i in range(N):
                prev_rt_i = prev_run_time[i]
                current_rt_i = run_time_matrix[i,t]
                min_down_time_i = min_down_times[i]
                min_up_time_i = min_up_times[i]
                t_checkpoint = t+min_down_time_i-1
                if current_rt_i==-1:
                    # this means that uc[i,t-1]=1 and uc[i,t]=0 
                    if prev_rt_i>0 and prev_rt_i<min_up_time_i:
                        Z=Z+abs(min_up_time_i-prev_rt_i)
                    elif t_checkpoint<T and run_time_matrix[i,t_checkpoint]>(-1*min_down_time_i):
                        Z=Z+abs(run_time_matrix[i,t_checkpoint]+min_down_time_i)
                    elif t_checkpoint>=T:
                        sum_uc = np.sum(uc[i,t:])
                        if sum_uc:
                           Z=Z+sum_uc
            
            run_time_matrix = getRunTimeMatrix(uc, initial_status)
            prev_run_time = run_time_matrix[:,t]
        
        return Z
    def SpinningReserve_Violation(uc, unit_max_power, load, spinning_reserve, avg_full_load_cost):
        N,T = np.shape(uc)
        Z=0
        for t in range(T):
            uc_t = uc[:,t]
            power_supp = np.sum(unit_max_power*uc_t)
            load_t = load[t]
            sr_t = spinning_reserve[t]
            excess_sr_t = power_supp - load_t - sr_t
    
           
    
            for i in range(N):
                
                if excess_sr_t<0 :
                    Z=Z+abs(excess_sr_t)
                
            
            
        
        return Z
    def lambdaIterationELDDe(uc, units_max_power, units_min_power, load, tolerance, start_lambda, a, b, c):
        N,T = np.shape(uc)
        power_generation = np.zeros((N,T))
        Z=0
        for t in range(T):
            error = tolerance
            ELDlambda = start_lambda
            if np.sum(units_max_power*uc[:,t])<load[t]:
                Z=Z+abs(load[t]-np.sum(units_max_power*uc[:,t]))
                power_generation[:,t]=units_max_power*uc[:,t]
            else:
                
                while error>=tolerance:
                    sum_power = 0
                    for i in range(N):
                        if uc[i,t]==1:
                            power_i = (ELDlambda-b[i])/[2*c[i]]
                            if power_i>units_max_power[i]:
                                power_i = units_max_power[i]
                            elif power_i<units_min_power[i]:
                                power_i = units_min_power[i]
                            power_generation[i,t] = power_i
                            sum_power = sum_power + power_i
                    
                    error = load[t]-sum_power
                    if error>=0:
                        ELDlambda = 1.1*ELDlambda
                    else:
                        ELDlambda = 0.9*ELDlambda
                    
                    error = abs(error)
            # print(ELDlambda)
    
        return (power_generation,Z)
    def total_cost(uc,gen_no,generations):
        if np.shape(np.shape(uc))[0]==1:
            uc=GenometoUC(uc)
        # bar=60
        # if (gen_no<bar):
        #     k=(gen_no+1)/bar
        # else:
        #     k=1.1
        
        run_time_matrix=getRunTimeMatrix(uc, initial_state)
        
        
        power_generation=lambdaIterationELD(uc, P_max, P_min, demands, tolerance, start_lambda, a, b, c)
        Power_cost = P_cost(uc, power_generation, a, b, c, cold_start_hrs, cost_hot_start, cost_cold_start, run_time_matrix, initial_state)
        
        cost=Power_cost
        return(cost)
    def pen_cost(uc,gen_no,generations):
        if np.shape(np.shape(uc))[0]==1:
            uc=GenometoUC(uc)
        # bar=60
        # if (gen_no<bar):
        #     k=(gen_no+1)/bar
        # else:
        #     k=1.1
        k=1.1
        run_time_matrix=getRunTimeMatrix(uc, initial_state)
        sr=50
        ud=10000
        dc=200
        sr_cost=SpinningReserve_Violation(uc, P_max, demands, spinning_reserves, avg_full_load_cost)
        ud_cost=UpDownTimes_Violation(uc, min_up, min_down, run_time_matrix, initial_state)
        (power_generation,demand_cost)=lambdaIterationELDDe(uc, P_max, P_min, demands, tolerance, start_lambda, a, b, c)
        Power_cost = P_cost(uc, power_generation, a, b, c, cold_start_hrs, cost_hot_start, cost_cold_start, run_time_matrix, initial_state)
        Violation_cost=(1.2)*(k)*(sr*sr_cost+dc*demand_cost+ud*ud_cost)
        cost=Power_cost+Violation_cost
        return(cost*cost)
    # def act_cost(uc):
    #     if np.shape(np.shape(uc))[0]==1:
    #         uc=GenometoUC(uc)
    #     run_time_matrix=getRunTimeMatrix(uc, initial_state)
        
    #     (power_generation,demand_cost)=lambdaIterationELD(uc, P_max, P_min, demands, tolerance, start_lambda, a, b, c)
    #     Power_cost = P_cost(power_generation, a, b, c, cold_start_hrs, cost_hot_start,cost_cold_start, run_time_matrix, initial_state)
    #     return Power_cost
    def UCtoGenome(p):
        g=[p[i] for i in range(int(np.shape(p)[0]))]
        g=tuple(g)
        n=np.hstack(g)
        return n
    def GenometoUC(genome):
        
        uc=[genome[i*24:i*24+24] for i in range(int(np.shape(genome)[0]/24))]
        uc=tuple(uc)
        uc=np.vstack(uc)
        return uc
        
    def select_parents(pop,cost,ind):
        
        sel=np.random.choice(ind, 2, replace=False, p=cost)
        parents=np.vstack([pop[sel[0]],pop[sel[1]]])
        return parents
    
    def crossover(pop,cost,r_cross,ind):
        
        parents=select_parents(pop, cost,ind)
        offsprings=parents
        if np.random.rand()<r_cross: 
            
            pivot_point_1 = int(np.shape(pop[0])[0]*0.25)
            pivot_point_2 = int(np.shape(pop[0])[0]*0.70)
            
            offsprings[0] = np.hstack((parents[0][0:pivot_point_1],
                parents[1][pivot_point_1:pivot_point_2],
                parents[0][pivot_point_2:]))
            offsprings[1]=np.hstack((parents[1][0:pivot_point_1],parents[0][pivot_point_1:pivot_point_2],parents[1][pivot_point_2:]))
    
        return offsprings
    def mutation(genome,r_mut):
        if np.random.rand()<r_mut:
            #print("mut")
            n=np.random.randint(int(np.shape(genome)[0]))
            genome[n]= not(genome[n])
        return genome
    def swap_win_op(genome,N):
        
        if np.random.rand()<0.3:
            #print("swop")
            uc=GenometoUC(genome)
            u=np.random.choice(N, 2, replace=False)
            W=np.random.choice(24)
            temp=uc[u[0]][W:W+5]
            uc[u[0]][W:W+5]=uc[u[1]][W:W+5]
            uc[u[1]][W:W+5]=temp
            genome=UCtoGenome(uc)
        return genome
    def win_mutation(genome,N):
        
        if np.random.rand()<0.3:
            #print("wm")
            r=np.random.randint(0,2)
            uc=GenometoUC(genome)
            u=np.random.choice(N)
            W=np.random.choice(24)
            uc[u][W:W+5]=r
            genome=UCtoGenome(uc)
        return genome
    
    def swap_mutation(pop,cost):
        ix=np.argmin(cost)
        gen=pop[ix].copy()
        N=int(np.shape(pop[0])[0]/24)
        (i,j)=np.random.choice(N,2,replace=False)
        if np.random.rand()<0.5:
            #print("swm1")
            temp=gen[i*24+10].copy()
            gen[i*24+10]=gen[j*24+10]
            gen[j*24+10]=temp
        else:
            
            temp=gen.copy()
            temp[i*24+10]=not(temp[i*24+10])
            tc=pen_cost(temp,gen_no,generations)
            if tc<cost[ix]:
                #print("swm2suc")
                gen=temp
                pop[ix]=gen
                cost[ix]=tc
                
        pop[ix]=gen
        
        return (pop,cost)
    
    def swap_win_hill(pop,cost):
        
        if np.random.rand()<0.3:
            #print("swh")
            ix=np.argmax(cost)
            gen=pop[ix].copy()
            N=int(np.shape(pop[0])[0]/24)
            (i,j)=np.random.choice(N,2,replace=False)
                
            for k in range(19):
                tempgen=gen.copy()
                temp=tempgen[i*24+k:i*24+5+k]
                tempgen[i*24+k:i*24+5+k]=tempgen[j*24+k:j*24+5+k]
                tempgen[j*24+k:j*24+5+k]=temp
                tc=pen_cost(tempgen,gen_no,generations)
                if tc<cost[ix]:
                    #print("swmhsuc")
                    gen=tempgen
                    pop[ix]=gen
                    cost[ix]=tc
            
        return (pop,cost)
            
    #maincode
    if population_size<10:
        population_size=12    
    
    ind=[i for i in range(population_size)]
    uc=getRandomUC(N, T, population_size)
    for pop_index in range(population_size):
            
        uci = uc[:,:,pop_index]
        uci = repairSpinningReserve(uci, P_max, demands, spinning_reserves, avg_full_load_cost)
        run_time_matrix = getRunTimeMatrix(uci, initial_state)
        [uci, run_time_matrix] = repairUpDownTimes(uci, min_up, min_down, run_time_matrix, initial_state)
        [uci, run_time_matrix] = decommitExcessUnits(uci, P_max, demands, spinning_reserves, avg_full_load_cost, run_time_matrix,initial_state, min_up, min_down)
        uc[:,:,pop_index]=uci
    ac_cost=[total_cost(uc[:,:,i], 1, generations) for i in range(population_size)]
    ac_cost=np.array(ac_cost)
    print(min(ac_cost))
    cost=costtoprob(ac_cost)
    pop=[UCtoGenome(uc[:,:,i]) for i in range(np.shape(uc)[2])]
    pop=np.array(pop)
    
    
    plot1=[]
    plot2=[]
    start_time = time.time()
    if population_size<5:
        population_size=8
    print(population_size)
    for gen_no in range(generations):
        print("Gen_No:",gen_no)
        elit=pop[np.argmax(cost)]
        next_gen=[elit,pop[np.random.randint(population_size)]]
        for _ in range(int((population_size-2)/2)):
            
            offsprings=crossover(pop, cost, r_cross, ind)
            offsprings[0]=mutation(offsprings[0], r_mut)
            offsprings[1]=mutation(offsprings[1], r_mut) 
            next_gen=np.append(next_gen,offsprings,axis=0)
        
        # for h in range(population_size):
        #     if h==0:
        #         continue
        #     next_gen[h]=swap_win_op(next_gen[h], N)
        # for h in range(population_size):
        #     if h==0:
        #         continue
        #     next_gen[h]=win_mutation(next_gen[h],N)
        
        #(next_gen,ac_cost)=swap_mutation(next_gen, ac_cost)
        #(next_gen,ac_cost)=swap_win_hill(next_gen, ac_cost)
        pop=next_gen
        for pop_index in range(population_size):
            
            uc = pop[pop_index]
            uc = repairSpinningReserve(uc, P_max, demands, spinning_reserves, avg_full_load_cost)
            run_time_matrix = getRunTimeMatrix(uc, initial_state)
            [uc, run_time_matrix] = repairUpDownTimes(uc, min_up, min_down, run_time_matrix, initial_state)
            [uc, run_time_matrix] = decommitExcessUnits(uc, P_max, demands, spinning_reserves, avg_full_load_cost, run_time_matrix,initial_state, min_up, min_down)
            uc=UCtoGenome(uc)
            pop[pop_index]=uc
        m=np.sqrt(min(ac_cost))
        ac_cost=[total_cost(next_gen[i], gen_no, generations) for i in range(population_size)]
        ac_cost=np.array(ac_cost)
        m=min(ac_cost)
        cost=costtoprob(ac_cost)
        plot1.append(m)
        iter_num=gen_no
        max_iterations=generations
        progressbar.set(int((iter_num+1)/max_iterations*100))
        if gen_no+1>=generations:
            progressbar.set(100)
        root.update_idletasks()
        print("Actual Cost:",m)
        print("Execution time: ",time.time()-start_time, " seconds")
    total_time=time.time()-start_time
    mincost=min(ac_cost)
    minuc=GenometoUC(pop[np.argmax(cost)])
    gbest_power_generation=lambdaIterationELD(minuc, P_max, P_min, demands, tolerance, start_lambda, a, b, c)
    output_dict = {
        "best_total_cost":mincost,
        "best_uc":minuc,
        "best_power_gen":gbest_power_generation,
        "time_taken":total_time,
    }
    return output_dict






def uc_gap(input_dict,progressbar,root):
    initial_state=input_dict["initial_run_time"]
    P_min = input_dict["Pmin"]
    P_max = input_dict["Pmax"]
    r_mut=0.008
    r_cross=0.9
    cost_hot_start = input_dict["cost_hot_start"]
    cost_cold_start = input_dict["cost_cold_start"]
    cold_start_hrs = input_dict["cold_start_hrs"]
    min_up = input_dict["min_up_time"]
    min_down = input_dict["min_down_time"]
    a = input_dict["a"]
    b = input_dict["b"]
    c = input_dict["c"]

    tolerance=5
    start_lambda=16
    generations = int(input_dict["max_iters"])
    pop_size = int(input_dict["pop_size"])
    # Load = input_dict["Load"]
    demands=input_dict["Load"]

    bool_wind = input_dict["bool_wind"]
    bool_solar = input_dict["bool_solar"]

    # getting renewable data
    Pow_wind=0
    Pow_solar=0
    def P_wind(forecast_wind_speed,v1,v2,v3,Pwn):
            NW,T = np.shape(forecast_wind_speed)
            P = np.zeros((NW, T))
            for unit_i in range(NW):
                v1_i = v1[unit_i]
                v2_i = v2[unit_i]
                v3_i = v3[unit_i]
                Pwn_i = Pwn[unit_i]
                for t in range (T):  
                    fws = forecast_wind_speed[unit_i,t]
                    if fws>v1_i and fws<=v2_i:
                        P[unit_i,t]=Pwn_i*(fws-v1_i)/(v2_i-v1_i)
                    elif fws>v2 and fws<v3_i:
                        P[unit_i,t]=Pwn_i
            return P

    def P_solar(Psn,Gt,Gstd,Rc):
        NS,T = np.shape(Gt)
        P = np.zeros((NS, T))
        for unit_i in range(NS):
            Psn_i = Psn[unit_i]
            Rc_i = Rc[unit_i]
            Gstd_i = Gstd[unit_i]
            for t in range (T):  
                Gt_it = Gt[unit_i,t]
                if Gt_it>0 and Gt_it<Rc_i:
                    P[unit_i,t]=Psn_i*Gt_it*Gt_it/(Gstd_i*Rc_i)
                if Gt_it>Rc_i:
                    P[unit_i,t]=Psn_i*Gt_it/(Gstd_i)
        return P   
    if bool_wind:
        # True => wind present
        forecast_wind_speed = input_dict["forecast_wind_speed"]
        v1 = input_dict["v1"]
        v2 = input_dict["v2"]
        v3 = input_dict["v3"]
        Pwn = input_dict["Pwn"]
        Pow_wind = P_wind(forecast_wind_speed,v1,v2,v3,Pwn)
        
    if bool_solar:
        # True => solar present
        Psn = input_dict["Psn"]
        Gt = input_dict["Gt"]
        Gstd = input_dict["Gstd"]
        Rc = input_dict["Rc"]
        Pow_solar = P_solar(Psn,Gt,Gstd,Rc)  
        Pow_solar=P_solar(Psn,Gt,Gstd,Rc)
        Pow_wind=P_wind(forecast_wind_speed,v1,v2,v3,Pwn)

    
    
        #solar parameters
        
        #
        

    def costtoprob(cost):
        cost=1/(1+cost)
        cost=cost/sum(cost)
        return cost
    def getRandomUC(N, T, population_size):
        return np.random.randint(0,2,(N,T,population_size))

    def getRunTimeMatrix(unit_commitment_matrix, initial_state=0):
        N,T = np.shape(unit_commitment_matrix)
        run_time_matrix = np.empty((N,T), dtype=int)

        prev_state = initial_state
        
        for t in range(T):
            for n in range(N):
                uc = unit_commitment_matrix[n,t]
                if prev_state[n]<=0 and uc==0:
                    run_time_matrix[n,t] = prev_state[n]-1
                elif prev_state[n]<=0 and uc==1:
                    run_time_matrix[n,t] = 1
                elif prev_state[n]>=0 and uc==0:
                    run_time_matrix[n,t] = -1
                elif prev_state[n]>=0 and uc==1:
                    run_time_matrix[n,t] = prev_state[n]+1
            prev_state = run_time_matrix[:,t]

        return run_time_matrix
    def UpDownTimes_Violation(uc, min_up_times, min_down_times, run_time_matrix, initial_status):
        N,T = np.shape(uc)
        prev_run_time = initial_status
        Z=0
        for t in range(T):
            for i in range(N):
                prev_rt_i = prev_run_time[i]
                current_rt_i = run_time_matrix[i,t]
                min_down_time_i = min_down_times[i]
                min_up_time_i = min_up_times[i]
                t_checkpoint = t+min_down_time_i-1
                if current_rt_i==-1:
                    # this means that uc[i,t-1]=1 and uc[i,t]=0 
                    if prev_rt_i>0 and prev_rt_i<min_up_time_i:
                        Z=Z+abs(min_up_time_i-prev_rt_i)
                    elif t_checkpoint<T and run_time_matrix[i,t_checkpoint]>(-1*min_down_time_i):
                        Z=Z+abs(run_time_matrix[i,t_checkpoint]+min_down_time_i)
                    elif t_checkpoint>=T:
                        sum_uc = np.sum(uc[i,t:])
                        if sum_uc:
                            Z=Z+sum_uc
            
            run_time_matrix = getRunTimeMatrix(uc, initial_status)
            prev_run_time = run_time_matrix[:,t]
        
        return Z
    def SpinningReserve_Violation(uc, unit_max_power, load, spinning_reserve, avg_full_load_cost):
        N,T = np.shape(uc)
        Z=0
        for t in range(T):
            uc_t = uc[:,t]
            power_supp = np.sum(unit_max_power*uc_t)
            load_t = load[t]
            sr_t = spinning_reserve[t]
            excess_sr_t = power_supp - load_t - sr_t

        

            for i in range(N):
                
                if excess_sr_t<0 :
                    Z=Z+abs(excess_sr_t)
                
            
            
        
        return Z
    def lambdaIterationELD(uc, units_max_power, units_min_power, load, tolerance, start_lambda, a, b, c):
        N,T = np.shape(uc)
        power_generation = np.zeros((N,T))
        Z=0
        for t in range(T):
            error = tolerance
            ELDlambda = start_lambda
            if np.sum(units_max_power*uc[:,t])<load[t]:
                Z=Z+abs(load[t]-np.sum(units_max_power*uc[:,t]))
                power_generation[:,t]=units_max_power*uc[:,t]
            else:
                
                while error>=tolerance:
                    sum_power = 0
                    for i in range(N):
                        if uc[i,t]==1:
                            power_i = (ELDlambda-b[i])/[2*c[i]]
                            if power_i>units_max_power[i]:
                                power_i = units_max_power[i]
                            elif power_i<units_min_power[i]:
                                power_i = units_min_power[i]
                            power_generation[i,t] = power_i
                            sum_power = sum_power + power_i
                    
                    error = load[t]-sum_power
                    if error>=0:
                        ELDlambda = 1.1*ELDlambda
                    else:
                        ELDlambda = 0.9*ELDlambda
                    
                    error = abs(error)
            # print(ELDlambda)

        return (power_generation,Z)
    def P_cost(uc, power_generation, a, b, c, cold_start_hours, hot_start_cost, cold_start_cost, run_time_matrix, initial_state):
        N,T = np.shape(power_generation)
        A = np.transpose(np.tile(a, (T,1)))
        B = np.transpose(np.tile(b, (T,1)))
        C = np.transpose(np.tile(c, (T,1)))
        power_cost_arr = A*uc + B*power_generation + C*power_generation*power_generation
        power_cost = np.sum(power_cost_arr)
        start_up_cost = 0

        prev_run_time = initial_state
        current_run_time = run_time_matrix[:,0]
        for t in range(T):
            for i in range(N):
                prev_rt_i = prev_run_time[i]
                current_rt_i = current_run_time[i]
                if prev_rt_i<0 and current_rt_i>0:
                    # start up of the unit
                    cold_start_hr_i = cold_start_hours[i]
                    if prev_rt_i<=(-1*cold_start_hr_i):
                        # cold start
                        start_up_cost = start_up_cost + cold_start_cost[i]
                    else:
                        # hot start
                        start_up_cost = start_up_cost + hot_start_cost[i]
            if t<(T-1):
                prev_run_time = run_time_matrix[:,t]
                current_run_time = run_time_matrix[:,t+1]

        total_cost = power_cost + start_up_cost
        return total_cost
    def total_cost(uc,gen_no,generations, a, b, c, cold_start_hrs, cost_hot_start, cost_cold_start, initial_state,P_max, P_min, demands, tolerance, start_lambda,spinning_reserves, avg_full_load_cost,min_up, min_down):
        if np.shape(np.shape(uc))[0]==1:
            uc=GenometoUC(uc)
        bar=generations*60/200
        if (gen_no<bar):
            k=(gen_no+1)/bar
        else:
            k=1.1
        run_time_matrix=getRunTimeMatrix(uc, initial_state)
        sr=50
        ud=10000
        dc=200
        sr_cost=SpinningReserve_Violation(uc, P_max, demands, spinning_reserves, avg_full_load_cost)
        ud_cost=UpDownTimes_Violation(uc, min_up, min_down, run_time_matrix, initial_state)
        (power_generation,demand_cost)=lambdaIterationELD(uc, P_max, P_min, demands, tolerance, start_lambda, a, b, c)
        Power_cost = P_cost(uc,power_generation, a, b, c, cold_start_hrs, cost_hot_start,cost_cold_start, run_time_matrix, initial_state)
        Violation_cost=(1.2)*(k)*(sr*sr_cost+dc*demand_cost+ud*ud_cost)
        cost=Power_cost+Violation_cost
        return(cost*cost)

    def act_cost(uc,a, b, c, cold_start_hrs, cost_hot_start, cost_cold_start,  initial_state,P_max, P_min, demands, tolerance, start_lambda):
        if np.shape(np.shape(uc))[0]==1:
            uc=GenometoUC(uc)
        run_time_matrix=getRunTimeMatrix(uc, initial_state)
        
        (power_generation,demand_cost)=lambdaIterationELD(uc, P_max, P_min, demands, tolerance, start_lambda, a, b, c)
        Power_cost = P_cost(uc,power_generation, a, b, c, cold_start_hrs, cost_hot_start,cost_cold_start, run_time_matrix, initial_state)
        return Power_cost
    def UCtoGenome(p):
        g=[p[i] for i in range(int(np.shape(p)[0]))]
        g=tuple(g)
        n=np.hstack(g)
        return n
    def GenometoUC(genome):
        
        uc=[genome[i*24:i*24+24] for i in range(int(np.shape(genome)[0]/24))]
        uc=tuple(uc)
        uc=np.vstack(uc)
        return uc
        
    def select_parents(pop,cost,ind):
        
        sel=np.random.choice(ind, 2, replace=False, p=cost)
        parents=np.vstack([pop[sel[0]],pop[sel[1]]])
        return parents

    def crossover(pop,cost,r_cross,ind):
        
        parents=select_parents(pop, cost,ind)
        offsprings=parents
        if np.random.rand()<r_cross: 
            
            pivot_point_1 = int(np.shape(pop[0])[0]*0.25)
            pivot_point_2 = int(np.shape(pop[0])[0]*0.70)
            
            offsprings[0] = np.hstack((parents[0][0:pivot_point_1],
                parents[1][pivot_point_1:pivot_point_2],
                parents[0][pivot_point_2:]))
            offsprings[1]=np.hstack((parents[1][0:pivot_point_1],parents[0][pivot_point_1:pivot_point_2],parents[1][pivot_point_2:]))

        return offsprings
    def mutation(genome,r_mut):
        if np.random.rand()<r_mut:
            #print("mut")
            n=np.random.randint(int(np.shape(genome)[0]))
            genome[n]= not(genome[n])
        return genome
    def swap_win_op(genome,N):
        
        if np.random.rand()<0.3:
            #print("swop")
            uc=GenometoUC(genome)
            u=np.random.choice(N, 2, replace=False)
            W=np.random.choice(24)
            temp=uc[u[0]][W:W+5]
            uc[u[0]][W:W+5]=uc[u[1]][W:W+5]
            uc[u[1]][W:W+5]=temp
            genome=UCtoGenome(uc)
        return genome
    def win_mutation(genome,N):
        
        if np.random.rand()<0.3:
            #print("wm")
            r=np.random.randint(0,2)
            uc=GenometoUC(genome)
            u=np.random.choice(N)
            W=np.random.choice(24)
            uc[u][W:W+5]=r
            genome=UCtoGenome(uc)
        return genome

    def swap_mutation(pop,cost,gen_no,generations,a, b, c, cold_start_hrs, cost_hot_start, cost_cold_start, initial_state,P_max, P_min, demands, tolerance, start_lambda,spinning_reserves, avg_full_load_cost,min_up, min_down):
        ix=np.argmin(cost)
        gen=pop[ix].copy()
        N=int(np.shape(pop[0])[0]/24)
        (i,j)=np.random.choice(N,2,replace=False)
        if np.random.rand()<0.5:
            #print("swm1")
            temp=gen[i*24+10].copy()
            gen[i*24+10]=gen[j*24+10]
            gen[j*24+10]=temp
        else:
            
            temp=gen.copy()
            temp[i*24+10]=not(temp[i*24+10])
            tc=total_cost(temp,gen_no,generations,a, b, c, cold_start_hrs, cost_hot_start, cost_cold_start, initial_state,P_max, P_min, demands, tolerance, start_lambda,spinning_reserves, avg_full_load_cost,min_up, min_down)
            if tc<cost[ix]:
                #print("swm2suc")
                gen=temp
                cost[ix]=tc
                pop[ix]=gen
                
        pop[ix]=gen
        
        return (pop,cost)

    def swap_win_hill(pop,cost,gen_no,generations,a, b, c, cold_start_hrs, cost_hot_start, cost_cold_start, initial_state,P_max, P_min, demands, tolerance, start_lambda,spinning_reserves, avg_full_load_cost,min_up, min_down):
        
        if np.random.rand()<0.3:
            #print("swh")
            ix=np.argmax(cost)
            gen=pop[ix].copy()
            N=int(np.shape(pop[0])[0]/24)
            (i,j)=np.random.choice(N,2,replace=False)
                
            for k in range(19):
                tempgen=gen.copy()
                temp=tempgen[i*24+k:i*24+5+k]
                tempgen[i*24+k:i*24+5+k]=tempgen[j*24+k:j*24+5+k]
                tempgen[j*24+k:j*24+5+k]=temp
                tc=total_cost(tempgen,gen_no,generations,a, b, c, cold_start_hrs, cost_hot_start, cost_cold_start, initial_state,P_max, P_min, demands, tolerance, start_lambda,spinning_reserves, avg_full_load_cost,min_up, min_down)
                if tc<cost[ix]:
                    #print("swmhsuc")
                    gen=tempgen
                    cost[ix]=tc
                    pop[ix]=gen
            
        return (pop,cost)
            
    #maincode

    
        
        

    avg_full_load_cost = a/P_max + b + c*P_max
    Pow_solar_at_T = np.sum(Pow_solar, 0)
    Pow_wind_at_T = np.sum(Pow_wind, 0)
    P_renewable_at_T = (Pow_solar_at_T+Pow_wind_at_T).reshape((-1))
        
    spinning_reserves = 0.1*demands + 0.2*P_renewable_at_T
    net_demands = demands-P_renewable_at_T

    demands = net_demands 

    population_size=pop_size





    T = np.size(demands)
    N = np.size(P_max)
    ind=[i for i in range(population_size)]
    uc=getRandomUC(N, T, population_size)
    ac_cost=[total_cost(uc[:,:,i], 1, generations,a, b, c, cold_start_hrs, cost_hot_start, cost_cold_start, initial_state,P_max, P_min, demands, tolerance, start_lambda,spinning_reserves, avg_full_load_cost,min_up, min_down) for i in range(population_size)]
    ac_cost=np.array(ac_cost)
    cost=costtoprob(ac_cost)
    pop=[UCtoGenome(uc[:,:,i]) for i in range(np.shape(uc)[2])]
    pop=np.array(pop)

    plot1=[]
    plot2=[]
    start_time = time.time()
    for gen_no in range(generations):
        print("Gen_No:",gen_no)
        elit=pop[np.argmax(cost)]
        next_gen=[elit,pop[np.random.randint(population_size)]]
        for _ in range(int((population_size-2)/2)):
            offsprings=crossover(pop, cost, r_cross, ind)
            offsprings[0]=mutation(offsprings[0], r_mut)
            offsprings[1]=mutation(offsprings[1], r_mut) 
            next_gen=np.append(next_gen,offsprings,axis=0)
        
        for h in range(population_size):
            if h==0:
                continue
            next_gen[h]=swap_win_op(next_gen[h], N)
        for h in range(population_size):
            if h==0:
                continue
            next_gen[h]=win_mutation(next_gen[h],N)
        ac_cost=[total_cost(next_gen[i], gen_no, generations,a, b, c, cold_start_hrs, cost_hot_start, cost_cold_start, initial_state,P_max, P_min, demands, tolerance, start_lambda,spinning_reserves, avg_full_load_cost,min_up, min_down) for i in range(population_size)]
        ac_cost=np.array(ac_cost)
        
        (next_gen,ac_cost)=swap_mutation(next_gen, ac_cost,gen_no,generations,a, b, c, cold_start_hrs, cost_hot_start, cost_cold_start, initial_state,P_max, P_min, demands, tolerance, start_lambda,spinning_reserves, avg_full_load_cost,min_up, min_down)
        (next_gen,ac_cost)=swap_win_hill(next_gen, ac_cost,gen_no,generations,a, b, c, cold_start_hrs, cost_hot_start, cost_cold_start, initial_state,P_max, P_min, demands, tolerance, start_lambda,spinning_reserves, avg_full_load_cost,min_up, min_down)
        pop=next_gen
        
        m=np.sqrt(min(ac_cost))
        pc=act_cost(pop[np.argmin(ac_cost)],a, b, c, cold_start_hrs, cost_hot_start, cost_cold_start,  initial_state,P_max, P_min, demands, tolerance, start_lambda)
        
        plot1.append(m)
        plot2.append(pc)
        cost=costtoprob(ac_cost)
        print("Actual Cost:",pc,"Penalised ELITE cost:",np.sqrt(ac_cost[0]),"Penalised Population cost:",m)
        print("Execution time: ",time.time()-start_time, " seconds")
        iter_num=gen_no
        max_iterations=generations
        progressbar.set(int((iter_num+1)/max_iterations*100))
        if gen_no+1>=generations:
            progressbar.set(100)
        root.update_idletasks()
    total_time=time.time()-start_time
    mincost=min(ac_cost)
    minuc=GenometoUC(pop[np.argmax(cost)])
    (gbest_power_generation,demand_cost)=lambdaIterationELD(uc, P_max, P_min, demands, tolerance, start_lambda, a, b, c)
    output_dict = {
        "best_total_cost":mincost,
        "best_uc":minuc,
        "best_power_gen":gbest_power_generation,
        "time_taken":total_time,
    }

    return output_dict
    
    

