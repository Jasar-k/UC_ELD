import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
# from gui_uc_eld import progressbar


def getRandomELD(pop_size, num_units, Plower, Pupper):
    pop = np.random.uniform(Plower, Pupper, (num_units, pop_size))
    # pop = num_units*pop_size
    return pop

def repairOperationRange(pop, PZL1, PZU1, PZL2, PZU2, Plower, Pupper):
    num_units, pop_size = np.shape(pop)
    # iterating through all the units
    for unit_i in range(num_units):
        pzl1 = PZL1[unit_i]
        pzl2 = PZL2[unit_i]
        pzu1 = PZU1[unit_i]
        pzu2 = PZU2[unit_i]
        minimum = Plower[unit_i]
        maximum = Pupper[unit_i]
        # print(minimum) 
        for pop_i in range(pop_size):
            if pop[unit_i,pop_i]<minimum:
                pop[unit_i,pop_i] = minimum
            elif pop[unit_i,pop_i]>maximum:
                pop[unit_i,pop_i] = maximum
            elif pop[unit_i,pop_i]>pzl1 and pop[unit_i,pop_i]<pzu1:
                # pop[unit_i,pop_i] = random.choice([pzl1,pzu1])
                pop[unit_i,pop_i] = pzl1 + (pzu1-pzl1)*np.round((pop[unit_i,pop_i]-pzl1)/(pzu1-pzl1))
            elif pop[unit_i,pop_i]>pzl2 and pop[unit_i,pop_i]<pzu2:
                # pop[unit_i,pop_i] = random.choice([pzl2,pzu2])
                pop[unit_i,pop_i] = pzl2 + (pzu2-pzl2)*np.round((pop[unit_i,pop_i]-pzl2)/(pzu2-pzl2))
    return pop

def getCost(pop, a, b, c, d, e, Pmin):
    _, pop_size = np.shape(pop)
    cost = np.zeros([pop_size,1])
    for pop_i in range(pop_size):
        particle_i = pop[:, pop_i].reshape((-1,1))
        unitwise_cost = a + b*particle_i + c*particle_i*particle_i #+ np.abs(d*np.sin(e*(Pmin-particle_i)))
        cost[pop_i, 0] = np.sum(unitwise_cost)
    return cost

def getTransmissionLosses(pop, Bij, B0i, B00):
    _, pop_size = np.shape(pop)
    losses = np.zeros([pop_size,1])
    for pop_i in range(pop_size):
        particle_i = pop[:, pop_i].reshape((-1,1))
        losses[pop_i,0] = (np.dot(np.transpose(particle_i), np.dot(Bij, particle_i)) + np.dot(B0i, particle_i) + B00)
    return losses

def getEvaluationValue(pop, cost, Load, losses, Fmax, Fmin):
    # Financial/generatoin cost
    Fcost = 1 + (cost-Fmin)/(Fmax-Fmin)
    # power balance cost/penalty
    total_gen = np.reshape(np.sum(pop,0), (-1,1))
    diff = total_gen - losses - Load
    Ppbc = 1 + 100*diff*diff
    # evaluation value
    f = 1/(Fcost + Ppbc)
    return f

# def getCostCurves(Pmin, Pmax, a, b, c, d=0, e=0):
#     return 0

def P_wind(forecast_wind_speed,v1,v2,v3,Pwn):
    NW = np.size(forecast_wind_speed)
    P = np.zeros((1,NW))
    for unit_i in range(NW):
        v1_i = v1[unit_i]
        v2_i = v2[unit_i]
        v3_i = v3[unit_i]
        Pwn_i = Pwn[unit_i]
        fws = forecast_wind_speed[unit_i]
        if fws>v1_i and fws<=v2_i:
            P[unit_i]=Pwn_i*(fws-v1_i)/(v2_i-v1_i)
        elif fws>v2 and fws<v3_i:
            P[unit_i]=Pwn_i
    return P

def P_solar(Psn,Gt,Gstd,Rc):
    NS = np.size(Gt)
    P = np.zeros((1,NS))
    for unit_i in range(NS):
        Psn_i = Psn[unit_i]
        Rc_i = Rc[unit_i]
        Gstd_i = Gstd[unit_i]
        Gt_i = Gt[unit_i]
        if Gt_i>0 and Gt_i<Rc_i:
            P[unit_i]=Psn_i*Gt_i*Gt_i/(Gstd_i*Rc_i)
        if Gt_i>Rc_i:
            P[unit_i]=Psn_i*Gt_i/(Gstd_i)
    return P


def eld_pso(input_dict,progress_var,root):
    # create variables from dictionary
    Pnow = input_dict["Pnow"]
    Pmin = input_dict["Pmin"]
    Pmax = input_dict["Pmax"]
    UR = input_dict["RU"]
    DR = input_dict["RD"]
    PZL1 = input_dict["PZL1"]
    PZU1 = input_dict["PZU1"]
    PZL2 = input_dict["PZL2"]
    PZU2 = input_dict["PZU2"]
    a = input_dict["a"]
    b = input_dict["b"]
    c = input_dict["c"]
    d = input_dict["d"]
    e = input_dict["e"]
    B00 = input_dict["B00"]
    B0i = input_dict["B0i"]
    Bij = input_dict["Bij"]
    c1 = input_dict["c1"]
    c2 = input_dict["c2"]
    w_min = input_dict["w_min"]
    w_max = input_dict["w_max"]
    num_iters = int(input_dict["max_iters"])
    pop_size = int(input_dict["pop_size"])
    Load = input_dict["Load"]

    bool_wind = input_dict["bool_wind"]
    bool_solar = input_dict["bool_solar"]
    
    net_load = Load

    Pow_wind = 0
    Pow_solar = 0
    
    if bool_wind:
        # True => wind present
        forecast_wind_speed = input_dict["forecast_wind_speed"]
        v1 = input_dict["v1"]
        v2 = input_dict["v2"]
        v3 = input_dict["v3"]
        Pwn = input_dict["Pwn"]
        Pow_wind = P_wind(forecast_wind_speed,v1,v2,v3,Pwn)
        net_load = net_load-np.sum(Pow_wind)
    
    if bool_solar:
        # True => solar present
        Psn = input_dict["Psn"]
        Gt = input_dict["Gt"]
        Gstd = input_dict["Gstd"]
        Rc = input_dict["Rc"]
        Pow_solar = P_solar(Psn,Gt,Gstd,Rc)
        net_load = net_load-np.sum(Pow_solar)
        
    # initialise

    num_units = np.size(Pnow)

    Plower = np.maximum(Pmin, Pnow-DR)
    Pupper = np.minimum(Pmax, Pnow+UR)

    pop = getRandomELD(pop_size, num_units, Plower, Pupper)
    pop = repairOperationRange(pop, PZL1, PZU1, PZL2, PZU2, Plower, Pupper)

    dw_diter = (w_max-w_min)/num_iters

    V_min = -Pmin/2
    V_max = -Pmax/2
    velocity = np.random.uniform(V_min, V_max, (num_units, pop_size))

    p_best = np.zeros((num_units,pop_size))
    p_best_eval = np.zeros((pop_size))
    g_best = np.zeros((num_units,1))
    g_best_eval = 0

    all_fitness = np.zeros((num_iters, 1))

    Fmin = getCost(Pmin, a, b, c, d, e, Pmin)[0,0]
    Fmax = getCost(Pmax, a, b, c, d, e, Pmin)[0,0]

    
    renewables = True
    if bool_solar and bool_wind:
        NW = np.size(Pow_wind)
        NS = np.size(Pow_solar)
        N = NW + NS + num_units
        Pow_renewable = np.transpose(np.hstack((Pow_wind,Pow_solar)))
        pop_with_renewables = np.empty((N,pop_size))
        pop_with_renewables[num_units:,:] = np.tile(Pow_renewable,(1,pop_size))
        net_load = Load-np.sum(Pow_wind+Pow_solar)
    elif bool_wind:
        NW = np.size(Pow_wind)
        N = NW + num_units
        Pow_renewable = np.transpose(Pow_wind)
        pop_with_renewables = np.empty((N,pop_size))
        pop_with_renewables[num_units:,:] = np.tile(Pow_renewable,(1,pop_size))
        net_load = Load-np.sum(Pow_wind)
    elif bool_solar:
        NW = np.size(Pow_wind)
        NS = np.size(Pow_solar)
        N = NS + num_units
        Pow_renewable = np.transpose(Pow_solar)
        pop_with_renewables = np.empty((N,pop_size))
        pop_with_renewables[num_units:,:] = np.tile(Pow_renewable,(1,pop_size))
        net_load = Load-np.sum(Pow_solar) 
    else:
        net_load = Load
        N = num_units
        renewables = False
        
    # iterations
    start_time = time.time()
    for pso_iter in range(num_iters):
        # caclulate evaluation values
        if renewables:
            pop_with_renewables[0:num_units,:] = pop
            losses = getTransmissionLosses(pop_with_renewables, Bij, B0i, B00)
        else:
            losses = getTransmissionLosses(pop, Bij, B0i, B00)
        cost = getCost(pop, a, b, c, d, e, Pmin)
        fitness = getEvaluationValue(pop, cost, net_load, losses, Fmax, Fmin)
        # p_best and g_best update  
        for pop_i in range(pop_size):
            f = fitness[pop_i] 
            if f>p_best_eval[pop_i]:
                p_best_eval[pop_i] = f
                p_best[:,pop_i] = pop[:,pop_i]
        if np.max(p_best_eval)>g_best_eval:
            g_best_eval = np.max(p_best_eval)
            g_best = np.reshape(p_best[:,np.argmax(p_best_eval)],(-1,1))
        # population update
        all_fitness[pso_iter,0] = g_best_eval
        w = w_max - dw_diter*pso_iter
        rand1 = np.random.random_sample((num_units, pop_size))
        rand2 = np.random.random_sample((num_units, pop_size))
        velocity = w*velocity + c1*rand1*(p_best-pop) + c2*rand2*(g_best-pop)
        ## velocity bounding needs to be done
        pop = pop + velocity
        pop = repairOperationRange(pop, PZL1, PZU1, PZL2, PZU2, Plower, Pupper)

        progress_var.set((pso_iter+1)/num_iters*100)
        root.update_idletasks()

    best_cost = getCost(g_best, a, b, c, d, e, Pmin)[0,0]
    print("Optimal Power Production   ", g_best)
    print("Optimal Cost   ", best_cost)
    exec_time = time.time()-start_time
    print("time_taken   ", exec_time)

    output_dict = {
        "best_cost":best_cost,
        "best_power_gen":g_best,
        "time_taken":exec_time,
        "all_fitness":all_fitness,
        "Load":Load,
        "P_wind_total":np.sum(Pow_wind),
        "P_solar_total":np.sum(Pow_solar),
        }

    return output_dict




