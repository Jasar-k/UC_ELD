import xlwings
from pathlib import Path
import numpy as np


def colnum_string(n):
    # convertes column number to string
    string = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        string = chr(65 + remainder) + string
    return string


def getEldPsoExcelData(path):

    wb = xlwings.Book(path)
    all_sheets = wb.sheets

    ws_general = all_sheets["General"]
    ws_algo_params = all_sheets["Algorithm Parameters"]
    ws_thermal = all_sheets["Thermal Units"]
    ws_wind = all_sheets["Wind"]
    ws_pv = all_sheets["PV"]
    
    num_thermal_units = ws_general.range('B4').value
    num_pv_units = ws_general.range('B6').value
    num_wind_units = ws_general.range('B7').value

    total_units = num_thermal_units+num_pv_units+num_wind_units

    num_time_periods = ws_general.range('B9').value # always =1

    c1 = ws_algo_params.range('B2').value
    c2 = ws_algo_params.range('B3').value
    w_min = ws_algo_params.range('B4').value
    w_max = ws_algo_params.range('B5').value
    max_iters = ws_algo_params.range('B7').value
    pop_size = ws_algo_params.range('B6').value
    
    input_dict={
        "bool_wind":False,
        "bool_solar":False
    }
   
    if num_wind_units:
        end_col_num = int(2+num_wind_units-1)
        end_col = colnum_string(end_col_num)
        Pwn = ws_wind.range('B4:'+ end_col +str(4)).options(ndim=1).value
        v1 = ws_wind.range('B5:'+ end_col +str(5)).options(ndim=1).value
        v2 = ws_wind.range('B6:'+ end_col +str(6)).options(ndim=1).value
        v3 = ws_wind.range('B7:'+ end_col +str(7)).options(ndim=1).value
        end_row = int(13+num_wind_units-1)
        forecast_wind_speed = ws_wind.range('B13:B'+str(end_row)).options(ndim=1).value
        input_dict.update({
            "bool_wind":True,
            "v1":v1,
            "v2":v2,
            "v3":v3,
            "Pwn":Pwn,
            "forecast_wind_speed":forecast_wind_speed
        })
    
    if num_pv_units:
        end_col_num = int(2+num_pv_units-1)
        end_col = colnum_string(end_col_num)
        Psn = ws_pv.range('B4:'+ end_col +str(4)).options(ndim=1).value
        Gstd = ws_pv.range('B5:'+ end_col +str(5)).options(ndim=1).value
        Rc = ws_pv.range('B6:'+ end_col +str(6)).options(ndim=1).value
        end_row = int(13+num_wind_units-1)
        Gt = ws_pv.range('B13:B'+str(end_row)).options(ndim=1).value
        
        input_dict.update({
            "bool_solar":True,
            "Psn":Psn,
            "Gt":Gt,
            "Gstd":Gstd,
            "Rc":Rc
        })

    if num_thermal_units:
        start_col_num = 2
        end_col_num = int(start_col_num + num_thermal_units - 1)
        start_col = colnum_string(start_col_num)
        end_col = colnum_string(end_col_num)
        
        Pnow = np.array(ws_thermal.range(str(start_col)+'4:'+str(end_col)+'4').value).reshape((-1,1))
        
        Pmin = np.array(ws_thermal.range(str(start_col)+'7:'+str(end_col)+'7').value).reshape((-1,1))
        Pmax = np.array(ws_thermal.range(str(start_col)+'8:'+str(end_col)+'8').value).reshape((-1,1))
        
        RU = np.array(ws_thermal.range(str(start_col)+'10:'+str(end_col)+'10').value).reshape((-1,1))
        RD = np.array(ws_thermal.range(str(start_col)+'11:'+str(end_col)+'11').value).reshape((-1,1))
        
        PZL1 = np.array(ws_thermal.range(str(start_col)+'17:'+str(end_col)+'17').value).reshape((-1,1))
        PZU1 = np.array(ws_thermal.range(str(start_col)+'18:'+str(end_col)+'18').value).reshape((-1,1))
        PZL2 = np.array(ws_thermal.range(str(start_col)+'19:'+str(end_col)+'19').value).reshape((-1,1))
        PZU2 = np.array(ws_thermal.range(str(start_col)+'20:'+str(end_col)+'20').value).reshape((-1,1))
        
        a = np.array(ws_thermal.range(str(start_col)+'23:'+str(end_col)+'23').value).reshape((-1,1))
        b = np.array(ws_thermal.range(str(start_col)+'24:'+str(end_col)+'24').value).reshape((-1,1))
        c = np.array(ws_thermal.range(str(start_col)+'25:'+str(end_col)+'25').value).reshape((-1,1))
        d = np.array(ws_thermal.range(str(start_col)+'26:'+str(end_col)+'26').value).reshape((-1,1))
        e = np.array(ws_thermal.range(str(start_col)+'27:'+str(end_col)+'27').value).reshape((-1,1))

        end_col_num = int(start_col_num + total_units - 1)
        end_col = colnum_string(end_col_num)
        
        B00 = ws_thermal.range('B30').value
        B0i = np.array(ws_thermal.range(str(start_col)+'31:'+str(end_col)+'31').value)
        end_row = int(32+total_units-1)
        Bij = np.array(ws_thermal.range(str(start_col)+'32:'+str(end_col)+str(end_row)).options(ndim=2).value)

        Load = ws_general.range('B11').value

        input_dict.update({
            "Pnow":Pnow,
            "Pmin":Pmin,
            "Pmax":Pmax,
            "RU":RU,
            "RD":RD,
            "PZL1":PZL1,
            "PZU1":PZU1,
            "PZL2":PZL2,
            "PZU2":PZU2,
            "a":a,
            "b":b,
            "c":c,
            "d":d,
            "e":e,
            "B00":B00,
            "B0i":B0i,
            "Bij":Bij,
            "c1":c1,
            "c2":c2,
            "w_min":w_min,
            "w_max":w_max,
            "max_iters":max_iters,
            "pop_size":pop_size,
            "Load":Load
        })

    return input_dict


def getUcPsoExcelData(path):

    wb = xlwings.Book(path)
    all_sheets = wb.sheets

    ws_general = all_sheets["General"]
    ws_algo_params = all_sheets["Algorithm Parameters"]
    ws_thermal = all_sheets["Thermal Units"]
    ws_wind = all_sheets["Wind"]
    ws_pv = all_sheets["PV"]
    
    num_thermal_units = ws_general.range('B4').value
    num_pv_units = ws_general.range('B6').value
    num_wind_units = ws_general.range('B7').value

    total_units = num_thermal_units+num_pv_units+num_wind_units

    num_time_periods = ws_general.range('B9').value 
    
    end_col_number = int(2+num_time_periods-1)
    end_col = colnum_string(end_col_number)
    demands = np.array(ws_general.range('B11:'+end_col+'11').options(ndim=1).value)

    c1 = ws_algo_params.range('B2').value
    c2 = ws_algo_params.range('B3').value
    w_min = ws_algo_params.range('B4').value
    w_max = ws_algo_params.range('B5').value
    max_iters = ws_algo_params.range('B7').value
    pop_size = ws_algo_params.range('B6').value
    
    input_dict={
        "c1":c1,
        "c2":c2,
        "w_min":w_min,
        "w_max":w_max,
        "max_iters":max_iters,
        "pop_size":pop_size,
        "Load":demands,
        "bool_wind":False,
        "bool_solar":False,
        }
   
    if num_wind_units:
        end_col_num = int(2+num_wind_units-1)
        end_col = colnum_string(end_col_num)
        Pwn = np.array(ws_wind.range('B4:'+ end_col +str(4)).options(ndim=1).value)
        v1 = np.array(ws_wind.range('B5:'+ end_col +str(5)).options(ndim=1).value)
        v2 = np.array(ws_wind.range('B6:'+ end_col +str(6)).options(ndim=1).value)
        v3 = np.array(ws_wind.range('B7:'+ end_col +str(7)).options(ndim=1).value)
        end_row = int(13+num_wind_units-1)
        end_col_num = int(2+num_time_periods-1)
        end_col = colnum_string(end_col_num)
        forecast_wind_speed = np.array(ws_wind.range('B13:'+end_col+str(end_row)).options(ndim=2).value)
        input_dict.update({
            "bool_wind":True,
            "v1":v1,
            "v2":v2,
            "v3":v3,
            "Pwn":Pwn,
            "forecast_wind_speed":forecast_wind_speed
        })
    
    if num_pv_units:
        end_col_num = int(2+num_pv_units-1)
        end_col = colnum_string(end_col_num)
        Psn = np.array(ws_pv.range('B4:'+ end_col +str(4)).options(ndim=1).value)
        Gstd = np.array(ws_pv.range('B5:'+ end_col +str(5)).options(ndim=1).value)
        Rc = np.array(ws_pv.range('B6:'+ end_col +str(6)).options(ndim=1).value)
        end_row = int(13+num_wind_units-1)
        end_col_num = int(2+num_time_periods-1)
        end_col = colnum_string(end_col_num)
        Gt = np.array(ws_pv.range('B13:'+end_col+str(end_row)).options(ndim=2).value)
        
        input_dict.update({
            "bool_solar":True,
            "Psn":Psn,
            "Gt":Gt,
            "Gstd":Gstd,
            "Rc":Rc
        })

    if num_thermal_units:
        start_col_num = 2
        end_col_num = int(start_col_num + num_thermal_units - 1)
        start_col = colnum_string(start_col_num)
        end_col = colnum_string(end_col_num)
        
        # Pnow = np.array(ws_thermal.range(str(start_col)+'4:'+str(end_col)+'4').value).reshape((-1,1))

        run_time = np.array(ws_thermal.range(str(start_col)+'5:'+str(end_col)+'5').value)#.reshape((-1,1))
        
        Pmin = np.array(ws_thermal.range(str(start_col)+'7:'+str(end_col)+'7').value)#.reshape((-1,1))
        Pmax = np.array(ws_thermal.range(str(start_col)+'8:'+str(end_col)+'8').value)#.reshape((-1,1))
        
        # RU = np.array(ws_thermal.range(str(start_col)+'10:'+str(end_col)+'10').value).reshape((-1,1))
        # RD = np.array(ws_thermal.range(str(start_col)+'11:'+str(end_col)+'11').value).reshape((-1,1))
        
        # PZL1 = np.array(ws_thermal.range(str(start_col)+'22:'+str(end_col)+'22').value).reshape((-1,1))
        # PZU1 = np.array(ws_thermal.range(str(start_col)+'23:'+str(end_col)+'23').value).reshape((-1,1))
        # PZL2 = np.array(ws_thermal.range(str(start_col)+'24:'+str(end_col)+'24').value).reshape((-1,1))
        # PZU2 = np.array(ws_thermal.range(str(start_col)+'25:'+str(end_col)+'25').value).reshape((-1,1))

        cost_hot_start = np.array(ws_thermal.range(str(start_col)+'13:'+str(end_col)+'13').value)#.reshape((-1,1))
        cost_cold_start = np.array(ws_thermal.range(str(start_col)+'14:'+str(end_col)+'14').value)#.reshape((-1,1))
        cold_start_hrs = np.array(ws_thermal.range(str(start_col)+'15:'+str(end_col)+'15').value)#.reshape((-1,1))

        min_up_time = np.array(ws_thermal.range(str(start_col)+'18:'+str(end_col)+'18').value)#.reshape((-1,1))
        min_down_time = np.array(ws_thermal.range(str(start_col)+'19:'+str(end_col)+'19').value)#.reshape((-1,1))
        
        a = np.array(ws_thermal.range(str(start_col)+'28:'+str(end_col)+'28').value)#.reshape((-1,1))
        b = np.array(ws_thermal.range(str(start_col)+'29:'+str(end_col)+'29').value)#.reshape((-1,1))
        c = np.array(ws_thermal.range(str(start_col)+'30:'+str(end_col)+'30').value)#.reshape((-1,1))
        # d = np.array(ws_thermal.range(str(start_col)+'31:'+str(end_col)+'31').value).reshape((-1,1))
        # e = np.array(ws_thermal.range(str(start_col)+'32:'+str(end_col)+'32').value).reshape((-1,1))

        # end_col_num = int(start_col_num + total_units - 1)
        # end_col = colnum_string(end_col_num)
        
        # B00 = ws_thermal.range('B30').value
        # B0i = np.array(ws_thermal.range(str(start_col)+'31:'+str(end_col)+'31').value)
        # end_row = int(32+total_units-1)
        # Bij = np.array(ws_thermal.range(str(start_col)+'32:'+str(end_col)+str(end_row)).options(ndim=2).value)

        input_dict.update({
            # "Pnow":Pnow,
            "initial_run_time":run_time,
            "Pmin":Pmin,
            "Pmax":Pmax,
            # "RU":RU,
            # "RD":RD,
            # "PZL1":PZL1,
            # "PZU1":PZU1,
            # "PZL2":PZL2,
            # "PZU2":PZU2,
            "cost_hot_start":cost_hot_start,
            "cost_cold_start":cost_cold_start,
            "cold_start_hrs":cold_start_hrs,
            "min_up_time":min_up_time,
            "min_down_time":min_down_time,
            "a":a,
            "b":b,
            "c":c,
            # "d":d,
            # "e":e,
            # "B00":B00,
            # "B0i":B0i,
            # "Bij":Bij
        })

    return input_dict


def getUcBgsaExcelData(path):

    wb = xlwings.Book(path)
    all_sheets = wb.sheets

    ws_general = all_sheets["General"]
    ws_algo_params = all_sheets["Algorithm Parameters"]
    ws_thermal = all_sheets["Thermal Units"]
    ws_wind = all_sheets["Wind"]
    ws_pv = all_sheets["PV"]
    
    num_thermal_units = ws_general.range('B4').value
    num_pv_units = ws_general.range('B6').value
    num_wind_units = ws_general.range('B7').value

    total_units = num_thermal_units+num_pv_units+num_wind_units

    num_time_periods = ws_general.range('B9').value 
    
    end_col_number = int(2+num_time_periods-1)
    end_col = colnum_string(end_col_number)
    demands = np.array(ws_general.range('B11:'+end_col+'11').options(ndim=1).value)

    G0 = ws_algo_params.range('I12').value
    epsilon = ws_algo_params.range('I13').value
    K_best = ws_algo_params.range('I14').value
    max_iters = ws_algo_params.range('I16').value
    pop_size = ws_algo_params.range('I15').value
    
    input_dict={
        "G0":G0,
        "epsilon":epsilon,
        "K_best":K_best,
        "max_iters":max_iters,
        "pop_size":pop_size,
        "Load":demands,
        "bool_wind":False,
        "bool_solar":False,
        }
   
    if num_wind_units:
        end_col_num = int(2+num_wind_units-1)
        end_col = colnum_string(end_col_num)
        Pwn = np.array(ws_wind.range('B4:'+ end_col +str(4)).options(ndim=1).value)
        v1 = np.array(ws_wind.range('B5:'+ end_col +str(5)).options(ndim=1).value)
        v2 = np.array(ws_wind.range('B6:'+ end_col +str(6)).options(ndim=1).value)
        v3 = np.array(ws_wind.range('B7:'+ end_col +str(7)).options(ndim=1).value)
        end_row = int(13+num_wind_units-1)
        end_col_num = int(2+num_time_periods-1)
        end_col = colnum_string(end_col_num)
        forecast_wind_speed = np.array(ws_wind.range('B13:'+end_col+str(end_row)).options(ndim=2).value)
        input_dict.update({
            "bool_wind":True,
            "v1":v1,
            "v2":v2,
            "v3":v3,
            "Pwn":Pwn,
            "forecast_wind_speed":forecast_wind_speed
        })
    
    if num_pv_units:
        end_col_num = int(2+num_pv_units-1)
        end_col = colnum_string(end_col_num)
        Psn = np.array(ws_pv.range('B4:'+ end_col +str(4)).options(ndim=1).value)
        Gstd = np.array(ws_pv.range('B5:'+ end_col +str(5)).options(ndim=1).value)
        Rc = np.array(ws_pv.range('B6:'+ end_col +str(6)).options(ndim=1).value)
        end_row = int(13+num_wind_units-1)
        end_col_num = int(2+num_time_periods-1)
        end_col = colnum_string(end_col_num)
        Gt = np.array(ws_pv.range('B13:'+end_col+str(end_row)).options(ndim=2).value)
        
        input_dict.update({
            "bool_solar":True,
            "Psn":Psn,
            "Gt":Gt,
            "Gstd":Gstd,
            "Rc":Rc
        })

    if num_thermal_units:
        start_col_num = 2
        end_col_num = int(start_col_num + num_thermal_units - 1)
        start_col = colnum_string(start_col_num)
        end_col = colnum_string(end_col_num)
        
        # Pnow = np.array(ws_thermal.range(str(start_col)+'4:'+str(end_col)+'4').value).reshape((-1,1))

        run_time = np.array(ws_thermal.range(str(start_col)+'5:'+str(end_col)+'5').value)#.reshape((-1,1))
        
        Pmin = np.array(ws_thermal.range(str(start_col)+'7:'+str(end_col)+'7').value)#.reshape((-1,1))
        Pmax = np.array(ws_thermal.range(str(start_col)+'8:'+str(end_col)+'8').value)#.reshape((-1,1))
        
        # RU = np.array(ws_thermal.range(str(start_col)+'10:'+str(end_col)+'10').value).reshape((-1,1))
        # RD = np.array(ws_thermal.range(str(start_col)+'11:'+str(end_col)+'11').value).reshape((-1,1))
        
        # PZL1 = np.array(ws_thermal.range(str(start_col)+'22:'+str(end_col)+'22').value).reshape((-1,1))
        # PZU1 = np.array(ws_thermal.range(str(start_col)+'23:'+str(end_col)+'23').value).reshape((-1,1))
        # PZL2 = np.array(ws_thermal.range(str(start_col)+'24:'+str(end_col)+'24').value).reshape((-1,1))
        # PZU2 = np.array(ws_thermal.range(str(start_col)+'25:'+str(end_col)+'25').value).reshape((-1,1))

        cost_hot_start = np.array(ws_thermal.range(str(start_col)+'13:'+str(end_col)+'13').value)#.reshape((-1,1))
        cost_cold_start = np.array(ws_thermal.range(str(start_col)+'14:'+str(end_col)+'14').value)#.reshape((-1,1))
        cold_start_hrs = np.array(ws_thermal.range(str(start_col)+'15:'+str(end_col)+'15').value)#.reshape((-1,1))

        min_up_time = np.array(ws_thermal.range(str(start_col)+'18:'+str(end_col)+'18').value)#.reshape((-1,1))
        min_down_time = np.array(ws_thermal.range(str(start_col)+'19:'+str(end_col)+'19').value)#.reshape((-1,1))
        
        a = np.array(ws_thermal.range(str(start_col)+'28:'+str(end_col)+'28').value)#.reshape((-1,1))
        b = np.array(ws_thermal.range(str(start_col)+'29:'+str(end_col)+'29').value)#.reshape((-1,1))
        c = np.array(ws_thermal.range(str(start_col)+'30:'+str(end_col)+'30').value)#.reshape((-1,1))
        # d = np.array(ws_thermal.range(str(start_col)+'31:'+str(end_col)+'31').value).reshape((-1,1))
        # e = np.array(ws_thermal.range(str(start_col)+'32:'+str(end_col)+'32').value).reshape((-1,1))

        # end_col_num = int(start_col_num + total_units - 1)
        # end_col = colnum_string(end_col_num)
        
        # B00 = ws_thermal.range('B30').value
        # B0i = np.array(ws_thermal.range(str(start_col)+'31:'+str(end_col)+'31').value)
        # end_row = int(32+total_units-1)
        # Bij = np.array(ws_thermal.range(str(start_col)+'32:'+str(end_col)+str(end_row)).options(ndim=2).value)

        input_dict.update({
            # "Pnow":Pnow,
            "initial_run_time":run_time,
            "Pmin":Pmin,
            "Pmax":Pmax,
            # "RU":RU,
            # "RD":RD,
            # "PZL1":PZL1,
            # "PZU1":PZU1,
            # "PZL2":PZL2,
            # "PZU2":PZU2,
            "cost_hot_start":cost_hot_start,
            "cost_cold_start":cost_cold_start,
            "cold_start_hrs":cold_start_hrs,
            "min_up_time":min_up_time,
            "min_down_time":min_down_time,
            "a":a,
            "b":b,
            "c":c,
            # "d":d,
            # "e":e,
            # "B00":B00,
            # "B0i":B0i,
            # "Bij":Bij
        })

    return input_dict


def outputExcelUC(output_dict,output_path):
    uc = output_dict["best_uc"]
    cost = output_dict["best_total_cost"]
    powers = output_dict["best_power_gen"]
    time_taken = output_dict["time_taken"]

    NT,T = np.shape(uc)

    wb = xlwings.Book(output_path)
    all_sheets = wb.sheets

    ws_uc = all_sheets["UC schedule"]
    ws_powers = all_sheets["Power Outputs"]
    ws_summary = all_sheets["Summary"]
    
    end_col_num = int(2+T-1)
    end_col = colnum_string(end_col_num)
    end_row = int(2+NT-1)
    ws_uc.range('B2:'+end_col+str(end_row)).options(ndim=2).value = uc
    ws_powers.range('B2:'+end_col+str(end_row)).options(ndim=2).value = powers
    ws_summary.range('B1').value = cost
    ws_summary.range('B2').value = time_taken
    
    print("You can check the output at " + output_path)


def outputExcelELD(output_dict,output_path):
    wb = xlwings.Book(output_path)
    all_sheets = wb.sheets

    ws_summary = all_sheets["Summary"]
    
    thermal_power_gen = output_dict["best_power_gen"]
    thermal_pow = np.sum(thermal_power_gen) + output_dict["P_solar_total"]
    Ptot = thermal_pow + output_dict["P_wind_total"]
    N = np.size(thermal_power_gen)
    end_col_num = int(2+N-1)
    ws_summary.range('B1').value = output_dict["best_cost"]
    ws_summary.range('B2:'+colnum_string(end_col_num)+str(2)).options(ndim=2).value = np.transpose(thermal_power_gen)
    ws_summary.range('B3').value = output_dict["time_taken"]
    ws_summary.range('B4').value = output_dict["Load"]
    ws_summary.range('B5').value = Ptot
    ws_summary.range('B6').value = output_dict["P_wind_total"]
    ws_summary.range('B7').value = output_dict["P_solar_total"]
    ws_summary.range('B8').value = thermal_pow
    ws_summary.range('B9').value = Ptot-output_dict["Load"]

    print("You can see the output at " + output_path)

    # end_col_num = int(2+T-1)
    # end_col = colnum_string(end_col_num)
    # end_row = int(2+NT-1)
    # ws_uc.range('B2:'+end_col+str(end_row)).options(ndim=2).value = uc
    # ws_powers.range('B2:'+end_col+str(end_row)).options(ndim=2).value = powers
    # ws_summary.range('B1').value = cost
    # ws_summary.range('B2').value = time_taken
    
    
   



# C:/Users/Akshat jain/Desktop/UC and ELD/Input Data Files/UC_11units_including_renewables.xlsx
# C:/Users/Akshat jain/Desktop/Output UC.xlsx