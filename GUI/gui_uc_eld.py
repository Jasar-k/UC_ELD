from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Progressbar
from tkinter import ttk
import os
import numpy as np

from functions_excel import *  #getEldPsoExcelData,getUcPsoExcelData,getUcBgsaExcelData
from functions_eld_pso_renewable import *  #eld_pso
from functions_uc_renewable1 import *  #uc_epso,uc_ibpso,uc_bgsa


root = Tk()
root.title("UC and ELD")
root.config(background = "#DED8C9")
root.resizable(0, 0)



# --------------- main heading ---------------
label_title = Label(root, text="Unit Commitment and Economic Load Dispatch", font=('Helvetica', 18, 'bold'), bg="#DED8C9")

# --------------- widgets for uc eld choice ---------------
label_choice = Label(root, text="What do you want to perform ?", bg="#DED8C9")

var_uc_eld = IntVar()
radio_uc = Radiobutton(root, text="Unit Commitment", variable=var_uc_eld, value=1, bg="#DED8C9")
radio_eld = Radiobutton(root, text="Economic Load Dispatch", variable=var_uc_eld, value=2, bg="#DED8C9")


# --------------- widgets and functions for getting file input ---------------
def browseFiles():
    filename = filedialog.askopenfilename(initialdir = "/", title = "Select a File", filetypes = (("all files", "*.*"),("Text files", "*.txt*")))
    entry_file_explorer.delete(0, END)
    entry_file_explorer.insert(0, filename)

label_explore = Label(root, text="Select Data File", bg="#DED8C9")
entry_file_explorer = Entry(root, fg = "black", width=80, bg="#F7F5F2")
button_explore = Button(root, text = "Browse Files", command = browseFiles, bg="#C9BFA6")

# --------------- widgets for selecting algorithm ---------------

# UC
label_uc_algo = Label(root, text="Select algorithm for Unit Commitment", bg="#DED8C9")

var_uc_algo = IntVar()
radio_uc_algo1 = Radiobutton(root, text="Enhanced Particle Swarm Optimization", variable=var_uc_algo, value=1, bg="#DED8C9")
radio_uc_algo2 = Radiobutton(root, text="Improved Binary Particle Swarm Optimization (IBPSO)", variable=var_uc_algo, value=2, bg="#DED8C9")
radio_uc_algo3 = Radiobutton(root, text="Genetic Algorithm with Repair", variable=var_uc_algo, value=3, bg="#DED8C9")
radio_uc_algo4 = Radiobutton(root, text="Binary Gravitational Search Algorithm (BGSA)", variable=var_uc_algo, value=4, bg="#DED8C9")
radio_uc_algo5 = Radiobutton(root, text="Genetic Algorithm with  Penalty", variable=var_uc_algo, value=4, bg="#DED8C9")

# ELD
label_eld_algo = Label(root, text="Select algorithm for Economic Load Dispatch", bg="#DED8C9")

var_eld_algo = IntVar()
radio_eld_algo1 = Radiobutton(root, text="Particle Swarm Optimization", variable=var_eld_algo, value=1, bg="#DED8C9")
radio_eld_algo2 = Radiobutton(root, text="Genetic Algorithm", variable=var_eld_algo, value=2, bg="#DED8C9")

# --------------- status display ---------------
s = ttk.Style()
s.theme_use('clam')
s.configure("bar.Horizontal.TProgressbar", troughcolor='#F1EEE8', background='#6F6343')
progress_var = DoubleVar()
progressbar =  Progressbar(root,style="bar.Horizontal.TProgressbar", orient = HORIZONTAL, length = 750, mode = 'determinate', variable=progress_var)
progress_var.set(0)

# ----------------- Output file path -------------------

def browseFilesOutput():
    filename = filedialog.askopenfilename(initialdir = "/", title = "Select a File", filetypes = (("all files", "*.*"),("Text files", "*.txt*")))
    entry_output.delete(0, END)
    entry_output.insert(0, filename)

label_output = Label(root, text="Select Output File", bg="#DED8C9")
entry_output = Entry(root, fg = "black", width=80, bg="#F7F5F2")
button_explore_output = Button(root, text = "Browse Files", command = browseFilesOutput, bg="#C9BFA6")



# --------------- setting caclulate and edit data widget ---------------

def onClickEdit():
    # open the file path in entry in excel
    path = entry_file_explorer.get()
    # loc = path.rfind('/')
    # chdir_path = path[0:loc]
    temp = str.maketrans("/", "\\")
    path_open = path.translate(temp)
    os.startfile(path_open)

def onClickCalculate():
    path = entry_file_explorer.get()
    output_path = entry_output.get()
    uceld_choice = var_uc_eld.get()
    if uceld_choice==1:
        # unit commitment
        uc_algo_choice = var_uc_algo.get()
        
        if uc_algo_choice==1:
            # Enhanced Particle Swarm Optimization 
            input_dict = getUcPsoExcelData(path)
            output_dict = uc_epso(input_dict,progress_var,root)
            outputExcelUC(output_dict, output_path)
            # print(output_dict["best_cost"])

        elif uc_algo_choice==2:
            # Improved Binary Particle Swarm Optimization (IBPSO)
            input_dict = getUcPsoExcelData(path)
            output_dict = uc_ibpso(input_dict,progress_var,root)
            outputExcelUC(output_dict, output_path)
            # print(output_dict["best_cost"])

        elif uc_algo_choice==3:
            # Genetic Algorithm with Repair
            input_dict = getUcPsoExcelData(path)
            output_dict = uc_gar(input_dict,progress_var,root)
            outputExcelUC(output_dict, output_path)
            # outputExcelUC(output_dict, output_path)
            print(3)
        
        elif uc_algo_choice==4:
            # Binary Gravitational Search Algorithm (BGSA) 
            input_dict = getUcBgsaExcelData(path)
            output_dict = uc_gap(input_dict,progress_var,root)
            outputExcelUC(output_dict, output_path)
            # print(output_dict["best_cost"])
        
        elif uc_algo_choice==5:
            # Genetic Algorithm with Penalty
            # outputExcelUC(output_dict, output_path)
            print(5)

    elif uceld_choice==2:
        # economic load dispatch
        eld_algo_choice = var_eld_algo.get()

        if eld_algo_choice==1:
            # Particle Swarm Optimization
            input_dict = getEldPsoExcelData(path)
            output_dict = eld_pso(input_dict,progress_var,root)
            outputExcelELD(output_dict,output_path)
            
        elif eld_algo_choice==2:
            # Genetic Algorithm 
            print(2)



button_edit = Button(root, text="Edit Input Data", width=25, command=onClickEdit, bg="#C9BFA6")
label_spacer = Label(root, text="  ", bg="#DED8C9", width = 5)
button_calculate = Button(root, text="Calculate", width=25, command=onClickCalculate, bg="#C9BFA6")


# --------------- setting drop box widget for displaying graphs ---------------

# label_dropdown = Label(root, text="Select Visualization", bg="#DED8C9")

# variable = StringVar(root)
# variable.set("Thermal Cost Curve") # default value
# dropdown = OptionMenu(root, variable, "","Thermal Cost Curve")

# button_display = Button(root, text = "Display", bg="#C9BFA6")

# -------------- Takin output path -------------------

# --------------- setting minimum row and column size ---------------
col_count, row_count = root.grid_size()

for col in range(col_count):
    root.grid_columnconfigure(col, minsize=5)

for row in range(row_count):
    root.grid_rowconfigure(row, minsize=5)

root.grid_columnconfigure(0, minsize=10)


# --------------- empty rows and columns ---------------

label_empty_row1 = Label(root, text=" ", bg="#DED8C9")
label_empty_row2 = Label(root, text=" ", bg="#DED8C9")
label_empty_row3 = Label(root, text=" ", bg="#DED8C9")
label_empty_row4 = Label(root, text=" ", bg="#DED8C9")
label_empty_row5 = Label(root, text=" ", bg="#DED8C9")
label_empty_row6 = Label(root, text=" ", bg="#DED8C9")
label_empty_row7 = Label(root, text=" ", bg="#DED8C9")
label_empty_row_end = Label(root, text=" ", bg="#DED8C9")

label_empty_col1 = Label(root, text="     ", bg="#DED8C9")
label_empty_col2 = Label(root, text="     ", bg="#DED8C9")

# --------------- loctaing all widgets ---------------

label_title.grid(row=0, column=0, columnspan=10)

label_empty_row1.grid(row=1, column=0, columnspan=10)

label_choice.grid(row = 2, column = 0, columnspan=10, sticky="W")
radio_uc.grid(row = 3, column = 2, sticky="W")
radio_eld.grid(row = 4, column = 2, sticky="W")
label_empty_row2.grid(row = 5, column = 0, columnspan=10)

label_explore.grid(row = 6, column = 0)
label_empty_col1.grid(row = 6, column = 1)
entry_file_explorer.grid(row = 6, column = 2, columnspan=3)
button_explore.grid(row = 6, column = 8)
label_empty_row3.grid(row = 7, column = 0, columnspan=10)

label_uc_algo.grid(row = 8, column = 0, columnspan=10, sticky="W")
radio_uc_algo1.grid(row = 9, column = 2, sticky="W", columnspan=10)
radio_uc_algo2.grid(row = 10, column = 2, sticky="W", columnspan=10)
radio_uc_algo3.grid(row = 11, column = 2, sticky="W", columnspan=10)
radio_uc_algo4.grid(row = 12, column = 2, sticky="W", columnspan=10)
radio_uc_algo5.grid(row = 13, column = 2, sticky="W", columnspan=10)

label_eld_algo.grid(row = 17, column = 0, columnspan=10, sticky="W")
radio_eld_algo1.grid(row = 18, column = 2, sticky="W", columnspan=10)
radio_eld_algo2.grid(row = 19, column = 2, sticky="W", columnspan=10)

label_empty_row4.grid(row=20, column=0, columnspan=10)

label_output.grid(row =21, column = 0)
label_empty_col2.grid(row = 21, column = 1)
entry_output.grid(row = 21, column = 2, columnspan=3)
button_explore_output.grid(row = 21, column = 8)

label_empty_row7.grid(row = 22, column = 0, columnspan=10)

button_edit.grid(row = 25, column = 2)
label_spacer.grid(row = 25, column = 3)
button_calculate.grid(row = 25, column = 4, sticky="W")

label_empty_row5.grid(row = 26, column = 0, columnspan=10)

progressbar.grid(row = 27, column = 0, columnspan=10)

label_empty_row6.grid(row = 28, column = 0, columnspan=10)

# label_dropdown.grid(row = 29, column = 0)
# dropdown.grid(row = 29, column = 2, sticky="ew", columnspan=3)
# button_display.grid(row = 29, column = 8, sticky="ew")

label_empty_row_end.grid(row=30, column=0, columnspan=4)

root.mainloop()

