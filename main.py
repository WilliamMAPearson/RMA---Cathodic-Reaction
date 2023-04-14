# Reaction Mechanism Analysis - Hydrogen Evolution Reaction
# William Pearson
# With the help of Dr. Ramanathan Srinivasan and Dr. Fathima Fasmin 

###Packages###
import csv  # Read and Write Files
import math  # Use Math Functions (sqrt, e)
import numpy as np  # Use Matrices
from scipy.optimize import minimize  # Optimise Function
import matplotlib.pyplot as plt  # Plot Graphs
import datetime
import random
from eis_data import EISData
import reaction_mechanisms
import model
import plot

def ReactionMechanismAnalysis(task, experiment_number, mechanism_choice, number_of_trials=1):
    variables_to_check = {
        'r_sol_set': other_data['r_sol_set'],
        'q_set': other_data['q_set'],
        'alpha_set': other_data['alpha_set'],
        'iwf': iwf,
        'data_weighing_factor': data_weighing_factor
    }
    for var_name, var in variables_to_check.items():
        if number_of_eis_plots != len(var):
            print(f'{var_name} incorrect length (should contain a value for every EIS dataset supplied).')
            exit(1)

    if task == 'optimisation':
        load_from_here = experiment_number + '/final_parameters' + mechanism_choice + '.csv'
        model.create_global_variables(constants_and_parameters, constraints, load_from_here, other_data, datasets, weights,
                                      concentrations, mechanism_choice, number_of_eis_plots)
        model.adapt_parameter_lists(mechanism_choice)
        final_parameters = model.optimisation_program(number_of_trials, False)

        model.save_parameters(final_parameters, experiment_number + '/final_parameters' + mechanism_choice + '.csv')
        plot.rma_plot(experiment_number + '/final_parameters' + mechanism_choice + '.csv', True, constants_and_parameters, constraints, other_data, datasets, weights,
                     concentrations, mechanism_choice, number_of_eis_plots)

    elif task == 'adjustment':
        load_from_here = experiment_number + '/final_parameters' + mechanism_choice + '.csv'
        model.create_global_variables(constants_and_parameters, constraints, load_from_here, other_data, datasets, weights,
                                      concentrations, mechanism_choice, number_of_eis_plots)
        model.adapt_parameter_lists(mechanism_choice)
        final_parameters = model.adjustment_program(number_of_trials, 80, False)

        model.save_parameters(final_parameters, experiment_number + '/final_parameters' + mechanism_choice + '.csv')
        plot.rma_plot(experiment_number + '/final_parameters' + mechanism_choice + '.csv', True, constants_and_parameters, constraints, other_data, datasets, weights,
                     concentrations, mechanism_choice, number_of_eis_plots)

    elif task == 'plot_thetass':
        plot.thetass_plot(experiment_number + '/final_parameters' + mechanism_choice  + '.csv', True, constants_and_parameters, constraints, other_data, datasets, weights,
                         concentrations, mechanism_choice, number_of_eis_plots)

    elif task == 'visualise_graphs':
        parameters = experiment_number + '/final_parameters' + mechanism_choice + '.csv'
        load_from_here = None
        model.create_global_variables(constants_and_parameters, constraints, load_from_here, other_data, datasets,
                                      weights,
                                      concentrations, mechanism_choice, number_of_eis_plots)
        plot.rma_plot(parameters, True, constants_and_parameters, constraints, other_data, datasets, weights,
                      concentrations, mechanism_choice, number_of_eis_plots)

    elif task == 'save_data':
        use_final_parameters = experiment_number + '/final_parameters' + mechanism_choice + '.csv'
        load_from_here = None
        model.create_global_variables(constants_and_parameters, constraints, load_from_here, other_data, datasets,
                                      weights,
                                      concentrations, mechanism_choice, number_of_eis_plots)
        plot.rma_plot(use_final_parameters, True, constants_and_parameters, constraints, other_data, datasets,
                      weights,
                      concentrations, mechanism_choice, number_of_eis_plots)
        print('Use These Plots? .... y/n')
        user_input = input()
        if user_input == 'y':
            plot.save_plot_values(use_final_parameters)

    else:
        print(
            "Invalid task type. Please choose 'optimisation', 'adjustment', 'plot_thetass', 'visualise_graphs', or 'save_data'.")

constants_and_parameters = {
    "constant_names": reaction_mechanisms.constant_names,
    "temp_initial_parameters": reaction_mechanisms.temp_initial_parameters,
    "initial_parameters": reaction_mechanisms.initial_parameters,
    "lower_bound_parameters": reaction_mechanisms.lower_bound_parameters,
    "upper_bound_parameters": reaction_mechanisms.upper_bound_parameters,
    "all_bounds": reaction_mechanisms.all_bounds,
    "F": 96485,
    "R": 8.3145,
    "T": 353,
    "n": 1
}

mechanisms = ['2SB', '2SCB', '3SCB', '4SvtD', '4SVHD', '5SCD']

# Define experiment_number
experiment_number = "RMA Experiment Data"

# Give the location of the EIS data files
eis_files = [
    f"{experiment_number}/HER100mVEIS.txt",
    f"{experiment_number}/HER200mVEIS.txt",
    f"{experiment_number}/HER300mVEIS.txt",
    f"{experiment_number}/HER400mVEIS.txt",
    f"{experiment_number}/HERPolarisation.txt"
]

eis_data = EISData(eis_files)
frequency_set = eis_data.frequency_datasets
experimental_polarisation_data = eis_data.potential_current_data
z_data_set = eis_data.extracted_z_data
number_of_eis_plots = len(eis_data.eis_data)

datasets = {
    'frequency_set': frequency_set,
    'z_data_set': z_data_set,
    'experimental_polarisation_data': experimental_polarisation_data
}

###Data From Other Programs###
# one list item for each eis plot
other_data = {
    "potential_set": [-float(file_name.split('HER')[1].split('mV')[0]) / 1000 for file_name in eis_files if 'mV' in
                      file_name],  # get potential set from filenames
    "r_sol_set": [0, 0, 0, 0],  # solution Resistance Gained from CPP and EEC
    "q_set": [9.8296E-11, 9.82967E-11, 9.8296E-11, 9.8296E-11],  # Effective Capacitance Gained from CPP and EEC
    "alpha_set": [1, 1, 1, 1]  # Capacitance alpha Value Gained from CPP and EEC
}

###Species Concentration###
concentrations = {
    "pH": 3.96,  # pH value
    "conc_hp": 10 ** (-3.96),  # Concentration of H+ ions from pH value (Mol.dm-3)
    "kh_co2": 29.41,  # Equilibrium constant for CO2 (atm/M)
    "conc_co2": 1 / 29.41,  # Concentration of dissolved CO2 (M)
    "conc_h2": 1.7e-3 * (1 / 29.41) * 1000,  # Concentration of H2 ions from CO2 concentration (mM)
    "conc_h": ((2e-4 * (1.7e-3 * (1 / 29.41) * 1000)) / (10 ** (-3.96))) * 1000  # Concentration of H+ ions from H2
    # and H+ concentrations (mM)
}

###Constraints###
constraints = {
    'current_constraint': 0.0001,
    'Constraints': {'type': 'ineq', 'fun': model.non_linear_constraint}
}

###Weighing Factors###
# Impedance weighing Factors, frequency is in the denominator, and is raised to the power given here
iwf = [1, 1, 1, 1] # one for each eis plot
impedance_weight_factor = model.calculate_impedance_weight_factor(datasets['frequency_set'], iwf)

# Weighing Factors for Edc data
data_weighing_factor = np.array([1, 1, 1, 1]) # one for each eis plot
weights = {
    'impedance_weight_factor': impedance_weight_factor,
    'data_weighing_factor': data_weighing_factor
}

mechanism_choice = 1 # 0: 2SB, 1: 2SCB, 2: 3SCB, 3: 4SVTD, 4: 4SVHD, 5: 5SCD

# Call ReactionMechanismAnalysis with the desired task (Choose 'optimisation', 'adjustment', 'plot_thetass', 'visualise_graphs',
# or 'save_data') and number of trials (if applicable).
ReactionMechanismAnalysis('optimisation', experiment_number, mechanisms[mechanism_choice], number_of_trials=1)
