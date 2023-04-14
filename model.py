import csv  # Read and Write Files
import numpy as np  # Use Matrices
from scipy.optimize import minimize  # Optimise Function
import random
import reaction_mechanisms

def calculate_impedance_weight_factor(freq_set, iwf):
    try:
        z_weight_factor = np.array(
            [
                [1 / (j ** iwf[i]) for j in freqs]
                for i, freqs in enumerate(freq_set)
            ],
            dtype=object
        )
        return z_weight_factor
    except IndexError:
        print(
            'Index error. Make sure your iwf list is the same length as the number of EIS data files supplied.')
        exit(1)


def calculate_error(parameters):
    # Error is calculated as (z_simulated Data - Actual Data) multiplied by weighing factors

    # Calculate all error vectors using a list comprehension
    try:
        error_vectors = [
            (single_voltage_eis_simulation(potential_set[i], parameters, r_sol_set[i], q_set[i], alpha_set[i],
                                           frequency_set[i]) - z_data_set[i]) * impedance_weight_factor[i] *
            data_weighing_factor[i]
            for i in range(number_of_eis_plots_global)
        ]
    except IndexError:
        print('Index error. Make sure your sets and weighing factors lists are the same length as the number of EIS data files supplied.')
        exit(1)

    # Concatenate all error vectors into a single numpy array
    error_values = np.concatenate(error_vectors)

    # Calculates the magnitude of the vector
    normalised_error_value = np.linalg.norm(error_values)

    return normalised_error_value  # This is the value we want to be minimized


def non_linear_constraint(parameters):
    simulated_current = current_simulation(experimental_polarisation_data[:, 0], parameters)
    constraint = current_constraint - np.linalg.norm(simulated_current - experimental_polarisation_data[:, 1])
    return constraint


def single_voltage_eis_simulation(potential, parameters, r_sol, q, alpha, frequency_list):
    # Caluclates EIS based on all the input values and return the total impedance in the form Zr + jZi

    zt_values = []
    for frequency in frequency_list:
        new_zt_value = hydrogen_evolution_reaction_eis(potential,
                                                       parameters, r_sol, q, alpha, frequency)
        zt_values.append(new_zt_value)

    return np.array(zt_values)


def current_simulation(potential_list, parameters):
    # Simulates all the current values for the given set of potential values
    current_list = []

    for potential in potential_list:
        new_simulated_current = hydrogen_evolution_reaction_dc(potential, parameters)
        current_list.append(new_simulated_current)

    return np.array(current_list)


def thetass_simulation(potential_list, parameters, n, F, R, T, conc_hp, conc_h2, conc_h, mechanism_choice):
    thetass_list = []

    for potential in potential_list:
        new_simulated_thetass = thetass_dc(potential, parameters, n, F, R, T, conc_hp, conc_h2, conc_h,
                                           mechanism_choice)
        thetass_list.append(new_simulated_thetass)

    return np.array(thetass_list)


def hydrogen_evolution_reaction_dc(potential, parameters):
    variables = [n, F, R, T, conc_hp, conc_h2, conc_h]

    mechanisms = {
        '2SB': reaction_mechanisms.two_step_buffer_vt_mechanism_dc,
        '2SCB': reaction_mechanisms.two_step_cat_buffer_vh_mechanism_dc,
        '3SCB': reaction_mechanisms.three_step_cat_buffer_vht_mechanism_dc,
        '4SvtD': reaction_mechanisms.four_step_direct_vt45_mechanism_dc,
        '4SVHD': reaction_mechanisms.four_step_cat_direct_vh45_mechanism_dc,
        '5SCD': reaction_mechanisms.five_step_cat_direct_mechanism_dc,
    }

    mechanism_function = mechanisms.get(mechanism_choice_global)
    if mechanism_function:
        return mechanism_function(potential, parameters, variables)[0]
    else:
        raise ValueError(f"Invalid mechanism_choice_global: {mechanism_choice_global}")


def hydrogen_evolution_reaction_eis(potential, parameters, r_sol, Q, alpha, freq):
    variables = [n, F, R, T, conc_hp, conc_h2, conc_h]

    mechanisms = {
        '2SB': reaction_mechanisms.two_step_buffer_vt_mechanism_eis,
        '2SCB': reaction_mechanisms.two_step_cat_buffer_vh_mechanism_eis,
        '3SCB': reaction_mechanisms.three_step_cat_buffer_vht_mechanism_eis,
        '4SvtD': reaction_mechanisms.four_step_direct_vt45_mechanism_eis,
        '4SVHD': reaction_mechanisms.four_step_cat_direct_vh45_mechanism_eis,
        '5SCD': reaction_mechanisms.five_step_cat_direct_mechanism_eis,
    }

    mechanism_function = mechanisms.get(mechanism_choice_global)
    if mechanism_function:
        return mechanism_function(potential, parameters, r_sol, Q, alpha, freq, variables)
    else:
        raise ValueError(f"Invalid mechanism_choice_global: {mechanism_choice_global}")


def thetass_dc(potential, parameters, n, F, R, T, conc_hp, conc_h2, conc_h, mechanism_choice):
    variables = [n, F, R, T, conc_hp, conc_h2, conc_h]

    mechanisms = {
        '2SB': reaction_mechanisms.two_step_buffer_vt_mechanism_dc,
        '2SCB': reaction_mechanisms.two_step_cat_buffer_vh_mechanism_dc,
        '3SCB': reaction_mechanisms.three_step_cat_buffer_vht_mechanism_dc,
        '4SvtD': reaction_mechanisms.four_step_direct_vt45_mechanism_dc,
        '4SVHD': reaction_mechanisms.four_step_cat_direct_vh45_mechanism_dc,
        '5SCD': reaction_mechanisms.five_step_cat_direct_mechanism_dc,
    }

    mechanism_function = mechanisms.get(mechanism_choice)
    if mechanism_function:
        return mechanism_function(potential, parameters, variables)[1]
    else:
        raise ValueError(f"Invalid mechanism_choice: {mechanism_choice}")


def check_for_initial_parameters(random_parameters_bool):
    parameters = np.array(initial_parameters)

    try:
        parameters = load_parameters(load_from_here_global)
        print('Successfully loaded parameters from File.')
    except IOError:
        print('No initial potential only simulation parameters.  Using built in parameters.')
        if random_parameters_bool:
            print('Using random starting parameters')
            instantiate_random_parameters()
            parameters = np.array(initial_parameters)

    return parameters


def instantiate_random_parameters():
    for i in range(len(initial_parameters)):
        if i == 0:
            initial_parameters[0] = random.randint(lower_bound_parameters[0], upper_bound_parameters[0])

        if initial_parameters[i] == 0.5:
            initial_parameters[i] = random.randint(lower_bound_parameters[3] * 10, upper_bound_parameters[
                3] * 10) / 10  # returns int, need to get a value between 0 and 1.0, so multiply range by 10 then divide the int by 10
        else:
            initial_parameters[i] = random.randint(lower_bound_parameters[1], upper_bound_parameters[1])


def adapt_parameter_lists(mechanism_choice):
    remove_after_index = {
        '2SB': 10,
        '2SCB': 7,
        '3SCB': 10,
        '4SvtD': None,
        '4SVHD': None,
        '5SCD': None,
    }

    index = remove_after_index[mechanism_choice]

    if index is not None:
        del constant_names[index:]
        del initial_parameters[index:]
        del lower_bound_parameters[index:]
        del upper_bound_parameters[index:]
        del all_bounds[index:]


def create_global_variables(constants_and_parameters, constraints, load_from_here, other_data, datasets, weights,
                            concentrations, mechanism_choice, number_of_eis_plots):
    global constant_names, temp_initial_parameters, initial_parameters, lower_bound_parameters, upper_bound_parameters, all_bounds, F, R, T, n, \
        Constraints, load_from_here_global, potential_set, r_sol_set, q_set, alpha_set, frequency_set, z_data_set, \
        experimental_polarisation_data, impedance_weight_factor, data_weighing_factor, pH, conc_hp, kh_co2, conc_co2, conc_h2, \
        conc_h, mechanism_choice_global, current_constraint, number_of_eis_plots_global

    constant_names = constants_and_parameters['constant_names']
    temp_initial_parameters = constants_and_parameters['temp_initial_parameters']
    initial_parameters = constants_and_parameters['initial_parameters']
    lower_bound_parameters = constants_and_parameters['lower_bound_parameters']
    upper_bound_parameters = constants_and_parameters['upper_bound_parameters']
    all_bounds = constants_and_parameters['all_bounds']
    F = constants_and_parameters['F']
    R = constants_and_parameters['R']
    T = constants_and_parameters['T']
    n = constants_and_parameters['n']
    Constraints = constraints['Constraints']
    load_from_here_global = load_from_here
    potential_set = other_data['potential_set']
    r_sol_set = other_data['r_sol_set']
    q_set = other_data['q_set']
    alpha_set = other_data['alpha_set']
    frequency_set = datasets['frequency_set']
    z_data_set = datasets['z_data_set']
    experimental_polarisation_data = datasets['experimental_polarisation_data']
    impedance_weight_factor = weights['impedance_weight_factor']
    data_weighing_factor = weights['data_weighing_factor']
    pH = concentrations['pH']
    conc_hp = concentrations['conc_hp']
    kh_co2 = concentrations['kh_co2']
    conc_co2 = concentrations['conc_co2']
    conc_h2 = concentrations['conc_h2']
    conc_h = concentrations['conc_h']
    mechanism_choice_global = mechanism_choice
    current_constraint = constraints['current_constraint']
    number_of_eis_plots_global = number_of_eis_plots


def optimisation_program(number_of_trials, random_parameters):
    ###Optimisation Loop###
    parameter_variability = 30
    parameters = check_for_initial_parameters(random_parameters)
    initial_error = calculate_error(parameters)
    print(f'Initial Error: {initial_error}')
    min_error = initial_error

    ## Loop through number of trials, set new parameter values based on jiggled values then run the optimisation again.
    for i in range(number_of_trials):
        print(f'Trial {i + 1} of {number_of_trials}')

        Newparameters = parameters * (1 + (np.random.random_sample(
            len(initial_parameters)) - 0.5) * parameter_variability / 100)  # Add Variability

        solution = minimize(calculate_error, Newparameters, method='SLSQP', bounds=all_bounds, constraints=Constraints,
                            options={'maxiter': 1000})

        if solution.fun < min_error:
            print('New Error is Less')
            min_error = solution.fun
            parameters = solution.x
        else:
            print('New Error is More')

        print(f'current Error = {min_error}')

    # end of Function
    print(f'min_error = {min_error}')
    final_parameters = parameters
    print(f'Final parameters = {final_parameters}')

    return final_parameters


def adjustment_program(number_of_trials, deviation_percent, random_parameters):
    ###Optimisation Loop###
    parameterVariability = 20
    parameters = check_for_initial_parameters(random_parameters)
    initial_error = calculate_error(parameters)
    print(f'Initial Error: {initial_error}')
    min_error = initial_error
    deviation_amount = initial_error * deviation_percent / 100

    for i in range(number_of_trials):
        print(f'Trial {i + 1} of {number_of_trials}')

        Newparameters = parameters * (1 + (np.random.random_sample(
            len(initial_parameters)) - 0.5) * parameterVariability / 100)  # Add Vairability
        solution = minimize(calculate_error, Newparameters, method='SLSQP', bounds=all_bounds, constraints=Constraints,
                            options={'maxiter': 1000})

        print(f'minimised Error = {solution.fun}')

        if solution.fun <= (initial_error + deviation_amount):
            print('New Error is acceptable')
            min_error = solution.fun
            parameters = solution.x
        else:
            print('New Error is not acceptable')

        print(f'current Error = {min_error}')

    # end of Function
    print(f'min_error = {min_error}')
    final_parameters = parameters
    print(f'Final parameters = {final_parameters}')

    return final_parameters


def save_parameters(parameters, file_name):
    try:
        with open(file_name, mode='w') as File:  # Open file in append mode
            csv_writer = csv.writer(File, delimiter=',', lineterminator='\n')  # Prepare to write to this file
            csv_writer.writerow(parameters)  # Append the row to the file
        print('Successfully saved file under file name: ' + file_name)
    except FileNotFoundError:
        print(f'File {file_name} not found.')


def load_parameters(file_name):
    # don't put a try statement here it breaks the code
    with open(file_name, mode='r') as File:
        CSVReader = csv.reader(File)

        line_count = 0  # Resets Counter

        parameter_list = []  # Resets parameter list
        for row in CSVReader:
            for parameter in row:
                parameter_list.append(float(parameter))
            line_count = + 1
        return parameter_list
