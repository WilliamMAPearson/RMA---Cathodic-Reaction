import csv  # Read and Write Files
import numpy as np  # Use Matrices
import matplotlib.pyplot as plt  # Plot Graphs
import model
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import os
from scipy.signal import argrelextrema
from adjustText import adjust_text


def create_global_variables(constants_and_parameters, constraints, other_data, datasets, weights, concentrations,
                            mechanism_choice, number_of_eis_plots):
    global constant_names, temp_initial_parameters, initial_parameters, lower_bound_parameters, upper_bound_parameters, all_bounds, F, R, T, n, \
        Constraints, potential_set, r_solSet, q_set, alpha_set, frequency_set, z_data_set, \
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
    potential_set = other_data['potential_set']
    r_solSet = other_data['r_sol_set']
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


def thetass_plot(file_name, save_data, constants_and_parameters, constraints, other_data, datasets, weights,
                                  concentrations, mechanism_choice, number_of_eis_plots):
    create_global_variables(constants_and_parameters, constraints, other_data, datasets, weights,
                                  concentrations, mechanism_choice, number_of_eis_plots)
    final_parameters = model.load_parameters(file_name)

    exp_plot = experimental_polarisation_data[:,0]
    sim_thetass = model.thetass_simulation(experimental_polarisation_data[:,0], final_parameters, n, F, R, T, conc_hp, conc_h2, conc_h, mechanism_choice)

    print(exp_plot)
    print(sim_thetass)

    if save_data:
        file_name = 'thetass'
        with open(file_name, mode='w') as File:  # Open file in append mode
            csv_writer = csv.writer(File, delimiter=',', lineterminator='\n')  # Prepare to write to this file
            csv_writer.writerow(sim_thetass)  # Append the row to the file
        print('Succesfully Saved File Under File Name: ' + file_name)

    fig, (ax1) = plt.subplots(1, 1)

    ax1.plot(exp_plot, sim_thetass, 'tab:blue')
    ax1.set_title('potential (V) vs. Theta_SS')
    ax1.set(xlabel='potential (V)', ylabel='Theta_SS')

    ax1.plot(exp_plot, sim_thetass, 'tab:blue')
    ax1.set(xlabel='potential (V)', ylabel='Theta_SS')

    plt.show()


def save_plot_values(file_name):
    final_parameters_local = model.load_parameters(file_name)

    exp_plot = experimental_polarisation_data[:, 0]
    exp_curr = experimental_polarisation_data[:, 1]
    sim_curr = model.current_simulation(experimental_polarisation_data[:, 0], final_parameters_local)

    # Remove the .csv extension from the file_name
    file_name_without_extension, _ = os.path.splitext(file_name)

    # IE Plot
    potential_current_filename = f"{file_name_without_extension}_potential_current_plot_values.csv"
    with open(potential_current_filename, "w") as csv_writer:
        for i in range(len(exp_plot)):
            csv_writer.write(f"{exp_plot[i]}\t{exp_curr[i]}\t{sim_curr[i]}\n")

    # EIS Plots
    for i in range(len(potential_set)):
        Zt = model.single_voltage_eis_simulation(potential_set[i], final_parameters_local, r_solSet[i], q_set[i],
                                           alpha_set[i],
                                           frequency_set[i])

        eis_plot_file_name = f"{file_name_without_extension}_eis_plot_{potential_set[i]}.csv"
        with open(eis_plot_file_name, "w") as csv_writer:
            for m in range(len(Zt)):
                csv_writer.write(f"{frequency_set[i][m]}\t{Zt[m].real}\t{-Zt[m].imag}\n")


def rma_plot(file_name, override_frequency_list_bool, constants_and_parameters, constraints, other_data, datasets,
             weights, concentrations, mechanism_choice, number_of_eis_plots):
    create_global_variables(constants_and_parameters, constraints, other_data, datasets, weights,
                            concentrations, mechanism_choice, number_of_eis_plots)
    try:
        final_parameters = model.load_parameters(file_name)
        exp_plot = experimental_polarisation_data[:, 0]
        exp_curr = experimental_polarisation_data[:, 1]
        sim_pot = experimental_polarisation_data[:, 0]
        sim_curr = model.current_simulation(experimental_polarisation_data[:, 0], final_parameters)

        # Calculate the number of rows and columns in the grid
        grid_cols = 3
        grid_rows = ((number_of_eis_plots + 1) // grid_cols) + 1

        fig = plt.figure(figsize=(15, 8 * (grid_rows / 2)))
        gs = GridSpec(grid_rows, grid_cols)

        ax1 = plt.subplot(gs[0, :2])

        sns.lineplot(x=exp_plot, y=exp_curr, ax=ax1, color='tab:blue')
        sns.lineplot(x=sim_pot, y=sim_curr, ax=ax1, color='tab:orange')
        ax1.set_title('Current (A/cm2) vs. potential (V)')
        ax1.set(xlabel='Potential (V)', ylabel='Current (A/cm2)')
        legend_elements = [
            Line2D([0], [0], color='tab:blue', lw=2, label='Experimental'),
            Line2D([0], [0], color='tab:orange', lw=2, label='Simulation')
        ]
        ax1.legend(handles=legend_elements)

        # Create the impedance plot subplots
        impedance_axes = []
        for i in range(number_of_eis_plots):
            ax = plt.subplot(gs[(i + 2) // grid_cols, (i + 2) % grid_cols])
            impedance_axes.append(ax)

        cached_freq_set = []
        for freq_set in frequency_set:
            cached_freq_set.append(freq_set)

        if override_frequency_list_bool:
            new_frequency_set = frequency_set
            override_frequency_set = np.logspace(np.log10(frequency_set[0][0]), np.log10(frequency_set[0][-1]),
                                                 num=1000)
            for i in range(len(frequency_set)):
                new_frequency_set[i] = override_frequency_set

        for i, ax in enumerate(impedance_axes):
            Zt = model.single_voltage_eis_simulation(other_data['potential_set'][i], final_parameters, other_data[
                'r_sol_set'][i], other_data['q_set'][i], other_data['alpha_set'][i],
                                                     new_frequency_set[i])
            # find the lowest three points of the Z data
            Z_real = np.real(z_data_set[i])
            Z_imag = np.imag(z_data_set[i])
            Z_sim_real = np.real(Zt)
            Z_sim_imag = np.imag(Zt)

            def custom_comparator(a, b, tol=1e-5):
                return np.greater_equal(a, b - tol)

            local_maxima_indices = argrelextrema(Z_imag, custom_comparator)
            local_maxima_indices_sim = argrelextrema(Z_sim_imag, custom_comparator)
            highest_points_with_indices = sorted(
                zip(Z_real[local_maxima_indices], Z_imag[local_maxima_indices], local_maxima_indices[0]),
                key=lambda x: x[1], reverse=True)[:3]
            highest_points_with_indices_sim = sorted(
                zip(Z_sim_real[local_maxima_indices_sim], Z_sim_imag[local_maxima_indices_sim],
                    local_maxima_indices_sim[0]),
                key=lambda x: x[1], reverse=True)[:3]
            frequencies = frequency_set[i]
            frequencies_sim = new_frequency_set[i]
            highest_frequencies = [frequencies[index] for _, _, index in highest_points_with_indices]
            highest_frequencies_sim = [frequencies_sim[index] for _, _, index in highest_points_with_indices_sim]
            highest_points = [(z_real, z_imag) for z_real, z_imag, _ in highest_points_with_indices]
            highest_points_sim = [(z_real, z_imag) for z_real, z_imag, _ in highest_points_with_indices_sim]
            sns.scatterplot(x=np.real(z_data_set[i]), y=np.imag(z_data_set[i]), ax=ax, color='tab:blue', marker='s')
            sns.lineplot(x=np.array(Zt.real), y=np.array(Zt.imag), ax=ax, color='tab:orange', sort=False)

            texts = []  # A list to store the text labels for the current subplot

            # annotate highest three points of experimental Z data
            for point, freq in zip(highest_points, highest_frequencies):
                text = ax.annotate(
                    f"{freq:.2e} Hz",
                    xy=(point[0], point[1]),
                    xytext=(point[0] + 500, point[1] - 1000),
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='tab:blue', alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color='black'),
                    fontsize=6,
                    color='black'
                )
                texts.append(text)

            # annotate highest three points of simulated Z data
            for point, freq in zip(highest_points_sim, highest_frequencies_sim):
                text = ax.annotate(
                    f"{freq:.2e} Hz",
                    xy=(point[0], point[1]),
                    xytext=(point[0] + 500, point[1] - 1000),
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='tab:orange', alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color='black'),
                    fontsize=6,
                    color='black'
                )
                texts.append(text)

            # Adjust the labels to prevent overlaps for the current subplot
            adjust_text(texts, ax=ax, ensure_inside_axes=False, avoid_self=False,
                        only_move={'points': 'x', 'text': 'x'})
            # adjust_text(texts, ax=ax)

            ax.set_title(f'Voltage = {other_data["potential_set"][i]}')
            ax.set(xlabel='Z_re (Ohm.cm2)', ylabel='-Z_imag (Ohm.cm2)')
            ax.legend(['Experimental', 'Simulation'])
            ax.set_xlim(0, 6000)
            ax.set_ylim(0, -6000)

        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.show()
    except FileNotFoundError:
        print('File not found. Try running optimisation or adjustment first.')
        exit(1)

