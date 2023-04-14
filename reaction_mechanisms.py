### List of potential Reaction Mechanisms
# William Pearson
# With the help of Dr. Ramanathan Srinivasan and Dr. Fathima Fasmin 

###Packages###
import csv  # Read and Write Files
import math # Use Math Functions (sqrt, e)
import numpy as np # Use Matrices
from scipy.optimize import minimize # Optimise Function
import matplotlib.pyplot as plt # Plot Graphs
import datetime


###Initial Values and Upper/lower Boundaries for parameters###

constant_names = [  #Temporaryinitial_parameters                
                'Gamma',
                'k10',
                'km10',
                'alpha1',
                'k20',
                'km20',
                'alpha2',
                'k30',
                'km30',
                'alpha3',
                'k40',
                'km40',
                'alpha4',
                'k50',
                'km50',
                'alpha5'
                ]

temp_initial_parameters = [  #Temporaryinitial_parameters
            1.6710180e-09,  # Gamma
            1.0e-05,        # k10
            1.0e-07,        # km10
            0.5,            # alpha1
            1.0e-05,        # k20
            1.0e-07,        # km20
            0.5,            # alpha2
            1.0e-05,        # k30
            1.0e-07,        # km30
            0.5,             # alpha3
            1.0e-05,        # k40
            1.0e-07,        # km40
            0.5,             # alpha4
            1.0e-05,        # k50
            1.0e-07,        # km50
            0.5             # alpha5
        ]

initial_parameters = [(value if index % 4 == 3 else math.log10(value)) for index, value in enumerate(temp_initial_parameters)]

lower_bound_parameters = [
                        math.log10(1e-30),   #Gamma
                        math.log10(1e-20),   #k10
                        math.log10(1e-20),   #km10
                        0,                   #alpha1
                        math.log10(1e-20),   #k20
                        math.log10(1e-20),   #km20
                        0,                   #alpha2
                        math.log10(1e-20),   #k30
                        math.log10(1e-20),   #km30
                        0,                   #alpha3
                        math.log10(1e-20),   #k40
                        math.log10(1e-20),   #km40
                        0,                   #alpha4
                        math.log10(1e-20),   #k50
                        math.log10(1e-20),   #km50
                        0                    #alpha5
                        ]

upper_bound_parameters = [
                        math.log10(1e-2),   #Gamma
                        math.log10(1e-2),   #k10
                        math.log10(1e-2),   #km10
                        1,                  #alpha1
                        math.log10(1e-2),   #k20
                        math.log10(1e-2),   #km20
                        1,                  #alpha2
                        math.log10(1e-2),   #k30
                        math.log10(1e-2),   #km30
                        1,                  #alpha3
                        math.log10(1e-2),   #k40
                        math.log10(1e-2),   #km40
                        1,                  #alpha4
                        math.log10(1e-2),   #k50
                        math.log10(1e-2),   #km50
                        1                   #alpha5
                        ]

# Put all bounds together so that the minimise function can read it
all_bounds = [[lower, upper] for lower, upper in zip(lower_bound_parameters, upper_bound_parameters)]


def two_step_buffer_vt_mechanism_dc(potential, parameters, variables):

    n = variables[0]
    F = variables[1]
    R = variables[2]
    T = variables[3]
    conc_hp = variables[4] #M (mol/dm3)
    conc_hp= conc_hp /10 /10 /10 #(mol/cm3)

    Gamma       = 10**parameters[0] # Is not used in DC calculations
    k10         = 10**parameters[1]
    km10        = 10**parameters[2]
    a1          = parameters[3]
    k30         = 10**parameters[7]
    km30        = 10**parameters[8]
    a3          = parameters[9]

    b1 = -(a1*n*F)/(R*T)
    bm1 = -(-(1-a1)*n*F)/(R*T)
    #b3 = -(a3*n*F)/(R*T) Is not needed, Chemical step 3 (tafel step not dependant on potential)

    k1dc = k10 * math.exp(b1*potential)
    km1dc = km10 * math.exp(bm1*potential)
    k3 = k30 #* math.exp(b3) Is not needed, Chemical step 3 (tafel step not dependant on potential)

    a = -k3
    b = -(k1dc*conc_hp)-(km1dc)
    c = (k1dc*conc_hp)
    thetass_value1p = (-b + math.sqrt((b**2)-(4*a*c))) / (2*a)
    thetass_value2n = (-b - math.sqrt((b**2)-(4*a*c))) / (2*a)
    thetass_value = max(thetass_value1p, thetass_value2n)

    A = k1dc * (1-thetass_value)*conc_hp
    B = km1dc * thetass_value
    
    ifss_value = F * (-A+B) # A/cm2

    return ifss_value, thetass_value


def two_step_buffer_vt_mechanism_eis(potential, parameters, r_sol, Q, alpha, freq, variables):

    n = variables[0]
    F = variables[1]
    R = variables[2]
    T = variables[3]
    conc_hp = variables[4] #M (mol/dm3)
    conc_hp= conc_hp /10 /10 /10 #(mol/cm3)

    Gamma       = 10**parameters[0]
    k10         = 10**parameters[1]
    km10        = 10**parameters[2]
    a1          = parameters[3]
    k30         = 10**parameters[7]
    km30        = 10**parameters[8]
    a3          = parameters[9]
    
    b1 = -(a1*n*F)/(R*T)
    bm1 = -(-(1-a1)*n*F)/(R*T)
    #b3 = -(a3*n*F)/(R*T)

    k1dc = k10 * math.exp(b1*potential)
    km1dc = km10 * math.exp(bm1*potential)
    k3 = k30 #Chemical step 3 (tafel step not dependant on potential)

    omega = 2 * math.pi * freq
    
    a = -k3
    b = -(k1dc*conc_hp)-(km1dc)
    c = (k1dc*conc_hp)
    thetass_value1p = (-b + math.sqrt((b**2)-(4*a*c))) / (2*a)
    thetass_value2n = (-b - math.sqrt((b**2)-(4*a*c))) / (2*a)
    thetass_value = max(thetass_value1p, thetass_value2n)

    A = (k1dc*b1*(1-thetass_value)*conc_hp)-(km1dc*bm1*thetass_value)
    B = (k1dc*conc_hp)+(km1dc)+(2*k3*thetass_value)+(complex(0,omega*Gamma))
    dt_dp = A/B

    A = k1dc*-dt_dp*conc_hp
    B = k1dc*b1*(1-thetass_value)*conc_hp
    C = km1dc*bm1*thetass_value                                                                                        
    D = km1dc*dt_dp
    
    Zf = F*(-A -B +C +D)
    Zc = (Q*(complex(0, omega))**alpha) #Should be 1/ jwCdl but this gives divide by zero erros.
                                        #The next step is an inverse anyway so I do the inverse here instead.
    
    Zt = r_sol + (1 / ((Zc) + (Zf)))
    
    return Zt


def two_step_cat_buffer_vh_mechanism_dc(potential, parameters, variables):

    n = variables[0]
    F = variables[1]
    R = variables[2]
    T = variables[3]
    conc_hp = variables[4] #M (mol/dm3)
    conc_hp= conc_hp /10 /10 /10 #(mol/cm3)

    Gamma       = 10**parameters[0] # Is not used in DC calculations
    k10         = 10**parameters[1]
    km10        = 10**parameters[2]
    a1          = parameters[3]
    k20         = 10**parameters[4]
    km20        = 10**parameters[5]
    a2          = parameters[6]
    
    b1 = -(a1*n*F)/(R*T)
    bm1 = -(-(1-a1)*n*F)/(R*T)
    b2 = -(a2*n*F)/(R*T)

    k1dc = k10 * math.exp(b1*potential)
    km1dc = km10 * math.exp(bm1*potential)
    k2dc = k20 * math.exp(b2*potential)

    thetass_value = ((k1dc*conc_hp)/((k1dc*conc_hp)+km1dc+(k2dc*conc_hp)))

    A = k1dc * (1-thetass_value)*conc_hp
    B = km1dc * thetass_value
    C = k2dc * thetass_value*conc_hp
    
    ifss_value = F * (-A+B-C) # A/cm2
    
    return ifss_value, thetass_value


def two_step_cat_buffer_vh_mechanism_eis(potential, parameters, r_sol, Q, alpha, freq, variables):

    n = variables[0]
    F = variables[1]
    R = variables[2]
    T = variables[3]
    conc_hp = variables[4] #M (mol/dm3)
    conc_hp= conc_hp /10 /10 /10 #(mol/cm3)
    
    Gamma       = 10**parameters[0]
    k10         = 10**parameters[1]
    km10        = 10**parameters[2]
    a1          = parameters[3]
    k20         = 10**parameters[4]
    km20        = 10**parameters[5]
    a2          = parameters[6]
    
    b1 = -(a1*n*F)/(R*T)
    bm1 = -(-(1-a1)*n*F)/(R*T)
    b2 = -(a2*n*F)/(R*T)

    k1dc = k10 * math.exp(b1*potential)
    km1dc = km10 * math.exp(bm1*potential)
    k2dc = k20 * math.exp(b2*potential)

    omega = 2 * math.pi * freq

    thetass_value = (k1dc*conc_hp)/((k1dc*conc_hp)+km1dc+(k2dc*conc_hp))

    A = (k1dc*b1*(1-thetass_value)*conc_hp)-(km1dc*bm1*thetass_value)-(k2dc*b2*thetass_value*conc_hp)
    B = (k1dc*conc_hp)+(km1dc)+(k2dc*conc_hp)+(1j * omega * Gamma)
    dt_dp = A/B
    
    A = k1dc*-dt_dp*conc_hp
    B = k1dc*b1*(1-thetass_value)*conc_hp
    C = km1dc*bm1*thetass_value                                                                                        
    D = km1dc*dt_dp
    E = k2dc*dt_dp*conc_hp
    M = k2dc*b2*thetass_value*conc_hp
    
    Yf = F*(-A -B +C +D -E -M)

    Yc = (1j * omega * (Q**alpha)) #Should be 1/ jwCdl but this gives divide by zero erros.
                                        #The next step is an inverse anyway so I do the inverse here instead.

    Zt = r_sol + (1 / ((Yc) + (Yf)))

    return Zt


def three_step_cat_buffer_vht_mechanism_dc(potential, parameters, variables):

    n = variables[0]
    F = variables[1]
    R = variables[2]
    T = variables[3]
    conc_hp = variables[4] #M (mol/dm3)
    conc_hp= conc_hp /10 /10 /10 #(mol/cm3)

    Gamma       = 10**parameters[0] # Is not used in DC calculations
    k10         = 10**parameters[1]
    km10        = 10**parameters[2]
    a1          = parameters[3]
    k20         = 10**parameters[4]
    km20        = 10**parameters[5]
    a2          = parameters[6]
    k30         = 10**parameters[7]
    km30        = 10**parameters[8]
    a3          = parameters[9]

    b1 = -(a1*n*F)/(R*T)
    bm1 = -(-(1-a1)*n*F)/(R*T)
    b2 = -(a2*n*F)/(R*T)
    #b3 = -(a3*n*F)/(R*T) Is not needed, Chemical step 3 (tafel step not dependant on potential)

    k1dc = k10 * math.exp(b1*potential)
    km1dc = km10 * math.exp(bm1*potential)
    k2dc = k20 * math.exp(b2*potential)
    k3 = k30 #* math.exp(b3*potential) Is not needed, Chemical step 3 (tafel step not dependant on potential)

    a = -k3
    b = -(k1dc*conc_hp)-(km1dc)-(k2dc*conc_hp)
    c = (k1dc*conc_hp)
    thetass_value1p = (-b + math.sqrt((b**2)-(4*a*c))) / (2*a)
    thetass_value2n = (-b - math.sqrt((b**2)-(4*a*c))) / (2*a)
    thetass_value = max(thetass_value1p, thetass_value2n)

    A = k1dc * (1-thetass_value)*conc_hp
    B = km1dc * thetass_value
    C = k2dc * thetass_value*conc_hp
    
    ifss_value = F * (-A+B-C) # A/cm2

    return ifss_value, thetass_value


def three_step_cat_buffer_vht_mechanism_eis(potential, parameters, r_sol, Q, alpha, freq, variables):
    
    n = variables[0]
    F = variables[1]
    R = variables[2]
    T = variables[3]
    conc_hp = variables[4] #M (mol/dm3)
    conc_hp= conc_hp /10 /10 /10 #(mol/cm3)

    Gamma       = 10**parameters[0]
    k10         = 10**parameters[1]
    km10        = 10**parameters[2]
    a1          = parameters[3]
    k20         = 10**parameters[4]
    km20        = 10**parameters[5]
    a2          = parameters[6]
    k30         = 10**parameters[7]
    km30        = 10**parameters[8]
    a3          = parameters[9]
    
    b1 = -(a1*n*F)/(R*T)
    bm1 = -(-(1-a1)*n*F)/(R*T)
    b2 = -(a2*n*F)/(R*T)
    #b3 = -(a3*n*F)/(R*T) Is not needed, Chemical step 3 (tafel step not dependant on potential)

    k1dc = k10 * math.exp(b1*potential)
    km1dc = km10 * math.exp(bm1*potential)
    k2dc = k20 * math.exp(b2*potential)
    k3 = k30 #* math.exp(b3*potential) Is not needed, Chemical step 3 (tafel step not dependant on potential)

    omega = 2 * math.pi * freq
    
    a = -k3
    b = -(k1dc*conc_hp)-(km1dc)-(k2dc*conc_hp)
    c = (k1dc*conc_hp)
    thetass_value1p = (-b + math.sqrt((b**2)-(4*a*c))) / (2*a)
    thetass_value2n = (-b - math.sqrt((b**2)-(4*a*c))) / (2*a)
    thetass_value = max(thetass_value1p, thetass_value2n)

    A = (k1dc*b1*(1-thetass_value)*conc_hp)-(km1dc*bm1*thetass_value)-(k2dc*b2*thetass_value*conc_hp)
    B = (k1dc*conc_hp)+(km1dc)+(k2dc*conc_hp)+(2*k3*thetass_value)+(complex(0,omega*Gamma))
    dt_dp = A/B

    A = k1dc*-dt_dp*conc_hp
    B = k1dc*b1*(1-thetass_value)*conc_hp
    C = km1dc*bm1*thetass_value                                                                                        
    D = km1dc*dt_dp
    E = k2dc*dt_dp*conc_hp
    M = k2dc*b2*thetass_value*conc_hp
    
    Zf = F*(-A -B +C +D -E -M)    
    Zc = (Q*(complex(0, omega))**alpha) #Should be 1/ jwCdl but this gives divide by zero erros.
                                        #The next step is an inverse anyway so I do the inverse here instead.
    
    Zt = r_sol + (1 / ((Zc) + (Zf)))
    
    return Zt


def four_step_direct_vt45_mechanism_dc(potential, parameters, variables):
    
    n = variables[0]
    F = variables[1]
    R = variables[2]
    T = variables[3]
    conc_hp = variables[4] #M (mol/dm3)
    conc_hp= conc_hp /10 /10 /10 #(mol/cm3)
    conc_h2 = variables[5]#M (mol/dm3)
    conc_h2= conc_h2 /10 /10 /10 #(mol/cm3)
    conc_h = variables[6]#M (mol/dm3)
    conc_h= conc_h /10 /10 /10 #(mol/cm3)

    Gamma       = 10**parameters[0]
    k10         = 10**parameters[1]
    km10        = 10**parameters[2]
    a1          = parameters[3]
    k30         = 10**parameters[7]
    km30        = 10**parameters[8]
    a3          = parameters[9]
    k40         = 10**parameters[10]
    km40        = 10**parameters[11]
    a4          = parameters[12]
    k50         = 10**parameters[13]
    km50        = 10**parameters[14]
    a5          = parameters[15]

    b1 = -(a1*n*F)/(R*T)
    bm1 = -(-(1-a1)*n*F)/(R*T)
    #b3 = -(a3*n*F)/(R*T) Is not needed, Chemical step 3 (tafel step not dependant on potential)
    b4 = -(a4*n*F)/(R*T)
    b5 = -(a5*n*F)/(R*T)

    k1dc = k10 * math.exp(b1*potential)
    km1dc = km10 * math.exp(bm1*potential)
    k3 = k30 #* math.exp(b3*potential) Is not needed, Chemical step 3 (tafel step not dependant on potential)
    k4dc = k40 * math.exp(b4*potential)
    k5dc = k50 * math.exp(b5*potential)
    
    a = -k3
    b = -(k1dc*conc_hp)-(km1dc)-(k4dc*conc_h2)-(k5dc*conc_h)
    c = (k1dc*conc_hp)+(k4dc*conc_h2)+(k5dc*conc_h)
    thetass_value1p = (-b + math.sqrt((b**2)-(4*a*c))) / (2*a)
    thetass_value2n = (-b - math.sqrt((b**2)-(4*a*c))) / (2*a)
    thetass_value = max(thetass_value1p, thetass_value2n)
    
    A = k1dc * (1-thetass_value)*conc_hp
    B = km1dc * thetass_value
    D = k4dc * (1-thetass_value)*conc_h2
    E = k5dc * (1-thetass_value)*conc_h
    
    ifss_value = F * (-A+B-D-E) # A/cm2

    return ifss_value, thetass_value


def four_step_direct_vt45_mechanism_eis(potential, parameters, r_sol, Q, alpha, freq, variables):
    
    n = variables[0]
    F = variables[1]
    R = variables[2]
    T = variables[3]
    conc_hp = variables[4] #M (mol/dm3)
    conc_hp= conc_hp /10 /10 /10 #(mol/cm3)
    conc_h2 = variables[5]#M (mol/dm3)
    conc_h2= conc_h2 /10 /10 /10 #(mol/cm3)
    conc_h = variables[6]#M (mol/dm3)
    conc_h= conc_h /10 /10 /10 #(mol/cm3)

    Gamma       = 10**parameters[0]
    k10         = 10**parameters[1]
    km10        = 10**parameters[2]
    a1          = parameters[3]
    k30         = 10**parameters[7]
    km30        = 10**parameters[8]
    a3          = parameters[9]
    k40         = 10**parameters[10]
    km40        = 10**parameters[11]
    a4          = parameters[12]
    k50         = 10**parameters[13]
    km50        = 10**parameters[14]
    a5          = parameters[15]

    b1 = -(a1*n*F)/(R*T)
    bm1 = -(-(1-a1)*n*F)/(R*T)
    #b3 = -(a3*n*F)/(R*T) Chemical step 3 (tafel step not dependant on potential)
    b4 = -(a4*n*F)/(R*T)
    b5 = -(a5*n*F)/(R*T)

    k1dc = k10 * math.exp(b1*potential)
    km1dc = km10 * math.exp(bm1*potential)
    k3 = k30 # Chemical step 3 (tafel step not dependant on potential)
    k4dc = k40 * math.exp(b4*potential)
    k5dc = k50 * math.exp(b5*potential)

    omega = 2 * math.pi * freq

    a = -k3
    b = -(k1dc*conc_hp)-(km1dc)-(k4dc*conc_h2)-(k5dc*conc_h)
    c = (k1dc*conc_hp)+(k4dc*conc_h2)+(k5dc*conc_h)
    thetass_value1p = (-b + math.sqrt((b**2)-(4*a*c))) / (2*a)
    thetass_value2n = (-b - math.sqrt((b**2)-(4*a*c))) / (2*a)
    thetass_value = max(thetass_value1p, thetass_value2n)

    A = (k1dc*b1*(1-thetass_value)*conc_hp)-(km1dc*bm1*thetass_value)+(k4dc*b4*(1-thetass_value)*conc_h2)+(k5dc*b5*(1-thetass_value)*conc_h)
    B = (k1dc*conc_hp)+(km1dc)+(2*k3*thetass_value)+(k4dc*conc_h2)+(k5dc*conc_h)+(complex(0,omega*Gamma))
    dt_dp = A/B

    A = k1dc*-dt_dp*conc_hp
    B = k1dc*b1*(1-thetass_value)*conc_hp
    C = km1dc*bm1*thetass_value                                                                                        
    D = km1dc*dt_dp                                                                                       
    G = k4dc*-dt_dp*conc_h2
    H = k4dc*b4*(1-thetass_value)*conc_h2
    I = k5dc*-dt_dp*conc_h
    J = k5dc*b5*(1-thetass_value)*conc_h
    
    Zf = F*(-A -B +C +D -G -H -I -J)
    Zc = (Q*(complex(0, omega))**alpha) #Should be 1/ jwCdl but this gives divide by zero erros.
                                        #The next step is an inverse anyway so I do the inverse here instead.
    
    Zt = r_sol + (1 / ((Zc) + (Zf)))
    
    return Zt


def four_step_cat_direct_vh45_mechanism_dc(potential, parameters, variables):
    
    n = variables[0]
    F = variables[1]
    R = variables[2]
    T = variables[3]
    conc_hp = variables[4] #M (mol/dm3)
    conc_hp= conc_hp /10 /10 /10 #(mol/cm3)
    conc_h2 = variables[5]#M (mol/dm3)
    conc_h2= conc_h2 /10 /10 /10 #(mol/cm3)
    conc_h = variables[6]#M (mol/dm3)
    conc_h= conc_h /10 /10 /10 #(mol/cm3)

    Gamma       = 10**parameters[0]
    k10         = 10**parameters[1]
    km10        = 10**parameters[2]
    a1          = parameters[3]
    k20         = 10**parameters[4]
    km20        = 10**parameters[5]
    a2          = parameters[6]
    k40         = 10**parameters[10]
    km40        = 10**parameters[11]
    a4          = parameters[12]
    k50         = 10**parameters[13]
    km50        = 10**parameters[14]
    a5          = parameters[15]

    b1 = -(a1*n*F)/(R*T)
    bm1 = -(-(1-a1)*n*F)/(R*T)
    b2 = -(a2*n*F)/(R*T)
    b4 = -(a4*n*F)/(R*T)
    b5 = -(a5*n*F)/(R*T)

    k1dc = k10 * math.exp(b1*potential)
    km1dc = km10 * math.exp(bm1*potential)
    k2dc = k20 * math.exp(b2*potential)
    k4dc = k40 * math.exp(b4*potential)
    k5dc = k50 * math.exp(b5*potential)
    
    thetass_value = ((k1dc*conc_hp) + (k4dc*conc_h2) + (k5dc*conc_h)) / ((k1dc*conc_hp) + (km1dc) + (k2dc*conc_hp) + (k4dc*conc_h2) + (k5dc*conc_h))
    
    A = k1dc * (1-thetass_value)*conc_hp
    B = km1dc * thetass_value
    C = k2dc * thetass_value*conc_hp
    D = k4dc * (1-thetass_value)*conc_h2
    E = k5dc * (1-thetass_value)*conc_h
    
    ifss_value = F * (-A+B-C-D-E) # A/cm2

    return ifss_value, thetass_value


def four_step_cat_direct_vh45_mechanism_eis(potential, parameters, r_sol, Q, alpha, freq, variables):
    
    n = variables[0]
    F = variables[1]
    R = variables[2]
    T = variables[3]
    conc_hp = variables[4] #M (mol/dm3)
    conc_hp= conc_hp /10 /10 /10 #(mol/cm3)
    conc_h2 = variables[5]#M (mol/dm3)
    conc_h2= conc_h2 /10 /10 /10 #(mol/cm3)
    conc_h = variables[6]#M (mol/dm3)
    conc_h= conc_h /10 /10 /10 #(mol/cm3)

    Gamma       = 10**parameters[0]
    k10         = 10**parameters[1]
    km10        = 10**parameters[2]
    a1          = parameters[3]
    k20         = 10**parameters[4]
    km20        = 10**parameters[5]
    a2          = parameters[6]
    k40         = 10**parameters[10]
    km40        = 10**parameters[11]
    a4          = parameters[12]
    k50         = 10**parameters[13]
    km50        = 10**parameters[14]
    a5          = parameters[15]

    b1 = -(a1*n*F)/(R*T)
    bm1 = -(-(1-a1)*n*F)/(R*T)
    b2 = -(a2*n*F)/(R*T)
    b4 = -(a4*n*F)/(R*T)
    b5 = -(a5*n*F)/(R*T)

    k1dc = k10 * math.exp(b1*potential)
    km1dc = km10 * math.exp(bm1*potential)
    k2dc = k20 * math.exp(b2*potential)
    k4dc = k40 * math.exp(b4*potential)
    k5dc = k50 * math.exp(b5*potential)

    omega = 2 * math.pi * freq

    thetass_value = ((k1dc*conc_hp) + (k4dc*conc_h2) + (k5dc*conc_h)) / ((k1dc*conc_hp) + (km1dc) + (k2dc*conc_hp) + (k4dc*conc_h2) + (k5dc*conc_h))

    A = (k1dc*b1*(1-thetass_value)*conc_hp)-(km1dc*bm1*thetass_value)-(k2dc*b2*thetass_value*conc_hp)+(k4dc*b4*(1-thetass_value)*conc_h2)+(k5dc*b5*(1-thetass_value)*conc_h)
    B = (k1dc*conc_hp)+(km1dc)+(k2dc*conc_hp)+(k4dc*conc_h2)+(k5dc*conc_h)+(complex(0,omega*Gamma))
    dt_dp = A/B

    A = k1dc*-dt_dp*conc_hp
    B = k1dc*b1*(1-thetass_value)*conc_hp
    C = km1dc*bm1*thetass_value                                                                                        
    D = km1dc*dt_dp
    E = k2dc*dt_dp*conc_hp
    M = k2dc*b2*thetass_value*conc_hp                                                                                        
    G = k4dc*-dt_dp*conc_h2
    H = k4dc*b4*(1-thetass_value)*conc_h2
    I = k5dc*-dt_dp*conc_h
    J = k5dc*b5*(1-thetass_value)*conc_h
    
    Zf = F*(-A -B +C +D -E -M -G -H -I -J)

    Zc = (Q*(complex(0, omega))**alpha) #Should be 1/ jwCdl but this gives divide by zero erros.
                                        #The next step is an inverse anyway so I do the inverse here instead.
    
    Zt = r_sol + (1 / ((Zc) + (Zf)))
    
    return Zt


def five_step_cat_direct_mechanism_dc(potential, parameters, variables):
    
    n = variables[0]
    F = variables[1]
    R = variables[2]
    T = variables[3]
    conc_hp = variables[4] #M (mol/dm3)
    conc_hp= conc_hp /10 /10 /10 #(mol/cm3)
    conc_h2 = variables[5]#M (mol/dm3)
    conc_h2= conc_h2 /10 /10 /10 #(mol/cm3)
    conc_h = variables[6]#M (mol/dm3)
    conc_h= conc_h /10 /10 /10 #(mol/cm3)

    Gamma       = 10**parameters[0]
    k10         = 10**parameters[1]
    km10        = 10**parameters[2]
    a1          = parameters[3]
    k20         = 10**parameters[4]
    km20        = 10**parameters[5]
    a2          = parameters[6]
    k30         = 10**parameters[7]
    km30        = 10**parameters[8]
    a3          = parameters[9]
    k40         = 10**parameters[10]
    km40        = 10**parameters[11]
    a4          = parameters[12]
    k50         = 10**parameters[13]
    km50        = 10**parameters[14]
    a5          = parameters[15]

    b1 = -(a1*n*F)/(R*T)
    bm1 = -(-(1-a1)*n*F)/(R*T)
    b2 = -(a2*n*F)/(R*T)
    #b3 = -(a3*n*F)/(R*T) Is not needed, Chemical step 3 (tafel step not dependant on potential)
    b4 = -(a4*n*F)/(R*T)
    b5 = -(a5*n*F)/(R*T)

    k1dc = k10 * math.exp(b1*potential)
    km1dc = km10 * math.exp(bm1*potential)
    k2dc = k20 * math.exp(b2*potential)
    k3 = k30 #* math.exp(b3*potential) Is not needed, Chemical step 3 (tafel step not dependant on potential)
    k4dc = k40 * math.exp(b4*potential)
    k5dc = k50 * math.exp(b5*potential)
    
    a = -k3
    b = -(k1dc*conc_hp)-(km1dc)-(k2dc*conc_hp)-(k4dc*conc_h2)-(k5dc*conc_h)
    c = (k1dc*conc_hp)+(k4dc*conc_h2)+(k5dc*conc_h)
    thetass_value1p = (-b + math.sqrt((b**2)-(4*a*c))) / (2*a)
    thetass_value2n = (-b - math.sqrt((b**2)-(4*a*c))) / (2*a)
    thetass_value = max(thetass_value1p, thetass_value2n)
    
    A = k1dc * (1-thetass_value)*conc_hp
    B = km1dc * thetass_value
    C = k2dc * thetass_value*conc_hp
    D = k4dc * (1-thetass_value)*conc_h2
    E = k5dc * (1-thetass_value)*conc_h
    
    ifss_value = F * (-A+B-C-D-E) # A/cm2

    return ifss_value, thetass_value


def five_step_cat_direct_mechanism_eis(potential, parameters, r_sol, Q, alpha, freq, variables):
    
    n = variables[0]
    F = variables[1]
    R = variables[2]
    T = variables[3]
    conc_hp = variables[4] #M (mol/dm3)
    conc_hp= conc_hp /10 /10 /10 #(mol/cm3)
    conc_h2 = variables[5]#M (mol/dm3)
    conc_h2= conc_h2 /10 /10 /10 #(mol/cm3)
    conc_h = variables[6]#M (mol/dm3)
    conc_h= conc_h /10 /10 /10 #(mol/cm3)

    Gamma       = 10**parameters[0]
    k10         = 10**parameters[1]
    km10        = 10**parameters[2]
    a1          = parameters[3]
    k20         = 10**parameters[4]
    km20        = 10**parameters[5]
    a2          = parameters[6]
    k30         = 10**parameters[7]
    km30        = 10**parameters[8]
    a3          = parameters[9]
    k40         = 10**parameters[10]
    km40        = 10**parameters[11]
    a4          = parameters[12]
    k50         = 10**parameters[13]
    km50        = 10**parameters[14]
    a5          = parameters[15]

    b1 = -(a1*n*F)/(R*T)
    bm1 = -(-(1-a1)*n*F)/(R*T)
    b2 = -(a2*n*F)/(R*T)
    #b3 = -(a3*n*F)/(R*T) Chemical step 3 (tafel step not dependant on potential)
    b4 = -(a4*n*F)/(R*T)
    b5 = -(a5*n*F)/(R*T)

    k1dc = k10 * math.exp(b1*potential)
    km1dc = km10 * math.exp(bm1*potential)
    k2dc = k20 * math.exp(b2*potential)
    k3 = k30 # Chemical step 3 (tafel step not dependant on potential)
    k4dc = k40 * math.exp(b4*potential)
    k5dc = k50 * math.exp(b5*potential)

    omega = 2 * math.pi * freq

    a = -k3
    b = -(k1dc*conc_hp)-(km1dc)-(k2dc*conc_hp)-(k4dc*conc_h2)-(k5dc*conc_h)
    c = (k1dc*conc_hp)+(k4dc*conc_h2)+(k5dc*conc_h)
    thetass_value1p = (-b + math.sqrt((b**2)-(4*a*c))) / (2*a)
    thetass_value2n = (-b - math.sqrt((b**2)-(4*a*c))) / (2*a)
    thetass_value = max(thetass_value1p, thetass_value2n)

    A = (k1dc*b1*(1-thetass_value)*conc_hp)-(km1dc*bm1*thetass_value)-(k2dc*b2*thetass_value*conc_hp)+(k4dc*b4*(1-thetass_value)*conc_h2)+(k5dc*b5*(1-thetass_value)*conc_h)
    B = (k1dc*conc_hp)+(km1dc)+(k2dc*conc_hp)+(2*k3*thetass_value)+(k4dc*conc_h2)+(k5dc*conc_h)+(complex(0,omega*Gamma))
    dt_dp = A/B

    A = k1dc*-dt_dp*conc_hp
    B = k1dc*b1*(1-thetass_value)*conc_hp
    C = km1dc*bm1*thetass_value                                                                                        
    D = km1dc*dt_dp
    E = k2dc*dt_dp*conc_hp
    M = k2dc*b2*thetass_value*conc_hp                                                                                        
    G = k4dc*-dt_dp*conc_h2
    H = k4dc*b4*(1-thetass_value)*conc_h2
    I = k5dc*-dt_dp*conc_h
    J = k5dc*b5*(1-thetass_value)*conc_h
    
    Zf = F*(-A -B +C +D -E -M -G -H -I -J)

    Zc = (Q*(complex(0, omega))**alpha) #Should be 1/ jwCdl but this gives divide by zero erros.
                                        #The next step is an inverse anyway so I do the inverse here instead.
    
    Zt = r_sol + (1 / ((Zc) + (Zf)))
    
    return Zt
