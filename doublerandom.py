import numpy as np
import time

def trouver_mur(position):
    position_tot = 0
    for i, epaisseur in enumerate(thickness):
        position_tot += epaisseur
        if position <= position_tot:
            return i  # Retourne l'indice du mur
    return -1  # Si le neutron est après le dernier mur


def transmission(thickness, sigma_a, sigma_s, neutrons) :
    transmitted = 0  # Compte les neutrons qui passent le mur
    for _ in range(neutrons):  # Pour chaque neutron
        position = 0
        while position < sum(thickness) :
            wall = trouver_mur(position)
            # Sample le transition kernel pour le free flight
            free_flight = -np.log(np.random.rand()) / ((sigma_a[wall] + sigma_s[wall]))
            position += free_flight
            # Collision : p(i)=sigma(i)/sigma(t)
            if np.random.rand() <= sigma_a[wall] / (sigma_s[wall] + sigma_a[wall]):
                # Absorption : deal with next neutron
                break
            # If scattering : isotropic =} just next free flight
        else :
            transmitted += 1

    transmission_prob = transmitted / neutrons
    # Variance = success x failure / trials
    accuracy = np.sqrt(transmission_prob * (1 - transmission_prob) / neutrons)
    return transmission_prob, accuracy

# Variables

thickness = [100, 100]  # For typical concrete wall (cm)
sigma_a = [0.01, 0.001]  # Absorption for concrete wall  (cm-1)
sigma_s = [0.4, 0.1]   # Scattering for concrete wall  (cm-1)
neutrons = 10000

# Call
start_time = time.time()
prob, acc = transmission(thickness, sigma_a, sigma_s, neutrons)
end_time = time.time()
print("Transmission probability =", prob)
print("Accuracy =", acc)
print("Time =", end_time-start_time)
