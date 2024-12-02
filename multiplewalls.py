import numpy as np
import time

def trouver_mur(position):
    position_tot = 0
    for i, epaisseur in enumerate(thickness):
        position_tot += epaisseur
        if position < position_tot:
            return i
    return -1


def transmission(thickness, sigma_a, sigma_s, neutrons) :
    transmitted = 0
    for _ in range(neutrons):
        position = 0
        wall = 0
        # On suppose pour l'instant qu'il peut pas changer de mur au premier flight
        free_flight = -np.log(np.random.rand()) / (sigma_a[wall] + sigma_s[wall])
        position += free_flight
        while position < sum(thickness) :

            if np.random.rand() <= sigma_a[wall] / (sigma_s[wall] + sigma_a[wall]):
                break

            free_flight = -np.log(np.random.rand()) / (sigma_a[wall] + sigma_s[wall])
            if sum(thickness[:wall + 1]) < position + free_flight < sum(thickness):
                position = sum(thickness[:wall+1])  # On retourne à l'interface et sample free flight
                wall += 1
                free_flight = -np.log(np.random.rand()) / (sigma_a[wall] + sigma_s[wall])

            position += free_flight
        else:
            transmitted += 1

    transmission_prob = transmitted / neutrons
    accuracy = np.sqrt(transmission_prob * (1 - transmission_prob) / neutrons)
    return transmission_prob, accuracy

# Variables

thickness = [100,100]  # For typical concrete wall (cm)
sigma_a = [0.01, 0.01]  # Absorption for concrete wall  (cm-1)
sigma_s = [0.4, 0.4]   # Scattering for concrete wall  (cm-1)
neutrons = 10000

# Call
start_time = time.time()
prob, acc = transmission(thickness, sigma_a, sigma_s, neutrons)
end_time = time.time()
print("Transmission probability =", prob)
print("Accuracy =", acc)
print("Time =", end_time-start_time)
