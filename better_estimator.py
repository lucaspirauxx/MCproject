import numpy as np
import time

def transmission(thickness, sigma_a, sigma_s, neutrons) :
    transmitted = 0  # Compte les neutrons qui passent le mur
    for _ in range(neutrons):  # Pour chaque neutron
        position = 0
        # Premier vol libre avant la boucle de collision
        free_flight = -np.log(np.random.rand()) / (sigma_a + sigma_s)
        position += free_flight

        if position >= thickness:
            transmitted += np.exp(-thickness * (sigma_a + sigma_s))
            continue

        while position < thickness :
            # Collision : p(i)=sigma(i)/sigma(t)
            if np.random.rand() <= sigma_a / (sigma_s + sigma_a):
                # Absorption : deal with next neutron
                break
            # If scattering : isotropic =} just next free flight
            free_flight = -np.log(np.random.rand()) / (sigma_a + sigma_s)
            position += free_flight
        else :
            transmitted += np.exp(-(thickness-(position-free_flight)) * (sigma_a + sigma_s))

    transmission_prob = transmitted / neutrons
    # Variance = success x failure / trials
    accuracy = np.sqrt(transmission_prob * (1 - transmission_prob) / neutrons)
    return transmission_prob, accuracy

# Variables
thickness = 200  # For typical concrete wall (cm)
sigma_a = 0.01  # Absorption for concrete wall  (cm-1)
sigma_s = 0.4   # Scattering for concrete wall  (cm-1)
neutrons = 10000

# Call
start_time = time.time()
prob, acc = transmission(thickness, sigma_a, sigma_s, neutrons)
end_time = time.time()
print("Transmission probability =", prob)
print("Accuracy =", acc)
print("Time =", end_time-start_time)
