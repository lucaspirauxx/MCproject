import numpy as np
import time

def transmission(thickness, sigma_a, sigma_s, neutrons, bias) :
    transmitted = 0  # Compte les neutrons qui passent le mur
    for _ in range(neutrons):  # Pour chaque neutron
        position = 0
        weight = 1
        while position < thickness :
            # Sample le transition kernel pour le free flight
            free_flight = -np.log(np.random.rand()) / ((sigma_a + sigma_s)*bias)
            position += free_flight
            # Collision : p(i)=sigma(i)/sigma(t)
            if np.random.rand() <= sigma_a / (sigma_s + sigma_a):
                # Absorption : deal with next neutron
                break
            # If scattering : isotropic =} just next free flight
        else :
            transmitted += weight*bias

    transmission_prob = transmitted / neutrons
    # Variance = success x failure / trials
    accuracy = np.sqrt(transmission_prob * (1 - transmission_prob) / neutrons)
    return transmission_prob, accuracy

# Variables
thickness = 200  # For typical concrete wall (cm)
sigma_a = 0.01  # Absorption for concrete wall  (cm-1)
sigma_s = 0.4   # Scattering for concrete wall  (cm-1)
neutrons = 10000
bias = 1

# Call
start_time = time.time()
prob, acc = transmission(thickness, sigma_a, sigma_s, neutrons, bias)
end_time = time.time()
print("Transmission probability =", prob)
print("Accuracy =", acc)
print("Time =", end_time-start_time)
