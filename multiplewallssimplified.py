import numpy as np
import time

def transmission(thickness, sigma_a, sigma_s, neutrons) :
    transmitted = 0
    for _ in range(neutrons):
        transmitted_wall = 0
        for wall in range(len(thickness)):
            position = 0
            free_flight = -np.log(np.random.rand()) / (sigma_a[wall] + sigma_s[wall])
            position += free_flight
            while position < thickness[wall] :

                if np.random.rand() <= sigma_a[wall] / (sigma_s[wall] + sigma_a[wall]):
                    break

                free_flight = -np.log(np.random.rand()) / (sigma_a[wall] + sigma_s[wall])
                position += free_flight

            else:
                transmitted_wall += 1  # A optimiser, si ça passe pas on évalue pas les autres murs
        if transmitted_wall == len(thickness):
            transmitted += 1

    transmission_prob = transmitted / neutrons
    accuracy = np.sqrt(transmission_prob * (1 - transmission_prob) / neutrons)
    return transmission_prob, accuracy

# Variables

thickness = [100, 50, 50]  # For typical concrete wall (cm)
sigma_a = [0.01, 0.02, 0.05]  # Absorption for concrete wall  (cm-1)
sigma_s = [0.4, 0.6, 0.8]   # Scattering for concrete wall  (cm-1)
neutrons = 10000

# Call
start_time = time.time()
prob, acc = transmission(thickness, sigma_a, sigma_s, neutrons)
end_time = time.time()
print("Transmission probability =", prob)
print("Accuracy =", acc)
print("Time =", end_time-start_time)
