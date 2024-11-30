import time
import numpy as np

def transmission(thickness, sigma_a, sigma_s, neutrons) :
    transmitted = 0
    for _ in range(neutrons):  # Pour chaque neutron
        position = 0
        position_antithetic = 0
        absorbed = [False, False]
        traversed = [False, False]

        while not (absorbed[0] or traversed[0]) or not (absorbed[1] or traversed[1]):
            rand = np.random.rand()
            if not (absorbed[0] or traversed[0]):
                free_flight = -np.log(rand) / (sigma_a + sigma_s)
                position += free_flight
                if position >= thickness:
                    traversed[0] = True
                elif rand <= sigma_a / (sigma_s + sigma_a):
                    absorbed[0] = True
            if not (absorbed[1] or traversed[1]):
                free_flight_antithtetic = -np.log(1-rand) / (sigma_a + sigma_s)
                position_antithetic += free_flight_antithtetic
                if position_antithetic >= thickness:
                    traversed[1] = True
                elif 1 - rand <= sigma_a / (sigma_s + sigma_a):
                    absorbed[1] = True
        if traversed[0]:
            transmitted += 1
        if traversed[1]:
            transmitted += 1

    transmission_prob = (transmitted)/(neutrons*2)
    # Variance = success x failure / trials
    accuracy = np.sqrt(transmission_prob * (1 - transmission_prob) / neutrons)
    return transmission_prob, accuracy

# Variables
thickness = 200   # For typical concrete wall (cm)
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

