import numpy as np

def transmission(thickness, sigma_a, sigma_s, neutrons) :
    transmitted = 0
    transmitted_antithetic = 0
    for _ in range(neutrons):  # Pour chaque neutron
        position = 0
        position_antithetic = 0
        absorbed = [False, False]
        traversed = [False, False]

        while (not absorbed[0] or not traversed[0]) and (not absorbed[1] or not traversed[1]):
            rand = np.random.rand()
            if not absorbed[0]:
                free_flight = -np.log(rand) / (sigma_a + sigma_s)
                position += free_flight
                if position >= thickness:
                    traversed[0] = True
            if not absorbed[1]:
                free_flight_antithtetic = -np.log(1-rand) / (sigma_a + sigma_s)
                position_antithetic += free_flight_antithtetic
                if position_antithetic >= thickness:
                    traversed[1] = True

            if rand <= sigma_a / (sigma_s + sigma_a):
                absorbed[0] = True
            if 1 - rand <= sigma_a / (sigma_s + sigma_a):
                absorbed[1] = True

            if absorbed[0] and absorbed[1]:
                break
        else :
            if traversed[0]:
                transmitted += 1
            if traversed[1]:
                transmitted_antithetic += 1

    transmission_prob = (transmitted + transmitted_antithetic)/ 2*neutrons
    # Variance = success x failure / trials
    accuracy = np.sqrt(transmission_prob * (1 - transmission_prob) / neutrons)
    return transmission_prob, accuracy

# Variables
thickness = 200   # For typical concrete wall (cm)
sigma_a = 0.01  # Absorption for concrete wall  (cm-1)
sigma_s = 0.4   # Scattering for concrete wall  (cm-1)
neutrons = 10000

# Call
prob, acc = transmission(thickness, sigma_a, sigma_s, neutrons)

print("Transmission probability =", prob)
print("Accuracy =", acc)


