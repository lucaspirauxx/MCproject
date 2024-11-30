import numpy as np


def transmission(thickness, sigma_a, sigma_s, neutrons):
    transmitted = 0  # Compte les neutrons qui passent le mur

    for _ in range(neutrons):  # Pour chaque neutron
        position = 0
        position_antithetic = 0

        # Trajectoire normale
        while position < thickness:
            # Sample le transition kernel pour le free flight
            free_flight = -np.log(np.random.rand()) / (sigma_a + sigma_s)
            position += free_flight
            # Collision : p(i) = sigma(i) / sigma(t)
            if np.random.rand() <= sigma_a / (sigma_s + sigma_a):
                # Absorption : stop cette trajectoire
                break

        # Trajectoire antithétique
        while position_antithetic < thickness:
            free_flight_antithetic = -np.log(1 - np.random.rand()) / (sigma_a + sigma_s)
            position_antithetic += free_flight_antithetic
            if np.random.rand() <= sigma_a / (sigma_s + sigma_a):
                # Absorption pour la trajectoire antithétique
                break

        # Si l'une des trajectoires a traversé l'épaisseur, on l'ajoute
        if position >= thickness:
            transmitted += 1
        if position_antithetic >= thickness:
            transmitted += 1

    # Calcul de la probabilité de transmission
    transmission_prob = transmitted / (2 * neutrons)  # Moyenne des deux trajectoires
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

