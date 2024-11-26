import numpy as np
import matplotlib.pyplot as plt

def transmission(thickness, sigma_a, sigma_s, neutrons) :
    transmitted = 0  # Compte les neutrons qui passent le mur
    for _ in range(neutrons):  # Pour chaque neutron
        position = 0
        while position < thickness :
            # Sample le transition kernel pour le free flight
            free_flight = -np.log(np.random.rand()) / (sigma_a + sigma_s)
            position += free_flight
            # Collision : p(i)=sigma(i)/sigma(t)
            if np.random.rand() <= sigma_a / (sigma_s + sigma_a):
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
thickness_list = np.linspace(0, 100, 10)   # For typical concrete wall (cm)
sigma_a = 0.01  # Absorption for concrete wall  (cm-1)
sigma_s = 0.4   # Scattering for concrete wall  (cm-1)
neutrons = 10000

# Call
transmission_probability = []
accuracy = []
for thickness in thickness_list:
    prob, acc = transmission(thickness, sigma_a, sigma_s, neutrons)  # Transmission retourne prob et acc
    transmission_probability.append(prob)
    accuracy.append(acc)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(thickness_list, transmission_probability)
plt.xlabel('Thickness')
plt.ylabel('Transmission probability')

plt.subplot(1, 2, 2)
plt.plot(thickness_list, accuracy)
plt.xlabel('Thickness')
plt.ylabel('Accuracy')

plt.show()