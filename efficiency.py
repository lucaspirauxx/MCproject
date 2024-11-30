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
thickness_list = np.linspace(0, 100, 100)   # For typical concrete wall (cm)
sigma_a = 0.01  # Absorption for concrete wall  (cm-1)
sigma_s = 0.4   # Scattering for concrete wall  (cm-1)
neutrons = 1000

# Call
transmission_probability = []
accuracy = []
for thickness in thickness_list:
    prob, acc = transmission(thickness, sigma_a, sigma_s, neutrons)
    transmission_probability.append(prob)
    accuracy.append(acc)

# sigma_a/sigma_t sensitivity
ratio = [0.1, 0.3, 0.5, 1, 2, 3, 4]
transmission_probability_ratio = []
ratio_cross_section = []
for i in ratio:
    transmission_probability_ratio.append(transmission(50, sigma_a*i, sigma_s, neutrons)[0])
    ratio_cross_section.append(sigma_a*i/sigma_s)

# Plot
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(thickness_list, transmission_probability)
plt.xlabel('Thickness')
plt.ylabel('Transmission probability')

plt.subplot(2, 2, 2)
plt.plot(thickness_list, accuracy)
plt.xlabel('Thickness')
plt.ylabel('Accuracy')

plt.subplot(2, 2, 3)
plt.plot(ratio_cross_section, transmission_probability_ratio)
plt.xlabel('sigma_a/sigma_s')
plt.ylabel('Transmission probability')

plt.show()