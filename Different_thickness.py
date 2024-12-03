import numpy as np
import matplotlib.pyplot as plt

def transmission(thickness, sigma_a, sigma_s, neutrons) :
    transmitted = 0  # Compte les neutrons qui passent le mur
    for _ in range(neutrons):  # Pour chaque neutron
        position = 0
        mu = 1
        # First free flight
        free_flight = -np.log(np.random.rand()) / (sigma_a + sigma_s)
        position += free_flight

        if position >= thickness:
            transmitted += 1
            continue

        while position < thickness:
            # Collision : p(i)=sigma(i)/sigma(t)
            if np.random.rand() <= sigma_a / (sigma_s + sigma_a):
                # Absorption : deal with next neutron
                break
            # If scattering : isotropic =}
            mu_0 = np.random.rand()*2 - 1
            phi_0 = 2*np.pi*np.random.rand()
            mu = mu*mu_0 + np.sqrt(1-mu**2)*np.sqrt(1-mu_0**2)*np.cos(phi_0)
            # Sample le transition kernel pour le free flight
            free_flight = -np.log(np.random.rand()) / (sigma_a + sigma_s)
            position += free_flight * mu
        else:

            transmitted += 1

    transmission_prob = transmitted / neutrons
    # Variance = success x failure / trials
    ecart_type = np.sqrt(transmission_prob * (1 - transmission_prob) / neutrons)
    return transmission_prob, ecart_type

# Variables
thickness_list = np.linspace(0, 50, 5)   # For typical concrete wall (cm)
sigma_a = 0.01  # Absorption for concrete wall  (cm-1)
sigma_s = 0.4   # Scattering for concrete wall  (cm-1)
neutrons = np.linspace(100, 10000, 10)

# Call
transmission_probability = []
ecart_type = []
for thickness in thickness_list:
    prob, acc = transmission(thickness, sigma_a, sigma_s, 5000)
    transmission_probability.append(prob)
    ecart_type.append(acc)

# Call
transmission_probability_n = []
ecart_type_n = []
for neutron in neutrons:
    prob, acc = transmission(30, sigma_a, sigma_s, int(neutron))
    transmission_probability_n.append(prob)
    ecart_type_n.append(acc)

"""# sigma_a/sigma_t sensitivity
ratio = [0.1, 0.3, 0.5, 1, 2, 3, 4]
transmission_probability_ratio = []
ratio_cross_section = []
for i in ratio:
    transmission_probability_ratio.append(transmission(50, sigma_a*i, sigma_s, neutrons)[0])
    ratio_cross_section.append(sigma_a*i/sigma_s)"""

# Plot
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(thickness_list, ecart_type)
plt.xlabel('Thickness')
plt.ylabel('Ecart type')

plt.subplot(2, 2, 2)
plt.plot(thickness_list, transmission_probability)
plt.xlabel('Thickness')
plt.ylabel('Transmission probability')

plt.subplot(2, 2, 3)
plt.plot(neutrons, ecart_type_n)
plt.xlabel('Neutrons')
plt.ylabel('Ecart type')
plt.xscale("log")
plt.yscale("log")

plt.subplot(2, 2, 4)
plt.plot(neutrons, transmission_probability_n)
plt.xlabel('Neutrons')
plt.ylabel('Transmission probability')

plt.show()