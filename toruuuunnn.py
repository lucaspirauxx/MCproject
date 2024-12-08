import numpy as np
import matplotlib.pyplot as plt

def transmission(thickness, sigma_a, sigma_s, neutrons, thickness_list) :
    transmitted = []
    for _ in range(neutrons):
        position = 0
        wall = 0
        mu = 1
        # On suppose pour l'instant qu'il peut pas changer de mur au premier flight

        while position < thickness :

            free_flight = -np.log(np.random.rand()) / (sigma_a[wall] + sigma_s[wall])
            position += free_flight * mu

            if position > thickness:
                transmitted.append(1)
                break

            if sum(thickness_list[:wall + 1]) < position < thickness:
                position = sum(thickness_list[:wall+1])  # On retourne à l'interface et sample free flight
                wall += 1
                free_flight = -np.log(np.random.rand()) / (sigma_a[wall] + sigma_s[wall])
                position += free_flight*mu

            if 0 < position < sum(thickness_list[:wall]):
                position = sum(thickness_list[:wall])  # On retourne à l'interface et sample free flight
                wall -= 1
                free_flight = -np.log(np.random.rand()) / (sigma_a[wall] + sigma_s[wall])
                position += free_flight*mu

            if position > thickness:
                transmitted.append(1)
                break

            if np.random.rand() <= sigma_a[wall] / (sigma_s[wall] + sigma_a[wall]):
                transmitted.append(0)
                break

            mu_0 = np.random.rand() * 2 - 1  # See references
            phi_0 = 2 * np.pi * np.random.rand()
            mu = mu * mu_0 + np.sqrt(1 - mu ** 2) * np.sqrt(1 - mu_0 ** 2) * np.cos(phi_0)

    transmission_prob = np.mean(transmitted)
    return transmission_prob, np.var(transmitted)

# Variables

thickness_fct = [10, 10, 10,10,10,10]  # For typical concrete wall (cm)
sigma_a = [0.01, 0.01, 0.01,0.01,0.01]  # Absorption for concrete wall  (cm-1)
sigma_s = [0.4, 0.4, 0.4,0.4,0.4,0.4]


def generate_uncertain_properties():
    # Define uncertainty ranges around each value
    thickness = np.random.normal(10, 2, size=1)  # Mean of 10 cm, with some variation (e.g., ±2 cm)
    sigma_a= np.random.normal(0.01, 0.002, size=1)  # Absorption around 0.01 cm-1 with ±0.002 variation
    sigma_s = np.random.normal(0.4, 0.05, size=1)   # Scattering around 0.4 cm-1 with ±0.05 variation
    return thickness, sigma_a, sigma_s

# Call
transmission_probability = []
ecart_type = []
transmission_probability_1 = []
ecart_type_1 = []

neutrons = 50000
thickness_plot = np.linspace(1, 60, 60)
N = 10

for thickness in thickness_plot:
    prob, var = transmission(thickness, sigma_a, sigma_s, neutrons, thickness_fct)
    transmission_probability.append(prob)
    std = np.sqrt(var/neutrons)
    ecart_type.append(std/prob)

    results = []
    for _ in range(N):
        layers = 10  # Random layers
        sigma_a_list = []
        sigma_s_list = []
        thickness_list = []
        for i in range(layers):  # And create the properties of the wall, for each layer
            thickness_layer, sigma_a, sigma_s = generate_uncertain_properties()
            sigma_s_list.append(sigma_s[0])
            sigma_a_list.append(sigma_a[0])
            thickness_list.append(thickness_layer[0])
        # Then for this set of properties, run a simulation
        prob, var = transmission(thickness, sigma_a_list, sigma_s_list, neutrons, thickness_list)
        results.append((prob, var))  # Memorize the mean value and the variance


    # Total mean for every fixed data
    I_w = [I[0] for I in results]
    I_tot = np.mean(I_w)
    transmission_probability_1.append(I_tot)
    # Varriance for each data
    var_w = [var[1] for var in results]
    # Variance of fixed data estimations + variance of estimate due tu incertainties
    var = np.mean(var_w) + np.var(I_w)
    std = np.sqrt(var / neutrons)
    ecart_type_1.append(std/I_tot)
    print(thickness)

# Plot
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(thickness_plot,  transmission_probability, label='Normal', color='blue')
plt.plot(thickness_plot,  transmission_probability_1, label='Double random', color='red')
plt.legend()
plt.xlabel('Thickness (cm)')
plt.ylabel('Transmission probability')

plt.subplot(1, 2, 2)
plt.plot(thickness_plot, ecart_type,  label='Normal', color='blue')
plt.plot(thickness_plot,  ecart_type_1, label='Double random', color='red')
plt.legend()
plt.xlabel('Thickness (cm)')
plt.ylabel('Standard deviation')

plt.show()
