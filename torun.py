import numpy as np
import matplotlib.pyplot as plt

def transmission1(thickness, sigma_aa, sigma_ss, neutrons) :
    transmitted = []
    for _ in range(neutrons):
        position = 0
        wall = 0
        weight = 1
        mu = 1

        while position < thickness :

            free_flight = -np.log(np.random.rand()) / (sigma_aa[wall] + sigma_ss[wall])
            position += free_flight * mu

            if position > thickness:
                transmitted.append(weight)
                break

            if sum(thickness_fct[:wall + 1]) < position:
                position = sum(thickness_fct[:wall + 1])
                wall += 1
                free_flight = -np.log(np.random.rand()) / (sigma_aa[wall] + sigma_ss[wall])
                position += free_flight * mu

            elif 0 < position < sum(thickness_fct[:wall]):
                position = sum(thickness_fct[:wall])
                wall -= 1
                free_flight = -np.log(np.random.rand()) / (sigma_aa[wall] + sigma_ss[wall])
                position += free_flight * mu

            if position > thickness:
                transmitted.append(weight)
                break

            weight *= sigma_ss[wall] / (sigma_aa[wall] + sigma_ss[wall])
            if weight < 0.02:
                if np.random.rand() < 0.02:
                    weight /= 0.02
                else:
                    transmitted.append(0)
                    break

            mu_0 = np.random.rand() * 2 - 1  # See references
            phi_0 = 2 * np.pi * np.random.rand()
            mu = mu * mu_0 + np.sqrt(1 - mu ** 2) * np.sqrt(1 - mu_0 ** 2) * np.cos(phi_0)


    transmission_prob = np.mean(transmitted)
    accuracy = np.std(transmitted) / np.sqrt(neutrons)
    return transmission_prob, accuracy/transmission_prob

c = 0.06

def transmission2(thickness, sigma_aa, sigma_ss, neutrons) :
    transmitted = []
    for _ in range(neutrons):
        position = 0
        wall = 0
        weight = 1
        mu = 1

        while position < thickness :

            free_flight = -np.log(np.random.rand()) / (sigma_ss[wall] + sigma_aa[wall] - (c * mu))
            weight *= ((sigma_ss[wall] + sigma_aa[wall]) / (sigma_ss[wall] + sigma_aa[wall] - (c * mu))) * np.exp(-c * mu * free_flight)
            position += free_flight * mu

            if position >= thickness:
                transmitted.append(weight)
                break

            if sum(thickness_fct[:wall + 1]) <= position:
                position = sum(thickness_fct[:wall + 1])
                wall += 1
                free_flight = -np.log(np.random.rand()) / (sigma_ss[wall] + sigma_aa[wall] - (c * mu))
                weight *= ((sigma_ss[wall] + sigma_aa[wall]) / (sigma_ss[wall] + sigma_aa[wall] - (c * mu))) * np.exp(-c * mu * free_flight)
                position += free_flight * mu

            elif 0 <= position < sum(thickness_fct[:wall]):
                position = sum(thickness_fct[:wall])
                wall -= 1
                free_flight = -np.log(np.random.rand()) / (sigma_ss[wall] + sigma_aa[wall] - (c * mu))
                weight *= ((sigma_ss[wall] + sigma_aa[wall]) / (sigma_ss[wall] + sigma_aa[wall] - (c * mu))) * np.exp(-c * mu * free_flight)
                position += free_flight * mu

            if position > thickness:
                transmitted.append(weight)
                break

            if np.random.rand() <= sigma_aa[wall] / (sigma_ss[wall] + sigma_aa[wall]):
                transmitted.append(0)
                break

            mu_0 = np.random.rand() * 2 - 1  # See references
            phi_0 = 2 * np.pi * np.random.rand()
            mu = mu * mu_0 + np.sqrt(1 - mu ** 2) * np.sqrt(1 - mu_0 ** 2) * np.cos(phi_0)


    transmission_prob = np.mean(transmitted)
    accuracy = np.std(transmitted) / np.sqrt(neutrons)
    return transmission_prob, accuracy/transmission_prob


def find_wall(position):
    position_tot = 0
    for i, epaisseur in enumerate(thickness_fct):
        position_tot += epaisseur
        if position < position_tot:
            return i
    return -1

def transmission_split(thickness, sigma_a, sigma_s, neutrons):
    transmitted = [0] * neutrons

    stack = [(0, 1, 1, id) for id in range(neutrons)]

    while stack:
        position, mu, weight, id = stack.pop()
        wall = find_wall(position)
        while position < thickness:
            # Distance libre moyenne
            free_flight = -np.log(np.random.rand()) / (sigma_a[wall] + sigma_s[wall])
            position += free_flight * mu
            wall = find_wall(position)

            # Si le neutron traverse le mur
            if position >= thickness:
                transmitted[id] += weight
                break

            if find_wall(position) > find_wall(position-(free_flight*mu)):
                for _ in range(I):
                    if np.random.rand() <= sigma_a[wall] / (sigma_s[wall] + sigma_a[wall]):
                        continue
                    mu_0 = np.random.rand() * 2 - 1
                    phi_0 = 2 * np.pi * np.random.rand()
                    mu = mu * mu_0 + np.sqrt(1 - mu ** 2) * np.sqrt(1 - mu_0 ** 2) * np.cos(phi_0)
                    stack.append((position, mu, weight/I, id))
                break

            if find_wall(position) < find_wall(position - (free_flight * mu)):
                if np.random.rand() < 1/I:
                    weight *= I
                else:
                    break
            if np.random.rand() <= sigma_a[wall] / (sigma_s[wall] + sigma_a[wall]):
                    break
            mu_0 = np.random.rand() * 2 - 1
            phi_0 = 2 * np.pi * np.random.rand()
            mu = mu * mu_0 + np.sqrt(1 - mu ** 2) * np.sqrt(1 - mu_0 ** 2) * np.cos(phi_0)
        else:
            transmitted[id] += weight
    prob = np.mean(transmitted)
    std = np.std(transmitted)/np.sqrt(neutrons)
    return prob, std/prob


def transmission_layers(thickness, sigma_aa, sigma_ss, neutrons) :
    transmitted = []
    for _ in range(neutrons):
        position = 0
        wall = 0
        mu = 1

        while position < thickness :

            free_flight = -np.log(np.random.rand()) / (sigma_aa[wall] + sigma_ss[wall])
            position += free_flight * mu

            if position > thickness:
                transmitted.append(1)
                break

            if sum(thickness_fct[:wall + 1]) < position:
                position = sum(thickness_fct[:wall + 1])
                wall += 1
                free_flight = -np.log(np.random.rand()) / (sigma_aa[wall] + sigma_ss[wall])
                position += free_flight * mu

            elif 0 < position < sum(thickness_fct[:wall]):
                position = sum(thickness_fct[:wall])
                wall -= 1
                free_flight = -np.log(np.random.rand()) / (sigma_aa[wall] + sigma_ss[wall])
                position += free_flight * mu

            if position > thickness:
                transmitted.append(1)
                break

            if np.random.rand() <= sigma_aa[wall] / (sigma_ss[wall] + sigma_aa[wall]):
                transmitted.append(0)
                break

            mu_0 = np.random.rand() * 2 - 1  # See references
            phi_0 = 2 * np.pi * np.random.rand()
            mu = mu * mu_0 + np.sqrt(1 - mu ** 2) * np.sqrt(1 - mu_0 ** 2) * np.cos(phi_0)


    transmission_prob = np.mean(transmitted)
    accuracy = np.std(transmitted) / np.sqrt(neutrons)
    return transmission_prob, accuracy/transmission_prob

# Variables

thickness_fct = [5,20,5,20,5,20,10]  # For typical concrete wall (cm)
sigma_aa = [0.01, 0.005,0.01,0.005,0.01,0.1]  # Absorption for concrete wall  (cm-1)
sigma_ss = [0.4, 0.4,0.4,0.4,0.4,0.4,0.4]   # Scattering for concrete wall  (cm-1)
I = 2

# Variables
thickness_list = np.linspace(1, 60, 60)   # For typical concrete wall (cm)

# Call
transmission_probability = []
ecart_type = []
transmission_probability_1 = []
ecart_type_1 = []
transmission_probability_2 = []
ecart_type_2 = []
transmission_probability_3 = []
ecart_type_3 = []

for thickness in thickness_list:
    prob, acc = transmission_layers(thickness, sigma_aa, sigma_ss, 50000)
    transmission_probability.append(prob)
    ecart_type.append(acc)

    prob, acc = transmission1(thickness, sigma_aa, sigma_ss, 50000)
    transmission_probability_1.append(prob)
    ecart_type_1.append(acc)

    prob, acc = transmission2(thickness, sigma_aa, sigma_ss, 50000)
    transmission_probability_2.append(prob)
    ecart_type_2.append(acc)

    prob, acc = transmission_split(thickness, sigma_aa, sigma_ss, 50000)
    transmission_probability_3.append(prob)
    ecart_type_3.append(acc)
    print(thickness)


# Plot
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(thickness_list,  transmission_probability, label='Normal', color='blue')
plt.plot(thickness_list,  transmission_probability_1, label='Bias 1', color='red')
plt.plot(thickness_list,  transmission_probability_2, label='Bias 2', color='green')
plt.plot(thickness_list,  transmission_probability_3, label='Splitting', color='black')
plt.legend()
plt.xlabel('Thickness (cm)')
plt.ylabel('Transmission probability')

plt.subplot(1, 2, 2)
plt.plot(thickness_list, ecart_type,  label='Normal', color='blue')
plt.plot(thickness_list,  ecart_type_1, label='Bias 1', color='red')
plt.plot(thickness_list, ecart_type_2,  label='Bias 2', color='green')
plt.plot(thickness_list,  ecart_type_3, label='Splitting', color='black')
plt.legend()
plt.xlabel('Thickness (cm)')
plt.ylabel('Standard deviation')

plt.show()
