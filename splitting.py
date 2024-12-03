import numpy as np
import time

# Constants
m = 10  # Number of splits per scattering
I_ratio = 2

def transmission(thickness, sigma_a, sigma_s, neutrons):
    mfp = 1/(sigma_a+sigma_s)

    transmitted = 0  # Count neutrons that pass through the wall
    for _ in range(neutrons):  # For each neutron
        position = 0
        weight = 1

        # First free flight
        free_flight = -np.log(np.random.rand()) / (sigma_a + sigma_s)
        position += free_flight

        if position >= thickness:
            transmitted += 1
            continue

        while position < thickness:
            # Absorption (if the neutron is absorbed, stop the simulation)
            if np.random.rand() <= sigma_a / (sigma_s + sigma_a):
                break

            # Sample transition kernel for free flight
            free_flight = -np.log(np.random.rand()) / (sigma_a + sigma_s)
            position += free_flight

        else:
            weight /= m  # The weight is divided by m when splitting
            for _ in range(m):
                sub_position = position - free_flight
                while sub_position < thickness:
                    sub_free_flight = -np.log(np.random.rand()) / (sigma_a + sigma_s)
                    sub_position += sub_free_flight
                    if np.random.rand() <= sigma_a / (sigma_s + sigma_a):
                        break
                else:
                    transmitted += weight

    transmission_prob = transmitted / neutrons
    accuracy = np.sqrt(transmission_prob * (1 - transmission_prob) / neutrons)
    return transmission_prob, accuracy


# Variables
thickness = 200  # Thickness of concrete wall (cm)
sigma_a = 0.01  # Absorption for concrete wall (cm^-1)
sigma_s = 0.4  # Scattering for concrete wall (cm^-1)
neutrons = 10000  # Number of neutrons in the simulation

# Start the simulation
start_time = time.time()
prob, acc = transmission(thickness, sigma_a, sigma_s, neutrons)
end_time = time.time()

# Output results
print("Transmission probability =", prob)
print("Accuracy =", acc)
print("Time =", end_time - start_time)
