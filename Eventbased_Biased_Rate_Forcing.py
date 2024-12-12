import numpy as np
import matplotlib.pyplot as plt

# Event-based Monte Carlo simulation with rate forcing
def event_based_rate_forcing_mc_simulation(lambda_1, mu_1, lambda_s, mu_s, mission_time, iterations):
    # Biased failure rates
    biased_lambda_1 = lambda_1 * 10
    biased_lambda_s = lambda_s * 10
    # New Transition rate matrix
    A = np.array([
        [-biased_lambda_1, biased_lambda_1, 0, 0, 0, 0],
        [mu_1, -mu_1-biased_lambda_s, biased_lambda_s, 0, 0, 0],
        [0, mu_s, -mu_s-mu_1-biased_lambda_s, mu_1, 0, biased_lambda_s],
        [mu_s, 0, biased_lambda_1, -mu_s-biased_lambda_1, 0, 0],
        [0, 0, 0, 2*mu_s, -2*mu_s-biased_lambda_1, biased_lambda_1],
        [0, 0, 2*mu_s, 0, mu_1, -2*mu_s-mu_1]
    ])
    failure_state = 5
    availability_time = np.zeros(mission_time)
    reliability_time = np.zeros(mission_time)
    failure_counter = 0
    for _ in range(iterations):
        current_time = 0
        state = 0
        has_failed = False
        weight = 1
        while current_time < mission_time:
            transition_times = [-np.log(np.random.rand()) / abs(A[state][j]) if A[state, j] > 0 else np.inf for j in range(len(A))]
            min_time = min(transition_times)
            next_state = np.argmin(transition_times)
            time_end = min(int(current_time + min_time), mission_time)
            if state != failure_state:
                availability_time[int(current_time):time_end] += 1
                if not has_failed:
                    reliability_time[int(current_time):time_end] += 1
            current_time += min_time
            state = next_state
            if state == failure_state:
                failure_counter += 1
                has_failed = True
    availability = availability_time / iterations
    reliability = reliability_time / iterations
    unavailability = 1 - availability
    unreliability = 1 - reliability
    return availability, unavailability, reliability, unreliability

# Reliability cases
cases = [
    {"lambda_1": 0.1, "mu_1": 0.1, "lambda_s": 0.05, "mu_s": 0.05}, # Case 1: Balanced rates
    {"lambda_1": 0.05, "mu_1": 0.2, "lambda_s": 0.02, "mu_s": 0.1}, # Case 2: High repair rates
    {"lambda_1": 0.2, "mu_1": 0.05, "lambda_s": 0.1, "mu_s": 0.02}  # Case 3: High failure rates
]
mission_time = 1000
iterations = 10000

# Run simulations for each case
results = []
for case in cases:
    availability, unavailability, reliability, unreliability = event_based_rate_forcing_mc_simulation(
        lambda_1=case["lambda_1"],
        mu_1=case["mu_1"],
        lambda_s=case["lambda_s"],
        mu_s=case["mu_s"],
        mission_time=mission_time,
        iterations=iterations
    )
    results.append((availability, unavailability, reliability, unreliability))

# Plot availability and reliability for each case
time_points = np.arange(0, mission_time)
plt.figure(figsize=(8, 6))
for i, (availability, unavailability, reliability, unreliability) in enumerate(results):
    plt.plot(time_points, availability, label=f"Case {i+1}: Availability")
    plt.plot(time_points, reliability, linestyle='--', label=f"Case {i+1}: Reliability")
plt.xlabel("Time (s)")
plt.ylabel("Probability")
plt.legend()
plt.title("Availability and Reliability for Different Cases")
plt.grid()
plt.show()

# Plot unavailability and unreliability for each case
plt.figure(figsize=(8, 6))
for i, (availability, unavailability, reliability, unreliability) in enumerate(results):
    plt.plot(time_points, unavailability, label=f"Case {i+1}: Unavailability")
    plt.plot(time_points, unreliability, linestyle='--', label=f"Case {i+1}: Unreliability")
plt.xlabel("Time (s)")
plt.ylabel("Probability")
plt.legend()
plt.title("Unavailability and Unreliability for Different Cases")
plt.grid()
plt.show()

# Accuracy vs number of iterations
iteration_sizes = [100, 1000, 10000]
accuracies = []
for iterations in iteration_sizes:
    availability, unavailability, reliability, unreliability = event_based_rate_forcing_mc_simulation(
        lambda_1=cases[0]["lambda_1"],
        mu_1=cases[0]["mu_1"],
        lambda_s=cases[0]["lambda_s"],
        mu_s=cases[0]["mu_s"],
        mission_time=mission_time,
        iterations=iterations
    )
    accuracy_estimate = 1 / np.sqrt(iterations)  # Accuracy proportional to 1/sqrt(N)
    accuracies.append(accuracy_estimate)

# Plot accuracy vs number of iterations
plt.figure(figsize=(8, 6))
plt.plot(iteration_sizes, accuracies, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Number of Iterations (N)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of Iterations")
plt.grid()
plt.show()