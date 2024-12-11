import numpy as np
import matplotlib.pyplot as plt

# Define function to simulate event-based Monte Carlo
def event_based_mc_simulation(lambda_1, mu_1, lambda_s, mu_s, mission_time, iterations):
    # Transition rate matrix A
    A = np.array([
        [-lambda_1, lambda_1, 0, 0, 0, 0],
        [mu_1, -mu_1-lambda_s, lambda_s, 0, 0, 0],
        [0, mu_s, -mu_s-mu_1-lambda_s, mu_1, 0, lambda_s],
        [mu_s, 0, lambda_1, -mu_s-lambda_1, 0, 0],
        [0, 0, 0, 2*mu_s, -2*mu_s-lambda_1, lambda_1],
        [0, 0, 2*mu_s, 0, mu_1, -2*mu_s-mu_1]
    ])
    failure_state = 5  # Index of the failure state

    # Initialize counters
    availability_time = np.zeros(mission_time)
    failure_counter = 0
    unreliability_counter = 0
    # Event-based Monte Carlo simulation
    for _ in range(iterations):
        current_time = 0
        state = 0  # Start in the operational state
        failure_occurred = False
        while current_time < mission_time:
            # Sample transition times for all possible transitions
            transition_times = [
                -np.log(np.random.rand()) / abs(A[state][j]) if A[state, j] > 0 else np.inf
                for j in range(len(A))
            ]
            # Determine the next transition
            min_time = min(transition_times)
            next_state = np.argmin(transition_times)
            # Update availability time
            if state != failure_state:
                time_end = min(current_time + min_time, mission_time)
                availability_time[int(current_time):int(time_end)] += 1
            # Update time and state
            current_time += min_time
            state = next_state
            # Stop simulation if failure state is reached
            if state == failure_state:
                failure_counter += 1
                failure_occurred = True
                break
        # Track unreliability
        if failure_occurred:
            unreliability_counter += 1
    # Calculate availability, unavailability, and unreliability
    availability = availability_time / iterations
    unavailability = 1 - availability
    unreliability = unreliability_counter / iterations
    return availability, unavailability, unreliability

# Reliability cases
cases = [
    {"lambda_1": 0.1, "mu_1": 0.1, "lambda_s": 0.05, "mu_s": 0.05},  # Case 1: Balanced rates
    {"lambda_1": 0.05, "mu_1": 0.2, "lambda_s": 0.02, "mu_s": 0.1},  # Case 2: High repair rates
    {"lambda_1": 0.2, "mu_1": 0.05, "lambda_s": 0.1, "mu_s": 0.02}   # Case 3: High failure rates
]
mission_time = 1000
iterations = 10000

# Simulate for reliability cases
results = []
for case in cases:
    availability, unavailability, unreliability = event_based_mc_simulation(
        lambda_1=case["lambda_1"],
        mu_1=case["mu_1"],
        lambda_s=case["lambda_s"],
        mu_s=case["mu_s"],
        mission_time=mission_time,
        iterations=iterations
    )
    results.append((availability, unavailability, unreliability))

# Plot availability and unavailability
time_points = np.arange(0, mission_time)
plt.figure(figsize=(14, 8))
for i, (availability, unavailability, unreliability) in enumerate(results):
    # Plot availability
    plt.plot(time_points, availability, label=f"Case {i+1}: Availability")
    # Plot unavailability
    plt.plot(time_points, unavailability, linestyle='--', label=f"Case {i+1}: Unavailability")
    # Print numerical results
    print(f"Case {i+1} - Unreliability: {unreliability:.4f}")
    print(f"Case {i+1} - Final Availability: {availability[-1]:.4f}")
    print(f"Case {i+1} - Final Unavailability: {unavailability[-1]:.4f}")
plt.xlabel("Time")
plt.ylabel("Probability")
plt.legend()
plt.title("Availability and Unavailability for Reliability Cases")
plt.grid()
plt.show()

# Accuracy vs Iterations
iteration_sizes = [100, 1000, 10000, 100000]
accuracies = []
for iterations in iteration_sizes:
    availability, unavailability, unreliability = event_based_mc_simulation(
        lambda_1=cases[0]["lambda_1"], # Use the first case for accuracy analysis
        mu_1=cases[0]["mu_1"],
        lambda_s=cases[0]["lambda_s"],
        mu_s=cases[0]["mu_s"],
        mission_time=mission_time,
        iterations=iterations
    )
    accuracy_estimate = 1 / np.sqrt(iterations)  # Compute accuracy
    accuracies.append(accuracy_estimate)
# Plot accuracy vs number of iterations
plt.figure(figsize=(12, 6))
plt.plot(iteration_sizes, accuracies, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Number of Iterations (N)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of Iterations")
plt.grid()
plt.legend()
plt.show()