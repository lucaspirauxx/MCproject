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
    reliability_time = np.zeros(mission_time)
    failure_counter = 0
    # Event-based Monte Carlo simulation
    for _ in range(iterations):
        current_time = 0
        state = 0  # Start in the operational state
        has_failed = False
        while current_time < mission_time:
            # Sample transition times for all possible transitions
            transition_times = [-np.log(np.random.rand()) / abs(A[state][j]) if A[state, j] > 0 else np.inf for j in range(len(A))]
            # Determine the next transition
            min_time = min(transition_times)
            next_state = np.argmin(transition_times)
            time_end = min(int(current_time + min_time), mission_time)
            # Update availability (always operational if not in failure state)
            if state != failure_state:
                availability_time[int(current_time):time_end] += 1
                if not has_failed:
                    # Update reliability only if no failure has occurred
                    reliability_time[int(current_time):time_end] += 1
            # Advance time and state
            current_time += min_time
            state = next_state
            # Mark failure if we enter the failure state
            if state == failure_state:
                failure_counter += 1
                has_failed = True
    # Normalize counters to calculate probabilities
    availability = availability_time / iterations
    reliability = reliability_time / iterations
    unavailability = 1 - availability
    unreliability = 1 - reliability
    return availability, unavailability, reliability, unreliability

# Reliability cases
cases = [
    {"lambda_1": 0.1, "mu_1": 0.1, "lambda_s": 0.05, "mu_s": 0.05}, # Case 1: Balanced rates (Ratio lambda/mu = 1)
    {"lambda_1": 0.05, "mu_1": 0.2, "lambda_s": 0.02, "mu_s": 0.1}, # Case 2: High repair rates (Ratio lambda/mu < 1)
    {"lambda_1": 0.2, "mu_1": 0.05, "lambda_s": 0.1, "mu_s": 0.02}  # Case 3: High failure rates (Ratio lambda/mu > 1)
]
mission_time = 1000
iterations = 10000

# Simulate for reliability cases
results = []
for case in cases:
    availability, unavailability, reliability, unreliability = event_based_mc_simulation(
        lambda_1=case["lambda_1"],
        mu_1=case["mu_1"],
        lambda_s=case["lambda_s"],
        mu_s=case["mu_s"],
        mission_time=mission_time,
        iterations=iterations
    )
    results.append((availability, unavailability, reliability, unreliability))

# Plot unavailability and unreliability
time_points = np.arange(0, mission_time)
plt.figure(figsize=(8, 6))
for i, (availability, unavailability, reliability, unreliability) in enumerate(results):
    # Plot unavailability
    plt.plot(time_points, unavailability, label=f"Case {i+1}: Unavailability", linestyle='-')
    # Plot unreliability
    plt.plot(time_points, unreliability, label=f"Case {i+1}: Unreliability", linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Probability")
plt.legend()
plt.title("Unavailability and Unreliability for Reliability Cases")
plt.grid()
plt.show()

# Plot availability and reliability
plt.figure(figsize=(8, 6))
for i, (availability, unavailability, reliability, unreliability) in enumerate(results):
    # Plot availability
    plt.plot(time_points, availability, label=f"Case {i+1}: Availability", linestyle='-')
    # Plot reliability
    plt.plot(time_points, reliability, label=f"Case {i+1}: Reliability", linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Probability")
plt.legend()
plt.title("Availability and Reliability for Reliability Cases")
plt.grid()
plt.show()

# Accuracy vs Iterations
iteration_sizes = [100, 1000, 10000]
accuracies = []
for iterations in iteration_sizes:
    availability, unavailability, reliability, unreliability = event_based_mc_simulation(
        lambda_1=cases[0]["lambda_1"], # Use the first case for accuracy analysis
        mu_1=cases[0]["mu_1"],
        lambda_s=cases[0]["lambda_s"],
        mu_s=cases[0]["mu_s"],
        mission_time=mission_time,
        iterations=iterations
    )
    accuracy_estimate = 1 / np.sqrt(iterations) # Compute accuracy
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
plt.legend()
plt.show()