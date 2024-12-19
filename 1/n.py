import numpy as np
import matplotlib.pyplot as plt


# Values
lambda_1 = 0.1
mu_1 = 0.1
lambda_s = 0.1
mu_s = 0.1

failure_state = 5

t = 100  # time of the mission
delta_t = 1
iterations = 500
n = list(range(10, iterations))

a = []

A = [(-lambda_1, lambda_1, 0, 0, 0, 0), (mu_1, -mu_1-lambda_s, lambda_s, 0, 0, 0),
          (0, mu_s, -mu_s-mu_1-lambda_s, mu_1, 0, lambda_s), (mu_s, 0, lambda_1, -mu_s-lambda_1, 0, 0),
          (0, 0, 0, 2*mu_s, -2*mu_s-lambda_1, lambda_1), (0, 0, 2*mu_s, 0, mu_1, -2*mu_s-mu_1)]

for ite in range(10, iterations):
    print(ite)
    a_fin = []
    for it in range(ite):
        # t_mission will be incremented at every sample of the pdf
        t_mission = 0
        i = 0  # State of the system (5 is failure)
        first_failure = False  # To get the reliability, here the system works a t=0
        while t_mission < t:  # T is the time of the mission stop
            # Sample of pdf
            state_time = -np.log(np.random.rand()) / abs(A[i][i])

            if i == failure_state:
                # We save the first failure moment, and don't increment the availibility
                first_failure = True

            t_mission += state_time

            # State changement
            proba = []  # List of the proba of transitions
            for j in range(len(A[0])):
                if j == i:  # Of course proba to stay is 0 if element broke
                    proba.append(0)
                if j != i:  # Else proba = a(ij)/abs(a(ii))
                    proba.append(A[i][j] / abs(A[i][i]))
            # Then sample of discrete distribution
            rand = np.random.rand()
            for j in range(len(A[0])):
                if sum(proba[:j]) < rand <= sum(proba[:j + 1]):
                    i = j
        if i == failure_state:
            a_fin.append(1)
        else:
            a_fin.append(0)
    a.append(np.std(a_fin)/np.sqrt(it))


plt.figure("Std")

plt.plot(n,  a)
plt.xlabel('Iterations')
plt.title('Standard deviation')
plt.grid(True)

plt.show()





