import numpy as np
import matplotlib.pyplot as plt

# Values
lambda_1 = 0.01
mu_1 = 0.1
lambda_s = 0.05
mu_s = 0.05

# Matrix
A = [(-lambda_1, lambda_1, 0, 0, 0, 0), (mu_1, -mu_1-lambda_s, lambda_s, 0, 0, 0),
          (0, mu_s, -mu_s-mu_1-lambda_s, mu_1, 0, lambda_s), (mu_s, 0, lambda_1, -mu_s-lambda_1, 0, 0),
          (0, 0, 0, 2*mu_s, -2*mu_s-lambda_1, lambda_1), (0, 0, 2*mu_s, 0, mu_1, -2*mu_s-mu_1)]

failure_state = 5

t = 2000  # time of the mission
delta_t = 1  # Time retrieval
iterations = 2000  # Number of simulation over t

def system_based(bias):
    Av = [[0] * int((t / delta_t)) for _ in range(iterations)]
    Re = [[0] * int((t / delta_t)) for _ in range(iterations)]

    for it in range(iterations):
        t_mission = 0
        i = 0
        first_failure = False
        weight = 1
        while t_mission < t:
            state_time = -np.log(np.random.rand()) / (abs(A[i][i]))
            if i == failure_state:
                first_failure = True
            else:  # Counter over time of availaibility
                ind_start = t_mission // delta_t
                ind_final = min((t_mission + state_time) // delta_t, len(Av[0]))
                for index in range(int(ind_start), int(ind_final)):
                    Av[it][index] += weight
                    if not first_failure:
                        Re[it][index] += weight

            t_mission += state_time

            # State changement
            proba_original = []
            proba = []
            failure_possible = False
            for j in range(len(A[0])):
                if j == i:
                    proba_original.append(0)
                    proba.append(0)

                elif j == failure_state and A[i][j] > 0:
                        proba_original.append(A[i][j] / abs(A[i][i]))
                        proba.append(A[i][j] * bias / abs(A[i][i]))
                        failure_possible = True

                else:
                    proba_original.append(A[i][j] / abs(A[i][i]))
                    proba.append(A[i][j] / abs(A[i][i]))

            proba_original = [p / sum(proba_original) for p in proba_original]
            proba = [p / sum(proba) for p in proba]

            if failure_possible:
                weight *= proba_original[j] / proba[j]

            rand = np.random.rand()
            for j in range(len(A[0])):
                if sum(proba[:j]) < rand <= sum(proba[:j + 1]):
                    i = j
    return Av, Re

Av, Re = system_based(bias=1)
Availability = np.mean(Av, axis=0)
Unavailability = [1 - avail for avail in Availability]
Accuracy_av = np.std(Av, axis=0) / np.sqrt(iterations)
Reliability = np.mean(Re, axis=0)
Unreliability = [1 - relia for relia in Reliability]
Accuracy_re = np.std(Re, axis=0) / np.sqrt(iterations)

Av_bias, Re_bias = system_based(bias=1.5)
Availability_bias = np.mean(Av_bias, axis=0)
Unavailability_bias = [1 - avail for avail in Availability_bias]
Accuracy_av_bias = np.std(Av_bias, axis=0) / np.sqrt(iterations)
Reliability_bias = np.mean(Re_bias, axis=0)
Unreliability_bias = [1 - relia for relia in Reliability_bias]
Accuracy_re_bias = np.std(Re_bias, axis=0) / np.sqrt(iterations)


time_list = [t for t in range(0, int(t), int(delta_t))]

plt.figure("Availability and reliability")

plt.subplot(1, 2, 1)
plt.plot(time_list,  Unavailability, label='Unavailability')
plt.plot(time_list,  Unavailability_bias, label='Unavailability biased')
plt.legend()
plt.xlabel('Time(s)')
plt.title('Unavailability')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(time_list,  Unreliability, label='Unreliability')
plt.plot(time_list,  Unreliability_bias, label='Unreliability bias')
plt.legend()
plt.xlabel('Time(s)')
plt.title('Unreliability')
plt.grid(True)


plt.figure("Std")

plt.subplot(1, 2, 1)
plt.plot(time_list,  Accuracy_av, label='Std unavailability')
plt.plot(time_list,  Accuracy_av_bias, label='Std unavailability bias')
plt.legend()
plt.xlabel('Time(s)')
plt.title('Standard deviation unavailability')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(time_list,  Accuracy_re, label='Std unreliability')
plt.plot(time_list,  Accuracy_re_bias, label='Std unreliability bias')
plt.legend()
plt.xlabel('Time(s)')
plt.title('Standard deviation unreliability')
plt.grid(True)

plt.show()




