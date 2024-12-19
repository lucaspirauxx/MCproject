import numpy as np
import matplotlib.pyplot as plt

# Values
lambda_1 = 0.1
mu_1 = 0.1
lambda_s = 0.05
mu_s = 0.05

# Matrix
A = [(-lambda_1, lambda_1, 0, 0, 0, 0), (mu_1, -mu_1-lambda_s, lambda_s, 0, 0, 0),
          (0, mu_s, -mu_s-mu_1-lambda_s, mu_1, 0, lambda_s), (mu_s, 0, lambda_1, -mu_s-lambda_1, 0, 0),
          (0, 0, 0, 2*mu_s, -2*mu_s-lambda_1, lambda_1), (0, 0, 2*mu_s, 0, mu_1, -2*mu_s-mu_1)]

failure_state = 5

t = 1000  # time of the mission
t_max = 10
delta_t = 1  # Time retrieval
iterations = 10000  # Number of simulation over t


def system_based(time_limit):
    Av = [[0] * int((t / delta_t)) for _ in range(iterations)]
    Re = [[0] * int((t / delta_t)) for _ in range(iterations)]

    for it in range(iterations):
        t_mission = 0
        i = 0
        weight = 1
        first_failure = False

        while t_mission < t:
            state_time = -np.log(-np.random.rand()*(1-np.exp(-abs(A[i][i])*time_limit))+1) / abs(A[i][i])

            weight *= (1 - np.exp(-abs(A[i][i]) * time_limit))

            if weight < 0.02:
                if np.random.rand() > 0.02:
                    break
                else:
                    weight /= 0.02

            ind_start = t_mission // delta_t
            ind_end = min((t_mission + state_time) // delta_t, len(Av[0]))
            for index in range(int(ind_start), int(ind_end)):
                Av[it][index] += weight
                if not first_failure:
                    Re[it][index] += weight

            if i == failure_state:
                first_failure = True

            t_mission += state_time

            # Transition vers un nouvel état
            proba = [A[i][j] / abs(A[i][i]) if j != i else 0 for j in range(len(A))]
            rand = np.random.rand()
            for j in range(len(A)):
                if sum(proba[:j]) < rand <= sum(proba[:j + 1]):
                    i = j
                    break
    return Av, Re

Av, Re = system_based(time_limit=30)
Availability = np.mean(Av, axis=0)
Unavailability = [1 - avail for avail in Availability]
Accuracy_av = np.std(Av, axis=0) / np.sqrt(iterations)
Reliability = np.mean(Re, axis=0)
Unreliability = [1 - relia for relia in Reliability]
Accuracy_re = np.std(Re, axis=0) / np.sqrt(iterations)

Av_bias, Re_bias = system_based(time_limit=100)
Availability_bias = np.mean(Av_bias, axis=0)
Unavailability_bias = [1 - avail for avail in Availability_bias]
Accuracy_av_bias = np.std(Av_bias, axis=0) / np.sqrt(iterations)
Reliability_bias = np.mean(Re_bias, axis=0)
Unreliability_bias = [1 - relia for relia in Reliability_bias]
Accuracy_re_bias = np.std(Re_bias, axis=0) / np.sqrt(iterations)

time_list = [t for t in range(0, int(t), int(delta_t))]

plt.figure("Availability and reliability")

plt.subplot(1, 2, 1)
plt.plot(time_list,  Unavailability, label='Unavailability 30s')
plt.plot(time_list,  Unavailability_bias, label='Unavailability 100s')
plt.legend()
plt.xlabel('Time(s)')
plt.title('Unavailability')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(time_list,  Unreliability, label='Unreliability 30s')
plt.plot(time_list,  Unreliability_bias, label='Unreliability 100s')
plt.legend()
plt.xlabel('Time(s)')
plt.title('Unreliability')
plt.grid(True)


plt.figure("Std")

plt.subplot(1, 2, 1)
plt.plot(time_list,  Accuracy_av, label='Std unavailability 30s')
plt.plot(time_list,  Accuracy_av_bias, label='Std unavailability 100s')
plt.legend()
plt.xlabel('Time(s)')
plt.title('Standard deviation unavailability')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(time_list,  Accuracy_re, label='Std unreliability 30s')
plt.plot(time_list,  Accuracy_re_bias, label='Std unreliability 100s')
plt.legend()
plt.xlabel('Time(s)')
plt.title('Standard deviation unreliability')
plt.grid(True)

plt.show()