import numpy as np
import matplotlib.pyplot as plt


# Values
lambda_1 = [0.1,0.1]
mu_1 = [0.1,0.5]
lambda_s = [0.1,0.1]
mu_s = [0.1,0.5]

failure_state = 5

t = 150  # time of the mission
delta_t = 1
iterations = 10000

a = []
b = []
c = []
d = []

for set in range(len(lambda_s)):
    A = [(-lambda_1[set], lambda_1[set], 0, 0, 0, 0), (mu_1[set], -mu_1[set] - lambda_s[set], lambda_s[set], 0, 0, 0),
         (0, mu_s[set], -mu_s[set] - mu_1[set] - lambda_s[set], mu_1[set], 0, lambda_s[set]), (mu_s[set], 0, lambda_1[set], -mu_s[set] - lambda_1[set], 0, 0),
         (0, 0, 0, 2 * mu_s[set], -2 * mu_s[set] - lambda_1[set], lambda_1[set]), (0, 0, 2 * mu_s[set], 0, mu_1[set], -2 * mu_s[set] - mu_1[set])]

    Av = [[0] * int((t / delta_t)) for _ in range(iterations)]
    Re = [[0] * int((t / delta_t)) for _ in range(iterations)]

    for it in range(iterations):
        print(it)
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

            else:
                # Compute the number of elements of the vector of counter we will increment
                ind_counter_start = t_mission // delta_t  # here, every 1 second
                ind_counter_final = min((t_mission + state_time) // delta_t, len(Av[0]))
                for index in range(int(ind_counter_start), int(ind_counter_final)):
                    Av[it][index] += 1  # Then we increment the availibity for every instant
                    if not first_failure:  # And for reliability we don't if system has already failed
                        Re[it][index] += 1

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

    Availability = np.mean(Av, axis=0)
    Unavailability = [1 - avail for avail in Availability]
    a.append(Unavailability)
    Accuracy_av = np.std(Av, axis=0) / np.sqrt(iterations)
    b.append(Accuracy_av)

    Reliability = np.mean(Re, axis=0)
    Unreliability = [1 - relia for relia in Reliability]
    c.append(Unreliability)
    Accuracy_re = np.std(Re, axis=0) / np.sqrt(iterations)
    d.append(Accuracy_re)

print(a[0][-1],a[1][-1])

time_list = [t for t in range(0, int(t), int(delta_t))]

plt.figure("Availability and reliability")

plt.subplot(1, 2, 1)
plt.plot(time_list,  a[0], label='Unavailability A')
plt.plot(time_list,  a[1], label='Unavailability B')
plt.legend()
plt.xlabel('Time(s)')

plt.subplot(1, 2, 2)
plt.plot(time_list,  c[0], label='Unreliability A')
plt.plot(time_list,  c[1], label='Unreliability B')
plt.legend()
plt.xlabel('Time(s)')

plt.figure("Accuracy")

plt.subplot(1, 2, 1)
plt.plot(time_list,  b[0], label='Accuracy availability A')
plt.plot(time_list,  b[1], label='Accuracy availability B')
plt.legend()
plt.xlabel('Time(s)')

plt.subplot(1, 2, 2)
plt.plot(time_list,  d[0], label='Accuracy reliability A')
plt.plot(time_list,  d[1], label='Accuracy reliability B')
plt.legend()
plt.xlabel('Time(s)')


plt.show()





