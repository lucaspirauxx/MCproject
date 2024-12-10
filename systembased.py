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

t = 100  # time of the mission
delta_t = 1
i_t_mission = 0  # Counter of state at the end
counter = [0]*int((t/delta_t))  # Every element is for every delta time

iterations = 1000
for _ in range(iterations):
    t_mission = 0
    i = 0
    while t_mission < t:
        state_time = -np.log(np.random.rand()) / abs(A[i][i])

        if i != failure_state:  # Counter over time of availaibility
            ind_counter_start = t_mission // delta_t
            ind_counter_final = min((t_mission + state_time) // delta_t, len(counter))
            for index in range(int(ind_counter_start), int(ind_counter_final)):
                counter[index] += 1

        t_mission += state_time

        proba = []
        for j in range(len(A[0])):
            if j == i:
                proba.append(0)
            if j != i:
                proba.append(A[i][j] / abs(A[i][i]))
        rand = np.random.rand()
        for j in range(len(A[0])):
            if sum(proba[:j]) < rand <= sum(proba[:j + 1]):
                i = j

    if i != failure_state:
        i_t_mission += 1

print('A(t_mission) =',i_t_mission/iterations)
A_t = [count / iterations for count in counter]
print(A_t)

plt.figure(figsize=(12, 6))

time_list = [t for t in range(0, int(t), int(delta_t))]
#plt.subplot(1, 2, 1)
plt.plot(time_list,  A_t, label='Availibility')
#plt.plot(thickness_plot,  transmission_probability_1, label='Double random', color='red')
plt.legend()
plt.xlabel('Time(s)')
plt.ylabel('Availaibility')

plt.show()





