import numpy as np

# Values
lambda_1 = 0.1
mu_1 = 0.1
lambda_s = 0.05
mu_s = 0.05

# Matrix
A = [(-lambda_1, lambda_1, 0, 0, 0, 0), (mu_1, -mu_1-lambda_s, lambda_s, 0, 0, 0),
          (0, mu_s, -mu_s-mu_1-lambda_s, mu_1, 0, lambda_s), (mu_s, 0, lambda_1, -mu_s-lambda_1, 0, 0),
          (0, 0, 0, 2*mu_s, -2*mu_s-lambda_1, lambda_1), (0, 0, 2*mu_s, 0, mu_1, -2*mu_s-mu_1)]


t = 100
i_t_mission = 0
iterations = 100
for _ in range(iterations):
    t_mission = 0
    i = 0
    while t_mission < t:
        #print('Etat :', i)
        t_mission += -np.log(np.random.rand()) / abs(A[i][i])
        proba = []
        for j in range(len(A[0])):
            if j == i:
                proba.append(0)
            if j != i:
                proba.append(A[i][j] / abs(A[i][i]))
        #print('Probas :', proba)
        rand = np.random.rand()
        for j in range(len(A[0])):
            if sum(proba[:j]) < rand <= sum(proba[:j + 1]):
                i = j
    #print('Etat final :', i)

    if i != 5:
        i_t_mission += 1
print('A(t_mission) =',i_t_mission/iterations)





