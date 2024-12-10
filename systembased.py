import numpy as np

# Values
lambda_1 = 0.1
mu_1 = 0.1
lambda_s = 0.1
mu_s = 0.1

# Matrix
A = [(-lambda_1, lambda_1, 0, 0, 0, 0), (mu_1, -mu_1-lambda_s, lambda_s, 0, 0, 0),
          (0, mu_s, -mu_s-mu_1-lambda_s, mu_1, 0, lambda_s), (mu_s, 0, lambda_1, -mu_s-lambda_1, 0, 0),
          (0, 0, 0, 2*mu_s, -2*mu_s-lambda_1, lambda_1, (0, 0, 2*mu_s, 0, mu_1, -2*mu_s-mu_1))]

i = 0
t = 100
t_sejourn = 0
while t_sejourn < t:
    t_sejourn += -np.log(np.random.rand())/abs(A[i][i])
    proba = []
    for j in range(len(A[0])):
        if j == i:
            proba.append(0)
        if j != i:
            proba.append(A[i][j]/abs(A[i][i]))
    print(proba)
    rand = np.random.rand()
    for j in range(len(A[0])):
        if sum(proba[:j]) < rand <= sum(proba[:j+1]):
            i = j
            print(i)


