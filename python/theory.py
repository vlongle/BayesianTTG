import numpy as np
import matplotlib.pyplot as plt

t = 100

N = 6
Ws = np.array([1,2,3,4,5,6])
Ms = np.array([1] * N)


def sim(Ms, seed):
    np.random.seed(seed)
    res = Ms.copy()
    for _ in range(t):
        chosen = np.random.choice(range(N), size=N//2)
        not_chosen = [i for i in range(N) if i not in chosen]
        res[chosen] = res[chosen] + sum(Ws[chosen]) * (res[chosen])/(sum(res[chosen]))
        res[not_chosen] += Ws[not_chosen]

    return res


i = 0
j = 1

trials = 1
desirable = 0
for seed in range(trials):
    ret = sim(Ms, seed)
    print('ret:', ret)
    if (ret[j] - ret[i] > 0):
        desirable += 1

print('prob successs:', desirable/trials)
