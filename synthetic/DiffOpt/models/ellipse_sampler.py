import numpy as np 
import random 
# from trainer import Branin
from bayeso_benchmarks.two_dim_branin import Branin as BraninFunction

np.random.seed(42)


def branin_function(x1, x2):
        a = 1.0
        b = 5.1 / (4 * np.pi ** 2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        return -a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 - s * (1 - t) * np.cos(x1) + s

X1 = np.random.uniform(low=-5, high=5, size=(10000))
X2 = np.random.uniform(low=0, high=15, size=(10000))
print(X1.max(), X1.min())
print(X2.max(), X2.min())
X = np.column_stack((X1, X2))
print(X.shape, X[:,0].max(), X[0].min(), X[1].max(), X[1].min())


h, k = -0.2, 7.5  # coordinates of center
a, b = 3.6, 8.0  # length of axis
theta = np.radians(25)  # rotation angel

X_inside = []
for x, y in X:
    if (((x - h) * np.cos(theta) + (y - k) * np.sin(theta)) ** 2 / a**2 + ((x - h) * np.sin(theta) - (y - k) * np.cos(theta)) ** 2 / b**2) <= 1:
        X_inside.append(np.array([x, y]))
X = np.array(X_inside)
branin = BraninFunction()
Y = branin.output(X)
print(Y.shape, Y.max(), Y.min())


import pickle
pickle.dump([X, Y], open("../dataset/ellipse_branin_10000_new.p", "wb"))
