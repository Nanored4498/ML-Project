from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import zero_one_loss
import numpy as np

f = open("kernel.txt", "r")

def load_set():
	vals = f.readline().split()
	y = list(map(int, vals))
	n = len(y)
	K = []
	for _ in range(n):
		K.append(list(map(float, f.readline().split())))
	return - np.array(K), np.array(y)

K_train, y_train = load_set()
K_test, y_test = load_set()

svc = SVC(kernel='precomputed')

svc.fit(K_train, y_train)

y_pred = svc.predict(K_test)
print(y_test)
print(y_pred)
print(zero_one_loss(y_test, y_pred))