from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import zero_one_loss
from sklearn.decomposition import PCA
import numpy as np
import pylab as pl

y_train = np.array(list(map(int, input().split())))
n_train = len(y_train)
D_train = []
for _ in range(n_train):
	D_train.append(list(map(float, input().split())))
D_train = np.array(D_train)
y_test = np.array(list(map(int, input().split())))
n_test = len(y_test)
D_test = []
for _ in range(n_test):
	D_test.append(list(map(float, input().split())))
D_test = np.array(D_test)
mi = min(D_train.flatten())
D_train = D_train - mi
D_test = D_test - mi

ns = np.array([0]*10)
for i in range(n_train):
	ns[y_train[i]] += 1
print(ns)

def gauss(D, t):
	return np.exp(- D**0.7 / t)

L = []
for t in np.arange(3, 13, 0.5):
	for C in np.arange(4.5, 11, 0.5):
		K_train = gauss(D_train, t)
		K_test = gauss(D_test, t)
		svc = SVC(kernel='precomputed', C=C).fit(K_train, y_train)
		L.append((t, C, svc.score(K_test, y_test), len(svc.support_)))
ms = max(a[2] for a in L)
mt = 0
for t, C, s, ls in L:
	if s + 1.5 / n_test > ms:
		print(round(t, 4), C, ls, round(s, 4))
	if s == ms:
		mt = t

su = np.array([[0.0]*n_train for _ in range(10)])
K = gauss(D_train, mt)
for i in range(n_train):
	su[y_train[i]] += K[i]
su = (su.T / ns).T
pca = PCA(n_components=2).fit(su).transform(K)
figs = [[] for _ in range(10)]
for i in range(n_train):
	figs[y_train[i]].append(i)
inds = []
for i in range(10):
	xp, yp = pca[figs[i]].T
	pl.scatter(xp, yp, label=str(i))
	inds += figs[i]
inds = np.array(inds).reshape(n_train, 1)
K2 = K[inds, inds.T]
pl.legend()
pl.show()
pl.figure(figsize=(10, 10))
pl.xlim(0, n_train)
pl.ylim(n_train, 0)
pl.matshow(K2, 0)
sn = 0
for i in range(9):
	sn += ns[i]
	pl.plot([0, n_train], [sn, sn], 'r', linewidth=0.6)
	pl.plot([sn, sn], [0, n_train], 'r', linewidth=0.6)
pl.show()

# print(len(svc.support_))
# for i in range(len(K_train)):
# 	if i not in svc.support_:
# 		for j in range(len(K_test)):
# 			K_test[j][i] = 23232

# y_pred = svc.predict(K_test)
# print(y_test)
# print(y_pred)
# print(zero_one_loss(y_test, y_pred))