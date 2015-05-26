import cPickle
import numpy as np
import scipy as sp
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier


with open('data/arcene_full.pkl') as fh:
    fulldata = cPickle.load(fh)


feature_length = 10000

fulldata_x = fulldata[:, :-1]
# normalize&
fulldata_x = fulldata_x - fulldata_x.mean(axis=0, keepdims=True)
fulldata_x_norms = np.sqrt((fulldata_x ** 2).sum(axis=0, keepdims=True))
fulldata_x = fulldata_x / (fulldata_x_norms +
                           np.equal(fulldata_x_norms, 0))


y = np.array(fulldata[:, -1], dtype='int32')
X = fulldata_x

# score = np.zeros((feature_length))
# for i in range(feature_length):
#     regression = LogisticRegression()
#     regression.fit(X[:, i:i + 1], y)
#     score[i] = regression.score(X[:, i:i + 1], y)

# plt.plot(np.arange(score.shape[0]), np.sort(score))
# plt.show()

score_train = np.zeros((200))
score_test = np.zeros((200))
for data_length in range(10, 200):
    order = np.arange(200)
    np.random.shuffle(order)

    data = X[order, :]
    label = y[order]
    train = data[:data_length, :]
    train_label = label[:data_length]
    test = data[data_length:, :]
    test_label = label[data_length:]

    reg = LogisticRegression()
    reg.fit(train, train_label)
    score_train[data_length] = reg.score(train, train_label)
    score_test[data_length] = reg.score(test, test_label)

plt.plot(np.arange(score_train.shape[0]), score_train)
plt.plot(np.arange(score_test.shape[0]), score_test)
plt.show()



# model = ExtraTreesClassifier()
# model.fit(X, y)
# # display the relative importance of each attribute
# importance = model.feature_importances_

# plt.plot(np.arange(importance.shape[0]), importance)
# plt.show()

# M = fulldata_x
# print M.shape


# # data = fulldata_x[:data_length, :]
# # M = data.T

# s = linalg.svd(M, full_matrices=1, compute_uv=0)

# ind = np.argsort(s)[::-1]
# s = s[ind]

# # # if we use all of the PCs we can reconstruct the noisy signal perfectly
# # S = np.diag(s)
# # Mhat = np.dot(U, np.dot(S, V.T))
# # print "Using all PCs, MSE = %.6G" % (np.mean((M - Mhat)**2))

# # mse = np.zeros((200))
# # for pc in range(200):
# #     Mhat2 = np.dot(U[:, :pc + 1], np.dot(S[:pc + 1, :pc + 1], V[:, :pc + 1].T))
# #     mse[pc] = np.mean((M - Mhat2) ** 2)

# # plt.plot(np.arange(200), mse)
# # plt.show()

# print s.shape
# variance = np.zeros((data_length))
# tot = s.sum()
# for i in range(data_length):
#     variance[i] = np.sum(s[:i]) / tot

# plt.plot(np.arange(data_length), variance, np.arange(data_length),
#          np.ones((data_length)) * 0.95)
# plt.show()
