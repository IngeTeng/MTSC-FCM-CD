from sklearn import linear_model
import numpy as np

# the trasfer function of FCMS,  tanh is used here.
def transferFunc(x, belta=1, flag='-01'):
    if flag == '-01':
        return np.tanh(x)
    else:
        return 1 / (1 + np.exp(-belta * x))


def reverseFunc(y, belta=1, flag='-01'):
    if flag == '-01':
        if y > 0.99999:
            y = 0.99999
        elif y < -0.99999:
            y = -0.99999
        return np.arctanh(y)
    else:
        if y > 0.999:
            y = 0.999

        elif y < 0.00001:
            y = 0.001
        # elif -0.00001 < y < 0:
        #     y = -0.00001

        x = 1 / belta * np.log(y / (1 - y))
        return x



# form feature matrix from sequence
def create_dataset(seq, belta, Order, current_node):
    Nc, K = seq.shape
    samples = np.zeros(shape=(K, Order * Nc + 2))
    for m in range(Order, K):
        for n_idx in range(Nc):
            for order in range(Order):
                samples[m - Order, n_idx * Order + order + 1] = seq[n_idx, m - 1 - order]
        samples[m - Order, 0] = 1
        samples[m - Order, -1] = reverseFunc(seq[current_node, m], belta)
    return samples

# normalize data set into [0, 1] or [-1, 1]
def normalize(ori_data, flag='01'):
    data = ori_data.copy()
    if len(data.shape) > 1:   # 2-D
        N , K = data.shape
        minV = 0
        maxV = 0
        # print(N)
        # print(N,K)
        for i in range(N):
            # print(i)
            # a = data[i,:]
            minV = np.min(data[i, :])
            maxV = np.max(data[i, :])
            if np.abs(maxV- minV) > 0.00001:
                if flag == '01':   # normalize to [0, 1]
                    data[i, :] = (data[i, :] - minV) / (maxV - minV)
                else:
                    data[i, :] = 2 * (data[i, :] - minV) / (maxV - minV) - 1
        return data
    else:   # 1D
        minV = np.min(data)
        maxV = np.max(data)
        if np.abs(maxV - minV) > 0.00001:
            if flag == '01':  # normalize to [0, 1]
                data = (data - minV) / (maxV - minV)
            else:
                data = 2 * (data - minV) / (maxV - minV) - 1
        return data
#the ridge regression
def HFCM_L2(data , Nc , Order,alpha=1e-24):
    normalize_style = '-01'
    # dataset_copy = data.copy()
    data = normalize(data.T, normalize_style)


    # the ridge regression
    belta = 1
    tol = 1e-20
    clf = linear_model.Ridge(alpha=alpha, fit_intercept=False, tol=tol)
    # clf = linear_model.RidgeCV()
    W_learned = np.zeros(shape=(Nc, Nc * Order + 1))
    for node_solved in range(Nc):
        samples = create_dataset(data, belta, Order, node_solved)
        # use ridge regression
        ridge = clf.fit(samples[:, :-1], samples[:, -1])
        # print("training set score:{:.2f}".format(ridge.score(samples[:, :-1],samples[:, -1])))
        W_learned[node_solved,:]= clf.coef_

    steepness = np.max(np.abs(W_learned), axis=1)
    for i in range(Nc):
        if steepness[i] > 1:
            W_learned[i, :] /= steepness[i]
    # for i in range(W_learned.shape[0]):
    #     if np.abs(W_learned[i]) <= 0.05:
    #         W_learned[i] = 0
    #     else:
    #         W_learned[i] = '%0.8f' % W_learned[i]
    # print(W_learned)
    return W_learned
#Lasso regularization
def HFCM_L1(data , Nc , Order,alpha=1e-24):

    normalize_style = '-01'
    # dataset_copy = data.copy()
    data = normalize(data.T, normalize_style)

    belta = 1
    tol = 1e-10
    clf = linear_model.LassoCV()
    W_learned = np.zeros(shape=(Nc, Nc * Order + 1))
    for node_solved in range(Nc):
        samples = create_dataset(data, belta, Order, node_solved)
        # use ridge regression
        ridge = clf.fit(samples[:, :-1], samples[:, -1])
        print("training set score:{:.2f}".format(ridge.score(samples[:, :-1],samples[:, -1])))
        W_learned[node_solved,:]= clf.coef_

    steepness = np.max(np.abs(W_learned), axis=1)
    for i in range(Nc):
        if steepness[i] > 1:
            W_learned[i, :] /= steepness[i]
    # for i in range(W_learned.shape[0]):
    #     if np.abs(W_learned[i]) <= 0.05:
    #         W_learned[i] = 0
    #     else:
    #         W_learned[i] = '%0.8f' % W_learned[i]
    # print(W_learned)
    return W_learned
