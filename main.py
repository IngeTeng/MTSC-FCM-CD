#mts_HFCM_GraphMatch_Louvain
import numpy as np
import HFCM as hfcm
import common as cn
import one_order_similar as OneOrder
import second_order_similar as SecondOrder
import construct_network2 as con_nx2
import community as community_louvain
from sklearn import metrics
from scipy.io import loadmat
from sklearn.metrics import f1_score  

from sklearn.metrics import accuracy_score


def rand_index(y_true, y_pred):
    n = len(y_true)
    a, b = 0, 0
    for i in range(n):
        for j in range(i+1, n):
            if (y_true[i] == y_true[j]) & (y_pred[i] == y_pred[j]):
                a +=1
            elif (y_true[i] != y_true[j]) & (y_pred[i] != y_pred[j]):
                b +=1
            else:
                pass
    RI = (a + b) / (n*(n-1)/2)
    return RI

def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    if isinstance(y_true,list):
        y_true = np.array(y_true)
    if isinstance(y_pred,list):
        y_pred = np.array(y_pred)
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

if __name__ == "__main__":


    type = 1 #0-.mat,1-.arr,2-.data


    a_list = [0.5]
    T = 200
    t = 1/T
    t1 = t2 = 0.05
    t3 = 0.05
    Order = 1
    hi = 4
    # datasetname = 'PEMS'
    # datasetname = 'UWave'
    # datasetname = 'Libras'
    # datasetname = 'JapaneseVowels'
    # datasetname = 'lp5'
    # datasetname = 'EigenWorms'
    # datasetname = 'ArabicDigits'
    # datasetname = 'AtrialFibrillation'
    # datasetname = 'BasicMotions'
    # datasetname = 'DuckDuckGeese'
    # datasetname = 'Epilepsy'
    # datasetname = 'EthanolConcentration'
    # datasetname = 'FingerMovements'
    datasetname = 'HandMovementDirection'
    # datasetname = 'Heartbeat'
    # datasetname = 'MotorImagery'
    # datasetname = 'PhonemeSpectra'
    # datasetname = 'RacketSports'
    # datasetname = 'StandWalkJump'
    # datasetname = "CharacterTrajectories"
    # datasetname = 'CMUsubject16'
    # datasetname = 'Wafer'
    # datasetname = "AUSLAN"

    # datasetname = "ArticularyWordRecognition"
    # datasetname = "Cricket"
    # datasetname = "EigenWorms"
    # datasetname = "ERing"
    # datasetname = "Handwriting"
    # datasetname = "LSST"
    # datasetname = "NATOPS"
    # datasetname = "PEMS-SF"
    # datasetname = "UWaveGestureLibrary"
    # datasetname = "SpokenArabicDigits"
    # datasetname = "PenDigits"

    if type == 0:
    	#############mat data###################
        path = "datasets/mtsdata/" + datasetname + "/" + datasetname + ".mat"
        data = loadmat(path)
        # a = data['mts']
        y_train_tmp = data['mts']['trainlabels'][0, 0]
        x_train_tmp = data['mts']['train'][0, 0]
        y_test_tmp = data['mts']['testlabels'][0, 0]
        x_test_tmp = data['mts']['test'][0, 0]

        # label
        labels_true = np.concatenate([y_train_tmp.T[0] - 1, y_test_tmp.T[0] - 1]).tolist()

        # data
        max_len = 0
        # x_data = np.concatenate([x_train_tmp[0], x_test_tmp[0]])
        x_data = []
        for i in range(x_train_tmp[0].shape[0]):
            size = x_train_tmp[0][i].shape[1]
            if size > max_len:
                max_len = size
            x_data.append(x_train_tmp[0][i].T)

        for i in range(x_test_tmp[0].shape[0]):
            size = x_test_tmp[0][i].shape[1]
            if size > max_len:
                max_len = size
            x_data.append(x_test_tmp[0][i].T)

        # (640, 29, 12)
        dataset = np.zeros(
            shape=(x_train_tmp[0].shape[0] + x_test_tmp[0].shape[0], max_len, x_train_tmp[0][0].shape[0]))
        for i in range(x_train_tmp[0].shape[0] + x_test_tmp[0].shape[0]):
            for j in range(max_len):
                for k in range(x_train_tmp[0][0].shape[0]):
                    if j < x_data[i].shape[0]:
                        dataset[i, j, k] = x_data[i][j, k]

    elif type == 1:
        ############arr data################
        path = "datasets/" + datasetname + "/" + datasetname + ".npz"
        data = np.load(path)
        dataset = np.concatenate([data['x_train'], data['x_test']])
        labels_true = np.concatenate([data['y_train'], data['y_test']]).tolist()
        print(
            "data number is {} , maxlen is {} , dim is {}".format(dataset.shape[0], dataset.shape[1], dataset.shape[2]))

    elif type == 2:
        ######### .data#############
        data_path = "datasets/PLCSV/" + datasetname + "/" + datasetname + "data.npy"
        label_path = "datasets/PLCSV/" + datasetname + "/" + datasetname + "label.npy"
        dataset = np.load(data_path)
        labels_true = np.load(label_path)
        print(
            "data number is {} , maxlen is {} , dim is {}".format(dataset.shape[0], dataset.shape[1], dataset.shape[2]))

    ##########HFCM###############
    #eachMTS，train an FCM
    W_HFCM = []
    for i in range(dataset.shape[0]):
        non_zero_data =cn.no_zero(dataset[i])

        #learn FCM
        Nc = non_zero_data.shape[1]
        W_learned = hfcm.HFCM_L2(non_zero_data, Nc, Order, alpha=1e-24)
        W_HFCM.append(W_learned)

    W_HFCM = np.array(W_HFCM)
    print('HFCM finish')


    # ##########k-means#############
    # X = np.zeros(shape=(W_HFCM.shape[0], W_HFCM.shape[1]*W_HFCM.shape[2]))
    # for i in range(W_HFCM.shape[0]):
    #     X[i] = W_HFCM[i].reshape(-1)
    # clf = KMeans(n_clusters=9)
    # clf.fit(X) 
    # centers = clf.cluster_centers_ 
    # pred_label = clf.labels_ 
    # print(centers)
    # print(labels)

    ############calulate smilarity###############
  
    
    N = W_HFCM.shape[0]
    similar_1rd = np.zeros(shape=(N, N))
    similar_2rd = np.zeros(shape=(N, N))
    #S_1nd
    similar_1rd = OneOrder.one_order_similar(W_HFCM,t1)
    #S_2st
    similar_2rd = SecondOrder.second_order_similar(W_HFCM,t2,t3,hi)
    print('similar_2rd:',similar_2rd)

    for a in a_list:
        print('a:',a)
        similar = -(a * similar_1rd + (1-a) * similar_2rd)
        # similar = similar_2rd
        print('similar finish')

        ##############build relational network##################
        g_network = con_nx2.construct_network(similar,t)
        print('construct_network finish')

        #############Louvain#############
        partition = community_louvain.best_partition(g_network)
        pred_label = cn.keys_only(partition)
        print('community_louvain finish')
        cluster_num = len(list(set(pred_label)))
        print("cluster_num:",cluster_num)
    
        NMI =  metrics.normalized_mutual_info_score(labels_true, pred_label)
        RI = rand_index(labels_true, pred_label)
        purity = purity_score(labels_true, pred_label)
        print("RI:",RI)
        print("purity:",purity)
        print("NMI:",NMI)
       