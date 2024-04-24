import numpy as np
import HFCM as hfcm
def mtx_similar1(arr1:np.ndarray, arr2:np.ndarray) ->float:

    farr1 = arr1.ravel()
    farr2 = arr2.ravel()
    len1 = len(farr1)
    len2 = len(farr2)
    if len1 > len2:
        farr1 = farr1[:len2]
    else:
        farr2 = farr2[:len1]

    numer = np.sum(farr1 * farr2)
    denom = np.sqrt(np.sum(farr1**2) * np.sum(farr2**2))
    similar = numer / denom 
    return  (similar+1) / 2     

def mtx_similar2(arr1:np.ndarray, arr2:np.ndarray) ->float:
   
    if arr1.shape != arr2.shape:
        minx = min(arr1.shape[0],arr2.shape[0])
        miny = min(arr1.shape[1],arr2.shape[1])
        differ = arr1[:minx,:miny] - arr2[:minx,:miny]
    else:
        differ = arr1 - arr2
    numera = np.sum(differ**2)
    denom = np.sum(arr1**2)
    similar = 1 - (numera / denom)
    return similar


def mtx_similar3(arr1:np.ndarray, arr2:np.ndarray) ->float:
    if arr1.shape != arr2.shape:
        minx = min(arr1.shape[0],arr2.shape[0])
        miny = min(arr1.shape[1],arr2.shape[1])
        differ = arr1[:minx,:miny] - arr2[:minx,:miny]
    else:
        differ = arr1 - arr2
    dist = np.linalg.norm(differ, ord='fro')
    len1 = np.linalg.norm(arr1)
    len2 = np.linalg.norm(arr2)     
    denom = (len1 + len2) / 2
    similar = 1 - (dist / denom)
    return similar

def matrix_matrix(arr, brr):
    # return arr.dot(brr.T).diagonal() / ((np.sqrt(np.sum(arr * arr, axis=1))) * np.sqrt(np.sum(brr * brr, axis=1)))
    return np.sum(arr*brr, axis=1) / (np.sqrt(np.sum(arr**2, axis=1)) * np.sqrt(np.sum(brr**2, axis=1)))



def one_order_similar(weight_Mat,t1):

    N = weight_Mat.shape[0]
    similar = np.zeros(shape=(N, N))
    for i in range(N):
        for j in range(N):
           
            if i == j:
                similar[i,j] = 0
            else:
                
                similar[i,j] = mtx_similar1(weight_Mat[i],weight_Mat[j])
               
                if similar[i,j] < t1:
                    similar[i, j] = 0
    row, col = similar.shape
    similar1D = similar.reshape(-1)
   
    similar1D = hfcm.normalize(similar1D, flag='01')
    res_similar = similar1D.reshape((row,col))
    return res_similar


if __name__ == "__main__":
    arr1 = np.array([[1, -2, 3, 7], [-8, 2, 5, 9]])
    arr2 = np.array([[1, -2, 3, 7], [-8, 2, 6, 9]])
    arr3 = np.array([[-2, 3, 7], [2, 7, 9]])
    arr4 = np.array([[4, -2, 3], [-8, 2, 7]])
    print('similar arr1&2:', mtx_similar1(arr1, arr2),
          mtx_similar2(arr1, arr2), mtx_similar3(arr1, arr2), sep=' ')
    print('similar arr2&3:', mtx_similar1(arr2, arr3),
          mtx_similar2(arr2, arr3), mtx_similar3(arr2, arr3), sep=' ')
    print('similar arr2&4:', mtx_similar1(arr2, arr4),
          mtx_similar2(arr2, arr4), mtx_similar3(arr2, arr4), sep=' ')
    print('similar arr4&4:', mtx_similar1(arr4, arr4),
          mtx_similar2(arr4, arr4), mtx_similar3(arr4, arr4), sep=' ')
