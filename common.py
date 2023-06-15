import numpy as np

def no_zero(data):
    data_no_zero = []
    for i in range(data.shape[0]):
        if data[i,0] != 0 or data[i,1] !=0:
            data_no_zero.append(data[i])
    data_no_zero = np.array(data_no_zero)
    return data_no_zero

def keys_only(flat_dict):
    lst = []
    for k, v in flat_dict.items():
        lst.append(v)
    return lst