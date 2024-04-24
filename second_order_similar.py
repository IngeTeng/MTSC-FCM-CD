
import networkx as nx
# import the GED using the munkres algorithm
import gmatch4py as gm
import numpy as np
import copy
import HFCM as hfcm

def second_order_similar(weight_Mat,t2,t3,hi):
    G = []
    N = weight_Mat.shape[0]
    for i in range(N):
        Mat = weight_Mat[i,:,0:-1] 
 
        g = nx.Graph()
        for row in range(Mat.shape[0]):
            for col in range(Mat.shape[1]):
                # if Mat[row,col] > 0.1 or Mat[row,col] < -0.05:
                if np.abs(Mat[row,col]) > t3:
                    g.add_edge(row ,col)
        # g = nx.from_numpy_matrix(Mat)
        G.append(g)
   
    ged = gm.WeisfeleirLehmanKernel(h=hi)  
    distance = ged.compare(G, None)
    distance = np.array(distance)
    dis = copy.deepcopy(distance)

   
    row, col = dis.shape
   
    dis1D = dis.reshape(-1)
   
    dis1D = hfcm.normalize(dis1D, flag='01')
    
    dis2D = dis1D.reshape((row,col))

  
    res = np.zeros(shape=(row, col))
    for i in range(row):
        for j in range(col):
          
            if i == j:
                res[i,j] = 0
            else:
               
                res[i,j] = 1 - dis2D[i,j]
               
                if res[i,j] < t2:
                    res[i,j] = 0
    return res




if __name__ == "__main__":
  

    import gmatch4py as gm
    import networkx as nx
    import time
    from tqdm import tqdm
    import pandas as pd

    max_ = 100
    size_g = 10
    # l = range(5, max_, 5)
    graphs_all = [nx.random_tree(size_g) for i in range(max_)]
    result_compiled = []
    for size_ in tqdm(range(50, max_, 50)):
        graphs = graphs_all[:size_]
        comparator = None
        # for class_ in [gm.BagOfNodes, gm.WeisfeleirLehmanKernel, gm.GraphEditDistance, gm.GreedyEditDistance, gm.HED,
        #                gm.BP_2, gm.Jaccard, gm.MCS, gm.VertexEdgeOverlap]:
        for class_ in [gm.WeisfeleirLehmanKernel,gm.GreedyEditDistance]:
            deb = time.time()
            if class_ in (gm.GraphEditDistance, gm.BP_2, gm.GreedyEditDistance, gm.HED):
                comparator = class_(1, 1, 1, 1)
            elif class_ == gm.WeisfeleirLehmanKernel:
                comparator = class_(h=2)
            else:
                comparator = class_()
            matrix = comparator.compare(graphs, None)
            print([class_.__name__, size_, time.time() - deb])
            result_compiled.append([class_.__name__, size_, time.time() - deb])

    df = pd.DataFrame(result_compiled, columns="algorithm size_data time_exec_s".split())
    df.to_csv("new_gmatch4py_res_{0}graphs_{1}size.csv".format(max_, size_g))
