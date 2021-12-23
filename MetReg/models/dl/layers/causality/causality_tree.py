import numpy as np
from minepy import MINE
from numpy.lib.function_base import corrcoef
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests


class CausalTree():
    """Class of causality structure.

    Args:
      corr_thresold (float, optional): [description]. Defaults to 0.5.
      mic_thresold (float, optional): [description]. Defaults to 0.5.
      flag (list, optional): [description]. Defaults to [1, 0, 0].
      depth (int, optional): [description]. Defaults to 2.
    Call arguments:
      inputs: A 2D array with shape: `(time, features)`

    """
    def __init__(self, flag=[1, 0, 0], threshold=[0.5, 0.5, None], depth=2):
        self.flag = flag
        self.threshold = threshold
        self.depth = depth     

        self.tree = {}
        self.child = {}

    def __call__(self, inputs):
        self.get_adjacency_matrix(inputs)
        self.adjacency_to_tree(self.adjacency_matrix)
        self.get_child_num()
        self.get_child_input_idx()
        self.get_child_state_idx()
        return self.child

    def get_adjacency_matrix(self, inputs):
        self.num_features = inputs.shape[-1]
        # init
        self.adjacency_matrix = np.zeros((self.num_features, self.num_features))

        # correlation test
        if self.flag[0] == 1: 
            corr_matrix, sign_corr_matrix = self.linear_correlation_test(inputs, self.threshold[0])
            if self.flag[1] == 1: 
                mic_matrix, sign_mic_matrix = self.mic_test(inputs, self.threshold[1])

                # combine correlation test
                sign_corr_matrix = sign_corr_matrix + sign_mic_matrix
                sign_corr_matrix[sign_corr_matrix == 2] = 1
        # causality test
        if self.flag[2] == 1: 
            granger_matrix, sign_granger_matrix = self.linear_granger_test(inputs)
            # 0 and -1 for non causality
            self.adjacency_matrix = sign_corr_matrix + sign_granger_matrix 
            self.adjacency_matrix[self.adjacency_matrix == -1] = 0 
        else:
            self.adjacency_matrix = sign_corr_matrix
        
        for i in range(self.num_features):
            self.adjacency_matrix[i, i] = 0

        # if root node have no causal drivers
        child_root = self.adjacency_matrix[:, -1]
        child_corr_root = corr_matrix[:-1, -1]

        if np.sum(child_root) == 0:
            i = np.argmax(np.abs(child_corr_root))
            self.adjacency_matrix[i, self.num_features-1] = 1
            self.adjacency_matrix[self.num_features-1, i] = 1

        return self.adjacency_matrix

    def linear_correlation_test(self, inputs, threshold=0.5):
        """linear correlation test."""
        # init
        corr_matrix = np.full((self.num_features, self.num_features), np.nan)
        sign_matrix = np.zeros((self.num_features, self.num_features))

        # corr & sign matrix
        for i in range(self.num_features):
            for j in range(self.num_features):
                corr, p = pearsonr(inputs[:, i], inputs[:, j])
                corr_matrix[i, j] = corr

                if corr > threshold and p < 0.05: sign_matrix[i, j] = 1
        return corr_matrix, sign_matrix

    def mic_test(self, inputs, threshold=0.5):
        """max Information-based Nonparametric Exploration."""
        # init
        mic_matrix = np.full((self.num_features, self.num_features), np.nan)
        sign_matrix = np.zeros((self.num_features, self.num_features))

        # mic & sign matrix
        for i in range(self.num_features):
            for j in range(self.num_features):
                if i != j:
                    mine = MINE(alpha=0.6, c=15)
                    mine.compute_score(inputs[:, i], inputs[:, j])
                    mic = mine.mic()
                    mic_matrix[i, j] = mic

                    if mic > threshold: sign_matrix[i, j] = 1
        return mic_matrix, sign_matrix

    def linear_granger_test(self, inputs, max_lag=2):
        """Linear granger causality test."""
        # init
        granger_matrix = np.full((self.num_features, self.num_features), np.nan)
        sign_matrix = np.zeros((self.num_features, self.num_features))

        for i in range(self.num_features):
            for j in range(self.num_features):
                gc = grangercausalitytests(inputs[:, [i, j]], maxlag=max_lag)

                for t in range(max_lag): 
                    p = gc[t+1][0]['ssr_ftest'][1]
                    if p > 0.01: sign_matrix[i, j] = -1
        return granger_matrix, sign_matrix

    def get_causal_driver(self, adjacency_matrix, feature):
        drivers = adjacency_matrix[:,feature]
        idx = np.where(drivers==1)[0]
        return [i for i in idx]
    
    def adjacency_to_tree(self, adjacency_matrix):
        """Change adjacency matrix to tree causality.

        Returns: for example, 
            tree: {'1': [[-1]],
                   '2': [[2, 3, 4]],
                   '3': [[3, 4, 5], [1,2], [3,4]]}
        """
        #self.adjacency_matrix[5,-1] = 1 # turn precipitation to layer 2.
        #self.adjacency_matrix[10,-1] = 1 # turn precipitation to layer 2.

        self.tree['1'] = [[self.num_features-1]] # set root node
        
        # add causal drivers for each layer
        for level in np.arange(1, self.depth+1):
            self.tree[str(level+1)] = []
            for node_group in self.tree[str(level)]:
                for node in node_group:
                    self.tree[str(level+1)].append(self.get_causal_driver(adjacency_matrix, node))

    def get_child_num(self):
        """the num of children nodes for each nodes"""
        child_num = []
        # the top layers of causality structure
        for node_group in self.tree[str(self.depth+1)]:
            for _ in node_group:
                child_num.append(0)

        # the rest layers of causality structure
        for level in np.arange(self.depth, 0, -1):
            for node_group in self.tree[str(level+1)]:
                child_num.append(len(node_group))

        self.child['child_num'] = child_num

    def get_child_input_idx(self):
        child_input_idx = []
        
        # input index of nodes in each layers
        for level in np.arange(self.depth+1,0,-1):
            for node_group in self.tree[str(level)]:
                for node in node_group:
                    child_input_idx.append([int(node)])   
        self.child['child_input_idx'] = child_input_idx

    def get_child_state_idx(self):
        child_state_idx = []

        for node_group in self.tree[str(self.depth+1)]:
            for node in node_group:
                child_state_idx.append([])
                
        count = -1
        
        for level in np.arange(self.depth,0,-1):
            for node_group in self.tree[str(level+1)]:
                _idx = []
                for node in node_group:
                    count += 1
                    _idx.append(count)
                
                child_state_idx.append(_idx)    

        self.child['child_state_idx'] = child_state_idx