#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import itertools as it
import random
import time
import warnings
from collections import defaultdict
from copy import deepcopy
from math import log

import graphviz
import lingam
import numpy as np
import pandas as pd
import scipy.spatial as ss
from scipy.special import digamma
from scipy.stats.stats import kendalltau, pearsonr, spearmanr
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold


def timer(func):
    def wrapper(*args,**kwds):
        t0 = time.time()
        func(*args,**kwds)
        t1 = time.time()
        print(t1-t0)
    return wrapper
   
class CausalityTest():

    """Test for causal inferrence
       (lagged/contemporaneous causal relationship)

    parameters
    __________
    
    data: dataframe likes, (n_times, n_features)

    max_tau: max lagged time for test on causal relationship.

    """
        
    def __init__(self,
                 data=np.zeros((2,2)),
                 max_tau=0):

        self.data = data
        self.T, self.N = self.data.shape
        self.max_tau = max_tau       
        
    def _get_lagged_matrix_single(self,
                                  data,
                                  max_tau):
        """
        get matrix of past specific tau-timestep matrix of all variables
        """
        _T,_N = np.shape(data)
        
        # initialize (T,N) contrain lagged tau matrix
        _lagged_array = np.zeros((_T, _N))
        for i in range(_N):
            # roll tau-timestep of data
            _lagged_array[:,i] = np.roll(data[:,i],max_tau)
            # lagged time tau
            if max_tau < 0:
                np.put(_lagged_array[:,i],range(len(_lagged_array[:,i])+max_tau,
                                            len(_lagged_array[:,i])),np.nan)
            elif max_tau > 0:
                # give top tau number of vector as nan
                np.put(_lagged_array[:,i],range(max_tau),np.nan)
    
        _lagged_array = np.squeeze(_lagged_array)
                
        return _lagged_array
    
    
    def _set_default_parents(self):
        
        """
        Done ✔️
        
        ----
        Goal
        ----
        ***get default parents default value for max_tau in each def
        ***dimisions is key N, value max_tau*N.
    
        ----------
        attributes
        ----------
        default_value      -> [(0,1),...,(0,max_tau),
                              (1,1),...,(1,max_tau), ..., 
                              (N-1,1),...,(N-1,max_tau)]
        
        default_parents    -> dict as {0: default_value,
                                       1: default_value,..., 
                                       N-1: default_value}
        
        _len_node          -> length of default parents 
                              for each variable.  T*N.                               
        """
    
        # initial dict for default parents and default value
        default_parents = defaultdict(dict)
        default_value = list()
        # loop key 0-N-1
        # construct default value as max_tau*N length
        for i in range(self.N):
            for t in range(1,self.max_tau+1):
                default_value.append((i,t))
        # length of default parents for each variable. T*N
        _len_node = len(default_value)
        # pass default value for every key
        for i in range(self.N):
            default_parents[i] = default_value 
        
        return default_parents, default_value, _len_node
    
    def _set_parents_matrix(self,
                            data,
                            target_variable,
                            parents=None,
                            nan=None):
        
        """
        Done ✔️
        
        ----
        Goal
        ----
        ***return target vector and parents matrix 
        ***by definite target_variable and parents dict
    
        ##########
        Attention: 
        ##########   
            returned matrix have been removed nan.
    
        ----------
        attributes
        ----------    
        parents_matrix -> for selected target variable, give matrix contain 
                          every time series according (i,tau) in parents dict.
                          default as np.array.
    
        Y              -> target vector, np.array.
    
        _len_node      -> number of parents node for target variable.
                          default as T*N, the same as _set_default_parents.
                          non-default as length of parents for target variable.
        """
    
        # inital parents
        if parents is None:
            parents, _, _ = self._set_default_parents(self.max_tau)
        # shape of data. df
        _T, _N = data.shape
        # parents list for target variable
        parent_node = parents[target_variable]
        _len_node = len(parent_node)
        # if parent node for target variable is []
        # set parents_matrix,Y as []
        if _len_node == 0:
            parents_matrix,Y, _len_node = [],[],0
            return parents_matrix, Y, _len_node
        # if parent node is not []          
        # intial parents matrix
        parents_matrix = np.zeros((_T,_len_node))
        # columns index of data. df
        columns_index = [i for i in data.columns]
        # get parents matrix. np.array
        for index_parent, (i,tau) in enumerate(parent_node):
            # select i columns &
            # changed (N,) to (N,1),used for np.shape in def
            _X = data[columns_index[i]].to_numpy().reshape(-1,1)
            # get parents matrix for each (j,tau) for target variable.
            parents_matrix[:,index_parent] = self._get_lagged_matrix_single \
                                                 (_X,tau)
        # get target vector. np.array
        Y = data[columns_index[target_variable]].to_numpy()
        # remove first max_tau rows cuz nan
        if nan is None:
            parents_matrix = self._remove_row_np(parents_matrix)
            Y = self._remove_row_np(Y)
    
        return parents_matrix, Y, _len_node
    
    def _remove_row_np(self,
                       data):
        
        """
        Done ✔️
    
        remove first max_tau rows in data. 
        
        ##########
        Attention: 
        ########## 
            data used here is np.array
            
        attributes
        __________
        
        data_drop -> data remove first max_tau rows.
        """
    
        data_drop = np.delete(data, range(self.max_tau+1), axis=0)
    
        return data_drop
    
    def _check_missing_data(self, data):
        pass
    
    
class Corr(CausalityTest):   

    """correlation test for target variables and lagged variables.

    We apply correlation test(e.g., pearson, spearman, kendall's tau-b) firstly
    get the lagged parent node for target variable.

    This method includes following steps
    (1) For target variable, construct lagged matrix for input parents matrix, 
        which shape as[(0,1),(0,4),...,(j,tau)]
    (2) Caculate correlation between each column of lagged matrix and target 
        variable, and remove non-significant link of input parent node.
    """

    def _correlation_test(self,
                          parents=None,
                          method=0,
                          corr_alpha=0.05):

        """correlation test of X,Y
        apply pearson, spearman and kendall's tau-b correlation 

        parameters
        __________

        method               -> 0 for pearson correlation
                                1 for spearman correlation
                                2 for kendall's tau-b correlation
                                
                                default as spearman correlation 
                                cuz non-gaussian distribution of data.
                
        corr_alpha           -> correlation significant alpha for two-tails test
                                default as 0.05.
                        
        attribute
        _________
        
        _corr_parents        -> correlation parents. nest dict.
        """

        # inital parents
        if parents is None:
            parents, _, _len_node = self._set_default_parents()
        # initial correlation parents
        _corr_parents = defaultdict(dict)
        # loop for all target variable
        for i in range(self.N):
            _corr_parents[i] = list()
            # return X,Y for parents
            X,Y,_len_node = self._set_parents_matrix(self.data,i,parents)
            if _len_node == 0:
                _corr_parents[i] = []
            else:        
                # caculate correlation coefficient and p value 
                # for each column in X and Y
                for j in range(_len_node):
                    # corr and p value for X column and Y for three methods.
                    if method == 0:
                        corr_result = pearsonr(X[:,j],Y)
                    elif method == 1:
                        corr_result = spearmanr(X[:,j],Y)
                    elif method == 2:
                        corr_result = kendalltau(X[:,j],Y)
                    # if significant, append selected link.
                    if corr_result[1] < corr_alpha:  
                         _corr_parents[i].append(parents[i][j])
                     
        return _corr_parents
    

class Granger(CausalityTest):

    """granger test for target variable and lagged variable.
    ***granger causality test can not get comtemporaneous causal relationship

    We apply granger causality test[1](linear, nonlinear) to get lagged parents 
    node of target variable.

    This method includes following steps.
    (1) For target variable, construct lagged matrix for input parents matrix, 
        which shape as[(0,1),(0,4),...,(j,tau)]
    (2) Remove each column of parents matrix(selected (j,tau) lagged variable), 
        contract r2 of full model and baseline model(i.e., full model). 
        if r2 decrease, means that selected lagged variable significant impact
        target variable. vice versa. From this remove non-significant link of 
        input parent node.

    parameters
    __________

    parents: interface of parents node nest dict. type as {i: [(j,tau)]}

    GC_type: 1: linear granger causality. default as ridge regression
             2: nonlinear granger causality. default as random forest
    """
    
    def _granger_test(self,
                      parents=None,
                      GC_type=1):

        # inital parents
        if parents is None:
            parents, _, _len_node = self._set_default_parents()
        # initial correlation parents
        _GC_parents = defaultdict(dict)
        

        if GC_type == 1:
            # Class Ridge regression. invoid overfitting
            _reg = Ridge(copy_X=True, fit_intercept=True, max_iter=None, 
                         normalize=True, solver='lsqr', tol=0.001)
            # initialize the ndarray to contain cv_r2 for baseline model
            _r2_resid = np.zeros((self.max_tau,1))
            # loop for all target variables
            for i in range(self.N):
                _GC_parents[i] = list()
                # return X,Y for parents
                X,Y,_len_node = self._set_parents_matrix(self.data,i,parents)
                # no parent node for target variable
                if _len_node ==0:
                    _GC_parents[i] = []
                # only one parent node for target variable
                elif _len_node ==1:
                    ones = np.ones((np.shape(X)[0],1))
                    _r2_full = self._cross_validation \
                                    (X, Y, _reg,cv_type='ridge',cv_folds=5)
                    _r2_baseline = self._cross_validation \
                                    (ones, Y, _reg,cv_type='ridge',cv_folds=5)
                    if _r2_full < _r2_baseline:
                        _GC_parents[i] = []                    
                elif _len_node >1:                    
                    _r2_full = self._cross_validation \
                                    (X, Y, _reg,cv_type='ridge',cv_folds=5) 
                    # initialize copy of X
                    _X = deepcopy(X)
                    _T,_N = _X.shape
            
                    for j in range(_len_node):
                        np.put(_X[:,j], range(_T), np.nan)
                
                        df = pd.DataFrame(_X)
                        _X_reg = np.array(df.dropna(axis=1))
                        _r2_resid = self._cross_validation(_X_reg, Y, _reg,                                                         
                                                           cv_type='ridge',
                                                           cv_folds=5)
                        if _r2_resid > _r2_full:
                            _X = _X
                            _r2_full = _r2_resid
                        else:
                            _X[:,j] = X[:,j]
                            _r2_full = _r2_full
                    
                    df = pd.DataFrame(_X)
                    # bool array of significant t of j variable on i variable
                    _sign_t = df.columns[df.notna().any()].tolist()
                
                    for _sign_t_value in _sign_t:
                        _GC_parents[i].append(parents[i][_sign_t_value])
                
                print(_GC_parents)
                                  
        elif GC_type == 2:
    
            """
            TODO(@lewlee) need improve
            """
    
            # Class Random Forest. invoid overfitting
            _reg = RandomForestRegressor(random_state=0, 
                                         n_estimators=100, 
                                         max_features='sqrt')
       
        else:
    
            """
            TODO(@lewlee) consider more machine learning regression method
            """
            raise ValueError('Need to be improved! \nlilu love panjinjing')  
    
        return _GC_parents  


    def _cross_validation(self,
                          X, 
                          Y, 
                          estimator,
                          cv_type,
                          cv_folds=5):
        """
        cross validation for different estimator.
        
        parameters:
        ___________
        
        X,Y: predict and target matrix
        
        estimator: regression class use to gridsearch, give hyperparameters
                   such as estimator = Ridge()
        
        cv_type: regression mode, 
                 'ridge' for linear and 'random_forest' for nonlinear.
                 Attention: hyperparameters of random_forest isn't used here.
        
        cv_folds: int, use to split matrix. 
                  default set as 5
                  
        Attributes
        __________
        
        r2 : mean r2 value of cv_folds(default as 5)-folds cv
        
        """
        # class kfold as num-fold cross validation. default as 5-fold
        kfold = KFold(cv_folds, False)
        # split X and Y, use to return train and test index
        kf = kfold.split(X, Y)
        #imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
        
        # initialize r2 matrix
        r2 = np.zeros((cv_folds,1))
        for i, (train_index,test_index) in enumerate(kf):
            # get train and test set for gridsearchcv
            _train_x, _train_y = X[train_index,:], Y[train_index]
            _test_x, _test_y = X[test_index,:],Y[test_index] 
   
            if cv_type == 'ridge':
                # hyperparameters for ridge regression.
                # use for L2 normalization intense.
                param_grid = {'alpha':[0.01,0.1,1,10,100,1000]}
                
            elif cv_type == 'rf':
        
                param_grid = {'n_estimators': [50,100], 
                              'max_features': ['sqrt', 'log2'],
                              'bootstrap': [False]}
              
            elif cv_type == 'ann':
                
                param_grid = {'hidden_layer_sizes': [100, 200],
                              'batch_size':[128, 256]}#,
                              #'optimizer': ['adam', 'rmsprop'],
                              #'batch_size': [100, 200]}
                
            elif cv_type == 'svr':
                
                param_grid = {'kernel':('linear', 'rbf'), 
                              'C': [1e0, 1e1, 1e2],
                              'gamma': np.logspace(-2, 2, 5)}

            # class gridsearchcv 
            grid = GridSearchCV(estimator,
                                param_grid, 
                                verbose=0, 
                                cv=cv_folds, 
                                scoring='r2')
            # fit regression use estimator. 
            grid.fit(_train_x, _train_y)
            # use best estimator represent the 'best' regression 
            y_true, y_pred = _test_y, \
                             grid.best_estimator_.predict(_test_x)
            # r2
            r2[i] = r2_score(y_true, y_pred) 
    
        # mean r2 of cv_folds-fold cv
        r2 = np.mean(r2)
   
        return r2      

class CMI(CausalityTest):
    
    """conditional mutual information test for lagged and target variable

    We apply conditional mutual information test[2].

    This method includes following steps
    (1) For target variable, construct lagged matrix for input parents matrix, 
        which shape as[(0,1),(0,4),...,(j,tau)]
    (2) boostrap each column of the lagged matrix and get corresponding 
        mutual information. and we set a alpha, if cmi value less
        than alpha, means that cmi approach 0, i.e., the selected link is 
        non-significant.
    (3) we random choose i column of lagged matrix, and set as conditional 
        matrix, then we loop i from 1 to max_cond_dims. The non-significant 
        link is removed from input parents dict.
    """

    def _get_cond_matrix(self,
                         data,
                         target_index,
                         node_index,
                         cond_dims,
                         max_tau,
                         parents=None):
        """
        """
        
        # initial dict contain conditional matrix
        # key-value: {0: conditional T x k matrix, 1: ......}
        Z = defaultdict(dict)
        # target variable vector(i)
        Y = data[:,target_index]
        # shape of data and initial same shape matrix for contain lagged matrix
        _T, _N = np.shape(data)
        parent = parents[target_index]
        _len_node = len(parent)
        _X_all = np.zeros((_T, _len_node))

        # loop of parents node for target variable
        for index_parent,(parent_node,tau) in enumerate(parent):
            # get lagged matrix of predict variable on predict variable(i)
            # warning: NaN for first tau rows
            _X_all[:,index_parent] = np.squeeze(self._get_lagged_matrix_single \
                                     (data[:,parent_node].reshape(-1,1),tau))

        # predict variable vector for select node_index of target variable    
        X = _X_all[:,node_index]

        # delete selected variable
        _X = np.delete(_X_all,node_index,axis=1)  
        # conditional matrix
        if cond_dims != 0:
            # random choice cond_dims variable of predict variable 
            # after excluding selected predict variable node_index
            for index_len,index_cond in enumerate \
                                     (it.combinations \
                                     (range(_len_node-1),cond_dims)):
                # pass conditional matrix in dict
                print((index_len,index_cond))
                Z[index_len] = np.delete(_X[:,list(index_cond)], 
                                         range(max_tau), axis=0)
                
        else:
            Z[0] = list([0])
                    
        _X_list = list()
        _Y_list = list()
            
        for i in X:
            _X_list.append(list([i]))
        for j in Y:
            _Y_list.append(list([j]))
            
        del _X_list[0:max_tau]
        del _Y_list[0:max_tau]
        
        return _X_list,_Y_list,Z

    
    def _CMI_test_single(self,
                         data,
                         target_index,
                         max_tau,
                         parents=None,
                         max_cond_dims=None,
                         alpha=0):
        """
        parameters
        __________
        
        parent_node: list contain turple
                     [(0,1),(0,2),...,(1,2),...]           
        
        parents: nest dict.
                 such as {0: [(0,1),(0,2),...,(1,2),...], 
                          1: [(2,3),(2,1),...,(4,1),...]...}
                 first key: target variables
                 first value: lisf of predict variables and lagged t

                 default set as None, means that all lagged t and 
                 all predict variables is selected.

                 this parameters could be _GC_parents get from _granger_test.
                
                TODO(lewlee): parents got from other causality test.
                
        Attributes
        __________
        
        parents: nest dict.
                 input parameters after conditional mutual information test        
        
        """
        
        # parents nodes of target variable
        parent_node = parents[target_index]
        #print(parent_node)
        # current number of selected link of target variable
        _N = len(parent_node)       
        # loop for all conditional dimisions
        for cond_dims in range(_N):  
            #print(parent_node)
            # initial non-significant link list
            non_sign_link = []
            sign_link = []
            cmi_parent_node = []
            # check if conditional test is completed
            if cond_dims > max_cond_dims:
                break                        
            # loop for each predict variable and lagged time 
            # as node of target variable (0,1)
            # TODO: need improved
            for index_link, link in enumerate(parent_node):                
                # get matrix of target, predict and conditional
                X, Y, Z = self._get_cond_matrix(data, target_index, 
                                                index_link, cond_dims,
                                                max_tau,parents)
                
                # key,value of all possible of conditional matrix
                for index_cond, cond in Z.items():
                    #print(cond_dims, index_link, index_cond)
                    # perform conditional mutual information test
                    _cmi_test,_cmi_value = self._cond_mutual_information(X,Y,cond,
                                                              alpha,cond_dims)
                    #print(_cmi_test,_cmi_value)
                    # if non-significant,break of loop and delete this link
                    if _cmi_test is True:
                        non_sign_link.append(index_link)
                        break
            
            sign_link = [i for i in range(len(parent_node)) if i not in non_sign_link]
            #print(non_sign_link)
            #print(sign_link)
            #print(parent_node)

            if len(sign_link) != 0:                
                # delete non-significant link and 
                # construct new parent_node for next iteration(cond_dims+1)
                for sign_index,sign_value in enumerate(sign_link):
                    cmi_parent_node.append(parent_node[sign_value])
                parent_node = cmi_parent_node
            else:
                parent_node = []
                break
            #print(parent_node)
            
        #print(parent_node)
        return parent_node
    
    def _CMI_test(self,
                  data,
                  max_tau,
                  max_cond_dims=5,
                  parents=None,
                  alpha=0):
        """
        """
        _cmi_parents = defaultdict(dict)

        for target_variable in range(self.N):
            parent_node = self._CMI_test_single(data,target_variable,max_tau,
                                                parents=parents,
                                                max_cond_dims=max_cond_dims,
                                                alpha=alpha)

            
            _cmi_parents[target_variable] = parent_node
        
        _normalized_score = self._normalized_score(data, _cmi_parents)
            
        return _cmi_parents, _normalized_score
        
    def _cond_mutual_information(self, X, Y, Z,
                                 alpha,
                                 cond_dims):
        """
        
        KNN-based method from [1]
        The mutual information estimator is from Kraskov et.al[2]
        The continuous entropy estimator is based on Kozachenko and Leonenko[3] 
        The conditional mutual information estimator comes from Palus et.al.[4] 
        The KL Divergence estimator comes from Wang et. al.[5]
        
        [1] Non-parametric Entropy Estimation Toolbox(NPEET)
            https://github.com/gregversteeg/NPEET/blob/master/npeet_doc.pdf
            
        [2] Alexander Kraskov, Harald St ̈ogbauer, and Peter Grassberger.
            Estimating mutual information. 
            Phys. Rev. E, 69:066138, Jun 2004.
            
        [3] L. F. Kozachenko and N. N. Leonenko. 
            Sample estimate of the entropy of a random vector. 
            Probl. Peredachi Inf., 23(2):95–101, November 1987.
            
        [4] M. Vejmelka and M. Paluˇs. 
            Inferring the directionality of coupling with 
            conditional mutual information. 
            Physical Review E, 77(2):026214, 2008.
            
        [5] Qing Wang, Sanjeev R. Kulkarni, and Sergio Verdu ́. 
            Divergence estimation for mul- tidimensional densities 
            via k-nearest-neighbor distances.
            IEEE Trans. Inf. Theor., 55:2392–2405, May 2009.
            
        parameters:
        ___________
        
        X, Y: input matrix
        
        CMI_type: 'mi' for mutual information
                  'cmi' for conditional mutual information
                  
        alpha: thresold to judge whether conditional independence
               if cmi/mi < alpha, means conditional independence
               default as 0.05
               
        Z: conditional matrix
               
        Attributes
        __________
        
        _cmi_test: True for conditional independence
                   False for conditional dependence
                   
        """
        """
        TODO:changed input variable
        """
        if cond_dims == 0:
            # shuffle the x’s so that they are uncorrelated with y, 
            # then estimates whichever information measure 
            # you specify with “measure”. 
            # E.g., mutual information with mi would return 
            # the average mutual information 
            # (which should be near zero, because of the shuffling) 
            # along with the confidence interval 
            # (95% by default, set with the keyword ci = 0.95). 
            # This gives a good sense of numerical error.
            _cmi_value = self.shuffle_test(self.mi, X, Y, z=False,ns=1,ci=0.95)
        else:
            # caculate CMI for I(X,Y|Z)
            _cmi_value = self.shuffle_test(self.cmi, X, Y, z=Z, ns=1,ci=0.95)
            
        # judge conditional independence
        if _cmi_value[0] < alpha:
            _cmi_test = True
        else:
            _cmi_test = False
            
        return _cmi_test,_cmi_value[0] 
    
    def _normalized_score(self,
                          data,
                          parents=None):
        
        return_score = defaultdict(dict)
        for i in range(self.N):
            return_score[i] = list()
            X, Y, _len_node = self._set_parents_matrix(pd.DataFrame(data),i,parents)
            for j in range(_len_node):               
                return_score[i].append \
                               (metrics.normalized_mutual_info_score(X[:,j],Y))
        return return_score
    
    
    # CONTINUOUS ESTIMATORS
    
    def entropy(self,x, k=3, base=2):
        """ The classic K-L k-nearest neighbor continuous entropy estimator
            x should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
            if x is a one-dimensional scalar and we have four samples
        """
        assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
        x = np.asarray(x)
        n_elements, n_features = x.shape
        x = self.add_noise(x)
        tree = ss.cKDTree(x)
        nn = self.query_neighbors(tree, x, k)
        const = digamma(n_elements) - digamma(k) + n_features * log(2)
        return (const + n_features * np.log(nn).mean()) / log(base)
    
    
    def centropy(self,x, y, k=3, base=2):
        """ The classic K-L k-nearest neighbor continuous entropy estimator for the
            entropy of X conditioned on Y.
        """
        xy = np.c_[x, y]
        entropy_union_xy = self.entropy(xy, k=k, base=base)
        entropy_y = self.entropy(y, k=k, base=base)
        return entropy_union_xy - entropy_y
    
    
    def tc(self,xs, k=3, base=2):
        xs_columns = np.expand_dims(xs, axis=0).T
        entropy_features = [self.entropy(col, k=k, base=base) for col in xs_columns]
        return np.sum(entropy_features) - self.entropy(xs, k, base)
    
    
    def ctc(self,xs, y, k=3, base=2):
        xs_columns = np.expand_dims(xs, axis=0).T
        centropy_features = [self.centropy(col, y, k=k, base=base) for col in xs_columns]
        return np.sum(centropy_features) - self.centropy(xs, y, k, base)
    
    
    def corex(self,xs, ys, k=3, base=2):
        xs_columns = np.expand_dims(xs, axis=0).T
        cmi_features = [self.mi(col, ys, k=k, base=base) for col in xs_columns]
        return np.sum(cmi_features) - self.mi(xs, ys, k=k, base=base)
    
    
    def mi(self,x, y, z=None, k=3, base=2):
        """ Mutual information of x and y (conditioned on z if z is not None)
            x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
            if x is a one-dimensional scalar and we have four samples
        """
        assert len(x) == len(y), "Arrays should have same length"
        assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
        x, y = np.asarray(x), np.asarray(y)
        x = self.add_noise(x)
        y = self.add_noise(y)
        points = [x, y]
        if z is not None:
            points.append(z)
        points = np.hstack(points)
        # Find nearest neighbors in joint space, p=inf means max-norm
        tree = ss.cKDTree(points)
        dvec = self.query_neighbors(tree, points, k)
        if z is None:
            a, b, c, d = self.avgdigamma(x, dvec), self.avgdigamma(y, dvec), digamma(k), digamma(len(x))
        else:
            xz = np.c_[x, z]
            yz = np.c_[y, z]
            a, b, c, d = self.avgdigamma(xz, dvec), self.avgdigamma(yz, dvec), self.avgdigamma(z, dvec), digamma(k)
        return (-a - b + c + d) / log(base)
    
    
    def cmi(self,x, y, z, k=3, base=2):
        """ Mutual information of x and y, conditioned on z
            Legacy function. Use mi(x, y, z) directly.
        """
        return self.mi(x, y, z=z, k=k, base=base)
    
    
    def kldiv(self,x, xp, k=3, base=2):
        """ KL Divergence between p and q for x~p(x), xp~q(x)
            x, xp should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
            if x is a one-dimensional scalar and we have four samples
        """
        assert k < min(len(x), len(xp)), "Set k smaller than num. samples - 1"
        assert len(x[0]) == len(xp[0]), "Two distributions must have same dim."
        d = len(x[0])
        n = len(x)
        m = len(xp)
        const = log(m) - log(n - 1)
        tree = ss.cKDTree(x)
        treep = ss.cKDTree(xp)
        nn = self.query_neighbors(tree, x, k)
        nnp = self.query_neighbors(treep, x, k - 1)
        return (const + d * (np.log(nnp).mean() - np.log(nn).mean())) / log(base)
    
    
    # DISCRETE ESTIMATORS
    def entropyd(self,sx, base=2):
        """ Discrete entropy estimator
            sx is a list of samples
        """
        unique, count = np.unique(sx, return_counts=True, axis=0)
        # Convert to float as otherwise integer division results in all 0 for proba.
        proba = count.astype(float) / len(sx)
        # Avoid 0 division; remove probabilities == 0.0 (removing them does not change the entropy estimate as 0 * log(1/0) = 0.
        proba = proba[proba > 0.0]
        return np.sum(proba * np.log(1. / proba)) / log(base)
    
    
    def midd(self,x, y, base=2):
        """ Discrete mutual information estimator
            Given a list of samples which can be any hashable object
        """
        assert len(x) == len(y), "Arrays should have same length"
        return self.entropyd(x, base) - self.centropyd(x, y, base)
    
    
    def cmidd(self,x, y, z, base=2):
        """ Discrete mutual information estimator
            Given a list of samples which can be any hashable object
        """
        assert len(x) == len(y) == len(z), "Arrays should have same length"
        xz = np.c_[x, z]
        yz = np.c_[y, z]
        xyz = np.c_[x, y, z]
        return self.entropyd(xz, base) + self.entropyd(yz, base) - self.entropyd(xyz, base) - self.entropyd(z, base)
    
    
    def centropyd(self,x, y, base=2):
        """ The classic K-L k-nearest neighbor continuous entropy estimator for the
            entropy of X conditioned on Y.
        """
        xy = np.c_[x, y]
        return self.entropyd(xy, base) - self.entropyd(y, base)
    
    
    def tcd(self,xs, base=2):
        xs_columns = np.expand_dims(xs, axis=0).T
        entropy_features = [self.entropyd(col, base=base) for col in xs_columns]
        return np.sum(entropy_features) - self.entropyd(xs, base)
    
    
    def ctcd(self,xs, y, base=2):
        xs_columns = np.expand_dims(xs, axis=0).T
        centropy_features = [self.centropyd(col, y, base=base) for col in xs_columns]
        return np.sum(centropy_features) - self.centropyd(xs, y, base)
    
    
    def corexd(self,xs, ys, base=2):
        xs_columns = np.expand_dims(xs, axis=0).T
        cmi_features = [self.midd(col, ys, base=base) for col in xs_columns]
        return np.sum(cmi_features) - self.midd(xs, ys, base)
    
    
    # MIXED ESTIMATORS
    def micd(self,x, y, k=3, base=2, warning=True):
        """ If x is continuous and y is discrete, compute mutual information
        """
        assert len(x) == len(y), "Arrays should have same length"
        entropy_x = self.entropy(x, k, base)
    
        y_unique, y_count = np.unique(y, return_counts=True, axis=0)
        y_proba = y_count / len(y)
    
        entropy_x_given_y = 0.
        for yval, py in zip(y_unique, y_proba):
            x_given_y = x[(y == yval).all(axis=1)]
            if k <= len(x_given_y) - 1:
                entropy_x_given_y += py * self.entropy(x_given_y, k, base)
            else:
                if warning:
                    warnings.warn("Warning, after conditioning, on y={yval} insufficient data. "
                                  "Assuming maximal entropy in this case.".format(yval=yval))
                entropy_x_given_y += py * entropy_x
        return abs(entropy_x - entropy_x_given_y)  # units already applied
    
    
    def midc(self,x, y, k=3, base=2, warning=True):
        return self.micd(y, x, k, base, warning)
    
    
    def centropycd(self,x, y, k=3, base=2, warning=True):
        return self.entropy(x, base) - self.micd(x, y, k, base, warning)
    
    
    def centropydc(self,x, y, k=3, base=2, warning=True):
        return self.centropycd(y, x, k=k, base=base, warning=warning)
    
    
    def ctcdc(self,xs, y, k=3, base=2, warning=True):
        xs_columns = np.expand_dims(xs, axis=0).T
        centropy_features = [self.centropydc(col, y, k=k, base=base, warning=warning) for col in xs_columns]
        return np.sum(centropy_features) - self.centropydc(xs, y, k, base, warning)
    
    
    def ctccd(self,xs, y, k=3, base=2, warning=True):
        return self.ctcdc(y, xs, k=k, base=base, warning=warning)
    
    
    def corexcd(self,xs, ys, k=3, base=2, warning=True):
        return self.corexdc(ys, xs, k=k, base=base, warning=warning)
    
    
    def corexdc(self,xs, ys, k=3, base=2, warning=True):
        return self.tcd(xs, base) - self.ctcdc(xs, ys, k, base, warning)
    
    
    # UTILITY FUNCTIONS
    
    def add_noise(self,x, intens=1e-10):
        # small noise to break degeneracy, see doc.
        return x + intens * np.random.random_sample(x.shape)
    
    
    def query_neighbors(self,tree, x, k):
        return tree.query(x, k=k + 1, p=float('inf'), n_jobs=-1)[0][:, k]
    
    
    def avgdigamma(self,points, dvec):
        # This part finds number of neighbors in some radius in the marginal space
        # returns expectation value of <psi(nx)>
        n_elements = len(points)
        tree = ss.cKDTree(points)
        avg = 0.
        dvec = dvec - 1e-15
        for point, dist in zip(points, dvec):
            # subtlety, we don't include the boundary point,
            # but we are implicitly adding 1 to kraskov def bc center point is included
            num_points = len(tree.query_ball_point(point, dist, p=float('inf')))
            avg += digamma(num_points) / n_elements
        return avg
    
    
    # TESTS
    
    def shuffle_test(self,measure, x, y, z=False, ns=200, ci=0.95, **kwargs):
        """ Shuffle test
            Repeatedly shuffle the x-values and then estimate measure(x, y, [z]).
            Returns the mean and conf. interval ('ci=0.95' default) over 'ns' runs.
            'measure' could me mi, cmi, e.g. Keyword arguments can be passed.
            Mutual information and CMI should have a mean near zero.
        """
        x_clone = np.copy(x)  # A copy that we can shuffle
        outputs = []
        for i in range(ns):
            #np.random.shuffle(x_clone)
            if z is not False:
                outputs.append(measure(x_clone, y, z, **kwargs))
            else:
                outputs.append(measure(x_clone, y, **kwargs))
        outputs.sort()
    #    import matplotlib.pyplot as plt
    #    
    #    plt.figure()
    #    plt.plot(outputs,label='boosting')
    #    plt.plot(outputs[int((1. - ci) / 2 * ns)],label='0.05')   
    #    plt.plot(outputs[int((1. + ci) / 2 * ns)],label='0.95')
    #    plt.legend(loc='best')
    #    plt.show()
        
        return np.mean(outputs), (outputs[int((1. - ci) / 2 * ns)], outputs[int((1. + ci) / 2 * ns)])


class DirectNGAM(CausalityTest):
    
    def make_graph(self,adjacency_matrix):
        idx = np.abs(adjacency_matrix) > 0.01
        dirs = np.where(idx)
        d = graphviz.Digraph(engine='dot')
        for to, from_, coef in zip(dirs[0], dirs[1], adjacency_matrix[idx]):
            d.edge(f'x{from_}', f'x{to}', label=f'{coef:.2f}')
        return d
    
    def _adjacency_matrix(self,df):
        
        model = lingam.DirectLiNGAM()
        model.fit(df)
        return model.adjacency_matrix_
    
    def _append_comteporaneous_link(self,df, parents=None):
        
        _adjacency_matrix = self._adjacency_matrix(df)
        i,j = np.nonzero(_adjacency_matrix)

        for index in range(len(i)):
            parents[i[index]].append((j[index],0))
        
        return parents
    

        
        
        
        
        
        
        
        
        
        
        
        

