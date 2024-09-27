import math
import numpy as np
import pandas as pd
import os

from typing import Any, Dict, List, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.metrics import Metric, NDCG
from replay.models.base_rec import HybridRecommender
from replay.utils.spark_utils import convert2spark
from scipy.stats import multivariate_normal
from replay.data.dataset import Dataset

#Object for interactions with a single arm in a TS disjoint framework
class lints_disjoint_arm():
    def __init__(self, arm_index, d, nu, alpha): #in case of lin ts with disjoint features: d = dimension of user's features solely
        # Track arm index
        self.arm_index = arm_index
        # Exploration parameter
        self.nu = nu
        self.alpha = alpha
        # Inverse of feature matrix for ridge regression
        self.A = self.alpha*np.identity(d)
        self.A_inv = (1./self.alpha)*np.identity(d)
        # right-hand side of the regression
        self.theta = np.zeros(d, dtype = float)
        self.cond_number = 1.0
    
    def feature_update(self, usr_features, relevances):
        """
        function to update featurs or each Lin-TS hand in the current model
        features:
            usr_features = matrix (np.array of shape (m,d)), where m = number of occurences of the current feature in the dataset;
            usr_features[i] = features of i-th user, who rated this particular arm (movie);
            relevances = np.array(d) - rating of i-th user, who rated this particular arm (movie);
        """
        # Update A which is (d * d) matrix.
        self.A += np.dot(usr_features.T, usr_features)
        self.A_inv = np.linalg.inv(self.A)
        # Update the parameter theta by the results  linear regression
        self.theta = np.linalg.lstsq(self.A, usr_features.T @ relevances, rcond = 1.0)[0]
        self.cond_number = np.linalg.cond(self.A) #this ome needed for deug only


class LinTS(HybridRecommender):
    """
    Function implementing the functional of linear TS 
    """
    
    _search_space = {
        "nu": {"type": "uniform", "args": [1.2, 2.5]},
        "alpha": {"type": "uniform", "args": [990.0, 1010.0]},
    }


    def __init__(
        self,
        nu: float, #exploration parameter
        alpha: float, #ridge parameter
        regr_type: str, #put here "disjoint" or "hybrid"
        random_state: Optional[int] = None,
    ):  # pylint: disable=too-many-arguments
        self.regr_type = regr_type
        self.random_state = random_state
        self.nu = nu
        self.alpha = alpha
        self.lints_arms = None #initialize only when working within fit method
        cpu_count = os.cpu_count()
        self.num_threads = cpu_count if cpu_count is not None else 1

    @property
    def _init_args(self):
        return {
            "regression type": self.regr_type,
            "seed": self.random_state,
        }


    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        #should not work if user features or item features are unavailable 
        if user_features is None:
            raise ValueError("User features are missing for fitting")
        if item_features is None:
            raise ValueError("Item features are missing for fitting")
        #assuming that user_features and item_features are both dataframes
        self._num_items = item_features.count()
        #now forget about pyspark until the better times
        log = log.toPandas()
        user_features = user_features.toPandas()
        item_features = item_features.toPandas()
        #check that the dataframe contains uer indexes
        if not 'user_idx' in user_features.columns:
            raise ValueError("User indices are missing in user features dataframe")
        self._user_dim_size = len(user_features.columns) - 1
        #now initialize an arm object for each potential arm instance
        self.lints_arms = [lints_disjoint_arm(arm_index = i, d = self._user_dim_size, nu = self.nu, alpha = self.alpha) for i in range(self._num_items)]
        #now we work with pandas
        for i in range(self._num_items):
            B = log.loc[log['item_idx'] == i]
            indx_list = B['user_idx'].values
            rel_list = B['relevance'].values
            if not B.empty:
                #if we have at least one user interacting with the hand i
                cur_usrs = user_features.query("user_idx in @indx_list").drop(columns=['user_idx'])
                self.lints_arms[i].feature_update(cur_usrs.to_numpy(),rel_list)
        condit_number = [self.lints_arms[i].cond_number for i in range(self._num_items)]
        #finished
        return 
        
    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        #create a large vectorized numpy array with inverse matrices:
        arr = [self.lints_arms[i].A_inv for i in range(self._num_items)]
        num_user_pred = users.count() #assuming it is a pyspark dataset
        users = users.toPandas()
        user_features = user_features.toPandas()
        indx_list = users['user_idx'].values
        if user_features is None:
            raise ValueError("Can not make predict in the Lin TS method")
        usrs_feat = user_features.query("user_idx in @indx_list").drop(columns=['user_idx']).to_numpy()
        rel_matrix = np.zeros((num_user_pred,self._num_items),dtype = float)
        #fill in relevance matrix
        for i in range(self._num_items):
            theta_tilda = self.lints_arms[i].theta.reshape(-1)
            var = multivariate_normal.rvs(np.zeros(self.lints_arms[i].theta.shape[0]),(self.nu*self.nu)*self.lints_arms[i].A_inv, 1, self.random_state)
            #if np.min(var) < 0:
            theta_tilda += var
            rel_matrix[:,i] = usrs_feat.dot(theta_tilda)
        #select top k predictions from each row (unsorted ones)
        big_k = 20*k
        topk_indices = np.argpartition(rel_matrix, -big_k, axis=1)[:, -big_k:]
        rows_inds,_ = np.indices((num_user_pred, big_k))
        #result df
        predict_inds = np.repeat(indx_list, big_k)
        predict_items = topk_indices.ravel()
        predict_rels = rel_matrix[rows_inds,topk_indices].ravel()
        #return everything in a PySpark template
        res_df = pd.DataFrame({'user_idx': predict_inds, 'item_idx': predict_items,'relevance': predict_rels})
        return convert2spark(res_df)