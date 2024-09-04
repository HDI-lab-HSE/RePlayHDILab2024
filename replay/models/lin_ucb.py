import math
import numpy as np
import pandas as pd
import os
from  timeit import default_timer

from typing import Any, Dict, List, Optional

from replay.data.dataset import Dataset
from replay.metrics import NDCG, Metric
from replay.utils import PYSPARK_AVAILABLE, SparkDataFrame

from .base_rec import HybridRecommender

if PYSPARK_AVAILABLE:
    from pyspark.sql import functions as sf

from replay.utils.spark_utils import convert2spark

#import for parameter optimization
from optuna import create_study
from optuna.samplers import TPESampler

import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import lsqr

import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from numba import njit
@njit
def _blockify(ind, ptr, major_dim):
    # convenient function to compute only diagonal
    # elements of the product of 2 matrices;
    # indices must be intp in order to avoid overflow
    # major_dim is shape[0] for csc format and shape[1] for csr format
    n = len(ptr) - 1
    for i in range(1, n): #first row/col is unchanged
        lind = ptr[i]
        rind = ptr[i+1]
        for j in range(lind, rind):
            shift_ind = i * major_dim
            ind[j] += shift_ind

def row_blockify(mat, block_size):
    # only for CSR matrices
    _blockify(mat.indices, mat.indptr, block_size)
    mat._shape = (mat.shape[0], block_size*mat.shape[0])           


#Object for interactions with a single arm in a UCB disjoint framework
class linucb_disjoint_arm():
    def __init__(self, arm_index, d, eps, alpha): #in case of lin ucb with disjoint features: d = dimension of user's features solely
        # Track arm index
        self.arm_index = arm_index
        # Exploration parameter
        self.eps = eps
        self.alpha = alpha
        # Inverse of feature matrix for ridge regression
        self.A = csr_matrix(self.alpha*np.identity(d))
        self.A_inv = (1./self.alpha)*np.identity(d)
        # right-hand side of the regression
        self.theta = np.zeros(d, dtype = float)
        self.cond_number = 1.0
    
    def feature_update(self, usr_features, cnt, relevances):
        """
        function to update featurs or each Lin-UCB hand in the current model
        features:
            usr_features = matrix (np.array of shape (m,d)), where m = number of occurences of the current feature in the dataset;
            usr_features[i] = features of i-th user, who rated this particular arm (movie);
            relevances = np.array(d) - rating of i-th user, who rated this particular arm (movie);
        """
        # Update A which is (d * d) matrix.
        #unique_users, counts = np.unique(usr_features, axis = 0, return_counts=True)
        diag =scipy.sparse.diags(cnt, format='csr')
        self.A += (usr_features.T)@(diag @ usr_features)
        #self.A_inv = np.ascontiguousarray(scipy.linalg.inv(self.A.toarray()))
        self.A_inv = csr_matrix(scipy.linalg.inv(self.A.toarray()))
        # Update the parameter theta by the results  linear regression
        self.theta = lsqr(self.A, usr_features.T.dot(cnt * relevances))[0]


class LinUCB(HybridRecommender):
    """
    Function implementing the functional of linear UCB 
    """
    _search_space = {
        "eps": {"type": "uniform", "args": [-10.0, 10.0]},
        "alpha": {"type": "uniform", "args": [0.001, 10.0]},
    }

    def __init__(
        self,
        eps: float, #exploration parameter
        alpha: float, #ridge parameter
        regr_type: str, #put here "disjoint" or "hybrid"
        random_state: Optional[int] = None,
    ):  # pylint: disable=too-many-arguments
        np.random.seed(42)
        self.regr_type = regr_type
        self.random_state = random_state
        self.eps = eps
        self.alpha = alpha
        self.linucb_arms = None #initialize only when working within fit method
        cpu_count = os.cpu_count()
        self.num_threads = cpu_count if cpu_count is not None else 1


        self._study = None #field required for proper optuna's optimization
        self._search_space = {
        "eps": {"type": "uniform", "args": [-10.0, 10.0]},
        "alpha": {"type": "uniform", "args": [0.001, 10.0]},
        }
        #self._objective = MainObjective


    @property
    def _init_args(self):
        return {
            "regression type": self.regr_type,
            "seed": self.random_state,
        }

    # pylint: disable=too-many-arguments
    def optimize(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        param_borders: Optional[Dict[str, List[Any]]] = None,
        criterion: Metric = NDCG,
        k: int = 10,
        budget: int = 10,
        new_study: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Searches best parameters with optuna.

        :param train: train data
        :param test: test data
        :param user_features: user features
        :param item_features: item features
        :param param_borders: a dictionary with search borders, where
            key is the parameter name and value is the range of possible values
            ``{param: [low, high]}``. In case of categorical parameters it is
            all possible values: ``{cat_param: [cat_1, cat_2, cat_3]}``.
        :param criterion: metric to use for optimization
        :param k: recommendation list length
        :param budget: number of points to try
        :param new_study: keep searching with previous study or start a new study
        :return: dictionary with best parameters
        """

        # self.logger.warning(
        #     "The UCB model has only exploration coefficient parameter, which cannot not be directly optimized"
        # )


        self.query_column = train_dataset.feature_schema.query_id_column
        self.item_column = train_dataset.feature_schema.item_id_column
        self.rating_column = train_dataset.feature_schema.interactions_rating_column
        self.timestamp_column = train_dataset.feature_schema.interactions_timestamp_column

        self.criterion = criterion(
            topk=k,
            query_column=self.query_column,
            item_column=self.item_column,
            rating_column=self.rating_column,
        )

        if self._search_space is None:
            self.logger.warning("%s has no hyper parameters to optimize", str(self))
            return None

        if self.study is None or new_study:
            self.study = create_study(direction="maximize", sampler=TPESampler())

        search_space = self._prepare_param_borders(param_borders)
        if self._init_params_in_search_space(search_space) and not self._params_tried():
            self.study.enqueue_trial(self._init_args)

        split_data = self._prepare_split_data(train_dataset, test_dataset)
        objective = self._objective(
            search_space=search_space,
            split_data=split_data,
            recommender=self,
            criterion=self.criterion,
            k=k,
        )

        self.study.optimize(objective, budget)
        best_params = self.study.best_params
        self.set_params(**best_params)
        return best_params
    

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        feature_schema = dataset.feature_schema
        #should not work if user features or item features are unavailable 
        if dataset.query_features is None:
            raise ValueError("User features are missing for fitting")
        if dataset.item_features is None:
            raise ValueError("Item features are missing for fitting")
        #assuming that user_features and item_features are both dataframes
        #now forget about pyspark until the better times
        log = dataset.interactions.toPandas()
        user_features = dataset.query_features.toPandas()
        item_features = dataset.item_features.toPandas()
        #check that the dataframe contains uer indexes
        if not feature_schema.query_id_column in user_features.columns:
            raise ValueError("User indices are missing in user features dataframe")
        self._num_items = item_features.shape[0]
        self._user_dim_size = user_features.shape[1] - 1
        self._item_dim_size = item_features.shape[1] - 1
        #now initialize an arm object for each potential arm instance
        self.linucb_arms = [linucb_disjoint_arm(arm_index = i, d = self._user_dim_size * self._item_dim_size , eps = self.eps, alpha = self.alpha) for i in range(self._num_items)]
        #now we work with pandas
        item_features_matrix = csr_matrix(item_features.drop(columns=[feature_schema.item_id_column]).values) 

        for i in range(self._num_items):
            B = log.loc[log[feature_schema.item_id_column] == i]
            idxs_list = B[feature_schema.query_id_column].values
            rel_list = B[feature_schema.interactions_rating_column].values
            if not B.empty:
                #if we have at least one user interacting with the hand i
                cur_usrs = user_features.query(f"{feature_schema.query_id_column} in @idxs_list").drop(columns=[feature_schema.query_id_column])
                users_feat_cols = cur_usrs.columns.values.tolist()
                cur_usrs[feature_schema.interactions_rating_column] = rel_list          
                cur_usrs_unique = cur_usrs.groupby(users_feat_cols + [feature_schema.interactions_rating_column]).size().to_frame('size').reset_index()
                z = scipy.sparse.kron(csr_matrix(cur_usrs_unique[users_feat_cols].values), item_features_matrix[i], format = 'csr')
                rel = cur_usrs_unique[feature_schema.interactions_rating_column].to_numpy()
                self.linucb_arms[i].feature_update(z, cur_usrs_unique['size'].to_numpy().reshape(-1), rel)
 
        condit_number = [self.linucb_arms[i].cond_number for i in range(self._num_items)]
        #finished
        return
        
    def _predict(
        self,   
        dataset: Dataset,
        k: int,
        users: SparkDataFrame,
        items: SparkDataFrame = None,
        filter_seen_items: bool = True,
    ) -> SparkDataFrame:
        feature_schema = dataset.feature_schema
        num_user_pred = users.count() #assuming it is a pyspark dataset
        users = users.toPandas()
        items = items.toPandas()
        user_features = dataset.query_features.toPandas()
        item_features = dataset.item_features.toPandas()
        idxs_list = users[feature_schema.query_id_column].values
        itm_idxs_list = items[feature_schema.item_id_column].values
        itm_feat = csr_matrix(item_features.query(f"{feature_schema.item_id_column} in @itm_idxs_list").drop(columns=[feature_schema.item_id_column]).values)
        batch_size = idxs_list.shape[0]
        num_it = idxs_list.shape[0]//batch_size
        rel_matrix = np.zeros((num_user_pred,self._num_items),dtype = float)
        if user_features is None:
            raise ValueError("Can not make predict in the Lin UCB method")

        for i in range(num_it+1):
            j = min((i+1)*batch_size, idxs_list.shape[0])
            if j == i*batch_size:
                break
            cur_idxs_list = idxs_list[i*batch_size:j]
            usrs_feat = csr_matrix(user_features.query(f"{feature_schema.query_id_column} in @cur_idxs_list").drop(columns=[feature_schema.query_id_column]).values)
            for k in range(self._num_items):
                z = scipy.sparse.kron(usrs_feat, itm_feat[k], format = 'csr')
                z_A_inv = z.dot(self.linucb_arms[k].A_inv)
                z_A_inv_z = z.multiply(z_A_inv)
                z_A_inv_z = np.sqrt(z_A_inv_z.sum(axis=1)).squeeze()
                rel_matrix[i*batch_size:j,k] = self.eps * z_A_inv_z + z.dot(self.linucb_arms[k].theta)
                # z_theta = z.dot(self.linucb_arms[k].theta)
                # z_A_inv = (z.dot(self.linucb_arms[k].A_inv.T)).toarray().reshape(-1)
                # row_blockify(z, z.shape[1])
                # z_A_inv_z = np.sqrt(z.dot(z_A_inv))
                # rel_matrix[i*batch_size:j,k] = self.eps * z_A_inv_z + z_theta
        
        #select top k predictions from each row (unsorted ones)
        big_k = self._num_items // 10
        topk_indices = np.argpartition(rel_matrix, -big_k, axis=1)[:, -big_k:]
        rows_inds,_ = np.indices((num_user_pred, big_k))
        #result df
        predict_inds = np.repeat(idxs_list, big_k)
        predict_items = topk_indices.ravel()
        predict_rels = rel_matrix[rows_inds,topk_indices].ravel()
        #return everything in a PySpark template
        res_df = pd.DataFrame({feature_schema.query_id_column: predict_inds, feature_schema.item_id_column: predict_items, feature_schema.interactions_rating_column: predict_rels})
        return convert2spark(res_df)