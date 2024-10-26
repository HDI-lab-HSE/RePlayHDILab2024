import logging
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import numpy as np
import pandas as pd
from obp.policy.base import BaseOfflinePolicyLearner
from optuna import create_study
from optuna.samplers import TPESampler
from pyspark.sql import DataFrame

from replay.data import Dataset, FeatureHint, FeatureInfo, FeatureSchema, FeatureType
from replay.experimental.scenarios.movielens_wrapper.obp_optuna_objective import OBPObjective
from replay.experimental.scenarios.movielens_wrapper.utils import split_bandit_feedback
from replay.models.base_rec import BaseRecommender
from replay.utils.spark_utils import convert2spark, get_top_k_recs, return_recs

from replay.metrics import HitRate, NDCG
from replay.models import UCB, Wilson, RandomRec, LinUCB, LinTS

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from replay.utils import PYSPARK_AVAILABLE, SparkDataFrame
from pyspark.sql.functions import col,lit
from pyspark.sql import functions as sf
from tqdm import tqdm


def obp2df(action: np.ndarray, reward: np.ndarray, timestamp: np.ndarray) -> Optional[pd.DataFrame]:
    """
    Converts OBP log to the pandas DataFrame
    """

    n_interactions = len(action)

    df = pd.DataFrame(
        {
            "user_idx": np.arange(n_interactions),
            "item_idx": action,
            "rating": reward,
            "timestamp": timestamp,
        }
    )

    return df


def context2df(context: np.ndarray, idx_col: np.ndarray, idx_col_name: str) -> Optional[pd.DataFrame]:
    """
    Converts OBP log to the pandas DataFrame
    """

    df1 = pd.DataFrame({idx_col_name + "_idx": idx_col})
    cols = [str(i) + "_" + idx_col_name for i in range(context.shape[1])]
    df2 = pd.DataFrame(context, columns=cols)

    return df1.join(df2)


@dataclass
class OBPOfflinePolicyLearner(BaseOfflinePolicyLearner):
    """
    Off-policy learner which wraps OBP data representation into replay format.

    :param n_actions: Number of actions.

    :param len_list: Length of a list of actions in a recommendation/ranking inferface,
                     slate size. When Open Bandit Dataset is used, 3 should be set.

    :param replay_model: Any model from replay library with fit, predict functions.

    :param dataset: Dataset of interactions (user_id, item_id, rating).
                Constructing inside the fit method. Used for predict of replay_model.
    """

    replay_model: Optional[BaseRecommender] = None
    log: Optional[DataFrame] = None
    item_features: DataFrame = None
    _study = None
    _logger: Optional[logging.Logger] = None
    _objective = OBPObjective

    def __post_init__(self) -> None:
        """Initialize Class."""
        self.feature_schema = FeatureSchema(
            [
                FeatureInfo(
                    column="user_idx",
                    feature_type=FeatureType.CATEGORICAL,
                    feature_hint=FeatureHint.QUERY_ID,
                ),
                FeatureInfo(
                    column="item_idx",
                    feature_type=FeatureType.CATEGORICAL,
                    feature_hint=FeatureHint.ITEM_ID,
                ),
                FeatureInfo(
                    column="relevance",
                    feature_type=FeatureType.NUMERICAL,
                    feature_hint=FeatureHint.RATING,
                ),
                FeatureInfo(
                    column="timestamp",
                    feature_type=FeatureType.NUMERICAL,
                    feature_hint=FeatureHint.TIMESTAMP,
                ),
            ]
        )
        self.replay_model.can_predict_cold_queries = True 

    @property
    def logger(self) -> logging.Logger:
        """
        :return: get library logger
        """
        if self._logger is None:
            self._logger = logging.getLogger("replay")
        return self._logger

    def fit(
        self,
        dataset: dict
    ) -> None:
        
        self.log = dataset['log']
        self.item_features = dataset['item_features']
        self.user_features = dataset['user_features']

        
        dataset = Dataset(
            feature_schema=self.feature_schema,
            interactions=self.log,
            item_features=self.item_features,
            query_features=self.user_features,
            categorical_encoded = True
        )

        self.replay_model._fit_wrap(dataset)
        
        self.used_actions = [[] for _ in range(self.user_features.count())]
        
        users = self.log.toPandas()['user_idx'].tolist()
        actions = self.log.toPandas()['item_idx'].tolist()
        for i in range(len(users)):
            self.used_actions[users[i]].append(actions[i])

    def predict(self, n_rounds: int = 1, users: SparkDataFrame = None) -> np.ndarray:
        """Predict best actions for new data.
        Action set predicted by this `predict` method can contain duplicate items.
        If a non-repetitive action set is needed, please use the `sample_action` method.

        :context: Context vectors for new data.

        :return: Action choices made by a classifier, which can contain duplicate items.
            If a non-repetitive action set is needed, please use the `sample_action` method.
        """
        items = convert2spark(pd.DataFrame({"item_idx": np.arange(self.item_features.count())}))
        

        dataset = Dataset(
            feature_schema=self.feature_schema,
            interactions=self.log,
            query_features=self.user_features,
            item_features=self.item_features,
            check_consistency=False,
        )

        if isinstance(self.replay_model, (LinUCB, LinTS)):
            pred = self.replay_model._predict(dataset, self.item_features.count(), users, items, filter_seen_items=False)
            action_dist = np.zeros((n_rounds, self.item_features.count(), 1))
            pred = pred.withColumn(
                "Softmax_Score",
                F.exp("relevance") / F.sum(F.exp("relevance")).over(Window.partitionBy("user_idx"))
            ).cache()

            users_list = users.toPandas()['user_idx'].tolist()

            user2ind = {}
            for i in range(len(users_list)):
                user2ind[users_list[i]] = i

            pred = pred.toPandas()
            rearranged_user_idx = pred['user_idx'].tolist()
            for i in range(len(rearranged_user_idx)):
                rearranged_user_idx[i] = user2ind[rearranged_user_idx[i]]

            pred['new_idx'] = rearranged_user_idx
            pred = convert2spark(pred)
        
            action_dist[pred.select('new_idx').toPandas().values, pred.select('item_idx').toPandas().values, 0] =  pred.select('Softmax_Score').toPandas().values
        else:       
            action_dist = self.replay_model._predict_proba(dataset, 1, users, items, filter_seen_items=False)

        return action_dist


    def predict_and_evaluate(self, bandit_feedback_test, K: int = None):
        
        items = convert2spark(pd.DataFrame({"item_idx": np.arange(bandit_feedback_test['item_features'].count())}))
        
        dataset = Dataset(
            feature_schema=self.feature_schema,
            interactions=self.log,
            query_features=self.user_features,
            item_features=self.item_features,
            check_consistency=True,
        )
        
        pos_log = bandit_feedback_test['log'].filter(sf.col('relevance') == 1)
        
        actions_list = np.array(pos_log.toPandas()['item_idx'].tolist())
        users_list = np.array(pos_log.toPandas()['user_idx'].tolist())
        ind2user = list(set(users_list))
        
        user2ind = {}
        for i in range(len(ind2user)):
            user2ind[ind2user[i]] = i
             
        pos_log_distinct = pos_log.toPandas().drop_duplicates(subset=["user_idx"], keep='first')
        
        holdout_actions = np.zeros((pos_log_distinct.shape[0], self.n_actions))
        for i in range(len(actions_list)):
            holdout_actions[user2ind[users_list[i]]][actions_list[i]] = 1

        ratings = np.zeros((pos_log_distinct.shape[0], self.n_actions))
        
        batch_size = 10
        num_batchs = pos_log_distinct.shape[0] // batch_size
        for i in tqdm(range(num_batchs+1)):
            j = min((i+1)*batch_size, pos_log_distinct.shape[0])
            if j == i*batch_size:
                break
            log_subset = pos_log_distinct.iloc[i*batch_size: j]
            
            users = convert2spark(log_subset).select('user_idx')
            
            pred = self.replay_model._predict(dataset, self.n_actions, users, items, filter_seen_items=False).toPandas()
            rearranged_user_idx = pred['user_idx'].tolist()
            for i in range(len(rearranged_user_idx)):
                rearranged_user_idx[i] = user2ind[rearranged_user_idx[i]]
        
            ratings[rearranged_user_idx, pred['item_idx'].tolist()] = pred['relevance'].tolist()
            
        
        for user in ind2user:
            seen_actions = self.used_actions[user]
            ratings[user2ind[user], seen_actions] = -np.inf
            
        
        recommended_items = np.argsort(-ratings, axis = 1)[:, :K]
        
        recommended_mask = np.zeros((pos_log_distinct.shape[0], self.n_actions))
        for i in range(len(ind2user)):
            recommended_mask[i][recommended_items[i]] = 1
        
        hits_mask = recommended_mask * holdout_actions

        # HR calculation
        hr = np.mean(hits_mask.any(axis=1))

        # MRR calculation
        hit_rank = np.ones(len(ind2user))
        for i in range(len(ind2user)):
            for j, item in enumerate(recommended_items[i]):
                if hits_mask[i][item]:
                    hit_rank[i] = j+1
                    break
                
                
        mrr = np.mean(hits_mask.any(axis=1)*(1/hit_rank))
        
        
        #NDCG calculation
        interactions = [[] for _ in range(len(ind2user))]
        for i in range(len(ind2user)):
            for j, item in enumerate(recommended_items[i]):
                interactions[i].append(holdout_actions[i][item])
        interactions = np.array(interactions)
        
        log_ranks = np.log2(np.arange(1, K+1) + 1)
        dcg = np.mean(np.sum(interactions/log_ranks, axis=1))
    
        
        ideal_interactions = [[] for _ in range(len(ind2user))]
        for i in range(len(ind2user)):
            for j in range(K):
                if j<K:
                    ideal_interactions[i].append(1)
                else:
                    ideal_interactions[i].append(0)
        ideal_interactions = np.array(ideal_interactions)
        idcg = np.mean(np.sum(ideal_interactions/log_ranks, axis=1))
    
        ndcg = dcg/idcg
        
        
        #COV calculation
        cov = np.unique(recommended_items).size / self.n_actions

        return {f'hr@{K}': hr, f'mrr@{K}': mrr, f'ndcg@{K}': ndcg, f'cov@{K}': cov}


    def predict_and_evaluate_new(self, bandit_feedback_test, K: int = None):
        
        items = convert2spark(pd.DataFrame({"item_idx": np.arange(bandit_feedback_test['item_features'].count())}))
        
        dataset = Dataset(
            feature_schema=self.feature_schema,
            interactions=self.log,
            query_features=self.user_features,
            item_features=self.item_features,
            check_consistency=True,
        )
        
        pos_log = bandit_feedback_test['log'].filter(sf.col('relevance') == 1)
        
        actions_list = np.array(pos_log.toPandas()['item_idx'].tolist())
        users_list = np.array(pos_log.toPandas()['user_idx'].tolist())
        ind2user = list(set(users_list))
        
        user2ind = {}
        for i in range(len(ind2user)):
            user2ind[ind2user[i]] = i
             
        pos_log_distinct = pos_log.toPandas().drop_duplicates(subset=["user_idx"], keep='first')
        ratings = np.zeros((pos_log_distinct.shape[0], self.n_actions))
        
        batch_size = 10
        num_batchs = pos_log_distinct.shape[0] // batch_size
        for i in tqdm(range(num_batchs+1)):
            j = min((i+1)*batch_size, pos_log_distinct.shape[0])
            if j == i*batch_size:
                break
            log_subset = pos_log_distinct.iloc[i*batch_size: j]
            
            users = convert2spark(log_subset).select('user_idx')
            
            pred = self.replay_model._predict(dataset, self.n_actions, users, items, filter_seen_items=False).toPandas()
            rearranged_user_idx = pred['user_idx'].tolist()
            for i in range(len(rearranged_user_idx)):
                rearranged_user_idx[i] = user2ind[rearranged_user_idx[i]]
        
            ratings[rearranged_user_idx, pred['item_idx'].tolist()] = pred['relevance'].tolist()
            
        for user in ind2user:
            seen_actions = self.used_actions[user]
            ratings[user2ind[user], seen_actions] = -np.inf
            
            
        n_predicitions = np.zeros(len(ind2user))
        n_hits = np.zeros(len(ind2user))
        ranks_inv = np.zeros(len(ind2user))
        log_ranks_inv = np.zeros(len(ind2user))
        ideal_log_ranks_inv = np.zeros(len(ind2user))
        
        
        
        recommended_items_set = set()
        
        # HR, MRR, NDCG calculation
        for row_ind in range(len(actions_list)):
            user_ind = user2ind[users_list[row_ind]]
            true_action = actions_list[row_ind]
            recommended_items = np.argsort(-ratings[user_ind])[:K]
            
            recommended_items_set.update(recommended_items)
            
            n_predicitions[user_ind] += 1
            ideal_log_ranks_inv[user_ind] += 1
            if true_action in recommended_items:
                n_hits[user_ind] += 1
                ranks_inv[user_ind] += 1/(list(recommended_items).index(true_action)+1)
                log_ranks_inv[user_ind] += 1/np.log2((list(recommended_items).index(true_action)+1)+1)
                
            ratings[user_ind][true_action] = -np.inf
            
        hr = (n_hits / n_predicitions).mean()
        mrr = (ranks_inv / n_predicitions).mean()
        dcg = (log_ranks_inv / n_predicitions).mean()
        idcg = (ideal_log_ranks_inv / n_predicitions).mean()
        ndcg = dcg/idcg
        
        cov = len(recommended_items_set) / self.n_actions
        
        return {f'hr@{K}': hr, f'mrr@{K}': mrr, f'ndcg@{K}': ndcg, f'cov@{K}': cov}
        

    def optimize(
        self,
        bandit_feedback: Dict[str, np.ndarray],
        val_size: float = 0.3,
        param_borders: Optional[Dict[str, List[Any]]] = None,
        criterion: str = "ipw",
        budget: int = 10,
        new_study: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Optimize model parameters using optuna.
        Optimization is carried out over the IPW/DR/DM scores(IPW by default).

        :param bandit_feedback: Bandit log data with fields
            ``[action, reward, context, action_context,
            n_rounds, n_actions, position, pscore]`` as in OpenBanditPipeline.

        :param val_size: Size of validation subset.

        :param param_borders: Dictionary of parameter names with pair of borders
                              for the parameters optimization algorithm.

        :param criterion: Score for optimization. Available are `ipw`, `dr` and `dm`.

        :param budget: Number of trials for the optimization algorithm.

        :param new_study: Flag to create new study or not for optuna.

        :return: Dictionary of parameter names with optimal value of corresponding parameter.
        """
        bandit_feedback_train, bandit_feedback_val = split_bandit_feedback(bandit_feedback, val_size)

        if self.replay_model._search_space is None:
            self.logger.warning("%s has no hyper parameters to optimize", str(self))
            return None

        if self._study is None or new_study:
            self._study = create_study(direction="maximize", sampler=TPESampler())

        search_space = self.replay_model._prepare_param_borders(param_borders)
        if self.replay_model._init_params_in_search_space(search_space) and not self.replay_model._params_tried():
            self._study.enqueue_trial(self.replay_model._init_args)

        objective = self._objective(
            search_space=search_space,
            bandit_feedback_train=bandit_feedback_train,
            bandit_feedback_val=bandit_feedback_val,
            learner=self,
            criterion=criterion,
            k=self.len_list,
        )

        self._study.optimize(objective, budget)
        best_params = self._study.best_params
        self.replay_model.set_params(**best_params)
        return best_params