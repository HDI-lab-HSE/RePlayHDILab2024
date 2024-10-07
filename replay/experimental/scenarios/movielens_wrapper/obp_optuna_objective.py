from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
from obp.ope import DirectMethod, DoublyRobust, InverseProbabilityWeighting, OffPolicyEvaluation
from optuna import Trial

from replay.experimental.scenarios.obp_wrapper.utils import get_est_rewards_by_reg
from replay.optimization.optuna_objective import ObjectiveWrapper, suggest_params
from replay.models import UCB, Wilson, RandomRec, LinUCB, LinTS
from replay.utils.spark_utils import convert2spark
from replay.experimental.scenarios.movielens_wrapper.utils import bandit_subset
from tqdm import tqdm

def get_dist(learner, bandit_feedback_test):
    all_action_dist = np.zeros((bandit_feedback_test["n_rounds"], bandit_feedback_test["n_actions"], 1))
    if isinstance(learner.replay_model, (LinUCB, LinTS)):
        log_distinct = bandit_feedback_test['log'].toPandas().drop_duplicates(subset=["user_idx"], keep='first')
        users_all = bandit_feedback_test['log'].toPandas()['user_idx'].tolist()
        batch_size = 10
        num_batchs = log_distinct.shape[0] // batch_size
        for batch_idx in tqdm(range(num_batchs+1)):
            j = min((batch_idx+1)*batch_size, log_distinct.shape[0])
            if j == batch_idx*batch_size:
                break
            log_subset = log_distinct.iloc[batch_idx*batch_size: j]
            n_rounds = log_subset.shape[0]
            
            action_dist = learner.predict(n_rounds, convert2spark(log_subset).select('user_idx'))

            users_distinct = log_subset['user_idx'].tolist()

            user2ind = {}
            for i in range(n_rounds):
                user2ind[users_distinct[i]] = i

            for i in range(bandit_feedback_test["n_rounds"]):
                if users_all[i] in users_distinct:
                    all_action_dist[i] = action_dist[user2ind[users_all[i]]]

    else:
        batch_size = 300
        num_batchs = bandit_feedback_test["n_rounds"] // batch_size
        for batch_idx in tqdm(range(num_batchs+1)):
            j = min((batch_idx+1)*batch_size, bandit_feedback_test["n_rounds"])
            if j == batch_idx*batch_size:
                break
            bandit_feedback_subset = bandit_subset([batch_idx*batch_size, j], bandit_feedback_test) #The first parameter is a slice of subset [a, b]
            action_dist = learner.predict(bandit_feedback_subset["n_rounds"], bandit_feedback_subset["log"].select('user_idx'))
            all_action_dist[batch_idx*batch_size:j] = action_dist
    return all_action_dist


def obp_objective_calculator(
    trial: Trial,
    search_space: Dict[str, List[Optional[Any]]],
    bandit_feedback_train: Dict[str, np.ndarray],
    bandit_feedback_val: Dict[str, np.ndarray],
    learner,
    criterion: str,
    k: int,
) -> float:
    """
    Sample parameters and calculate criterion value
    :param trial: optuna trial
    :param search_space: hyper parameter search space
    :bandit_feedback_train: dict with bandit train data
    :bandit_feedback_cal: dist with bandit validation data
    :param criterion: optimization metric
    :param k: length of a recommendation list
    :return: criterion value
    """

    params_for_trial = suggest_params(trial, search_space)
    learner.replay_model.set_params(**params_for_trial)

    learner.fit(
        bandit_feedback_train
    )
    
    action_dist = get_dist(learner, bandit_feedback_val)
    
    # action_dist = learner.predict(bandit_feedback_val["n_rounds"], bandit_feedback_val["context"])

    ope_estimator = None
    if criterion == "ipw":
        ope_estimator = InverseProbabilityWeighting()
    elif criterion == "dm":
        ope_estimator = DirectMethod()
    elif criterion == "dr":
        ope_estimator = DoublyRobust()
    else:
        msg = f"There is no criterion with name {criterion}"
        raise NotImplementedError(msg)

    ope = OffPolicyEvaluation(bandit_feedback=bandit_feedback_val, ope_estimators=[ope_estimator])

    estimated_rewards_by_reg_model = None
    if criterion in ("dm", "dr"):
        estimated_rewards_by_reg_model = get_est_rewards_by_reg(
            learner.n_actions, k, bandit_feedback_train, bandit_feedback_val
        )

    estimated_policy_value = ope.estimate_policy_values(
        action_dist=action_dist,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    )[criterion]

    return estimated_policy_value


OBPObjective = partial(ObjectiveWrapper, objective_calculator=obp_objective_calculator)
