from dataclasses import dataclass
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from abc import ABCMeta
from abc import abstractmethod




from logging import basicConfig
from logging import getLogger
from logging import INFO
from pathlib import Path
from typing import Optional
from typing import Tuple
from typing import Union

from scipy.stats import rankdata
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar

from typing import Dict
from typing import Union

import rs_datasets
from replay.experimental.preprocessing.data_preparator import Indexer, DataPreparator
from replay.splitters import TimeSplitter
from replay.utils.spark_utils import convert2spark
from pyspark.sql import functions as sf, types as st
from pyspark.sql.types import IntegerType


from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


class BaseBanditDataset(metaclass=ABCMeta):
    """Base Class for Synthetic Bandit Dataset."""

    @abstractmethod
    def obtain_batch_bandit_feedback(self) -> None:
        """Obtain batch logged bandit data."""
        raise NotImplementedError


class BaseRealBanditDataset(BaseBanditDataset):
    """Base Class for Real-World Bandit Dataset."""

    @abstractmethod
    def load_raw_data(self) -> None:
        """Load raw dataset."""
        raise NotImplementedError

    @abstractmethod
    def pre_process(self) -> None:
        """Preprocess raw dataset."""
        raise NotImplementedError
    



# dataset
BanditFeedback = Dict[str, Union[int, np.ndarray]]


logger = getLogger(__name__)
basicConfig(level=INFO)


@dataclass
class MovielensBanditDataset(BaseRealBanditDataset):

    dataset: rs_datasets.movielens.MovieLens

    def __post_init__(self) -> None:
        self.load_raw_data()
        self.pre_process()

    @property
    def n_rounds(self) -> int:
        """Size of the logged bandit data."""
        return self.context.shape[0]

    @property
    def n_actions(self) -> int:
        """Number of actions."""
        return self.item_features.count()

    @property
    def dim_context(self) -> int:
        """Dimensions of context vectors."""
        return self.context.shape[1]

    @property
    def len_list(self) -> int:
        """Length of recommendation lists, slate size."""
        return int(self.position.max() + 1)

    @classmethod
    def calc_on_policy_policy_value_estimate(
        cls,
        behavior_policy: str,
        campaign: str,
        data_path: Optional[Path] = None,
        test_size: float = 0.3,
        is_timeseries_split: bool = False,
    ) -> float:
 
        bandit_feedback = cls(
            behavior_policy=behavior_policy, campaign=campaign, data_path=data_path
        ).obtain_batch_bandit_feedback(
            test_size=test_size, is_timeseries_split=is_timeseries_split
        )
        if is_timeseries_split:
            bandit_feedback_test = bandit_feedback[1]
        else:
            bandit_feedback_test = bandit_feedback
        return bandit_feedback_test["reward"].mean()

    def load_raw_data(self) -> None:

        preparator = DataPreparator()

        #load logs
        log = preparator.transform(columns_mapping={'user_id': 'user_id',
                                                    'item_id': 'item_id',
                                                    'relevance': 'rating',
                                                    'timestamp': 'timestamp'}, data=self.dataset.ratings.iloc[:50000])

        indexer = Indexer(user_col='user_id', item_col='item_id')
        indexer.fit(users=log.select('user_id'), items=log.select('item_id'))
        log = indexer.transform(df=log).toPandas()
        log['relevance'] = log['relevance'].apply(lambda x: int(x>=3))
        self.log = convert2spark(log)

        self.reward = np.array(log['relevance'].tolist())
        self.action = np.array(log['item_idx'].tolist())
        self.position = np.zeros(log.shape[0]).astype(int)

        #load item features and action_context
        item_features_original = preparator.transform(columns_mapping={'item_id': 'item_id'}, data=self.dataset.items)
        item_features = indexer.transform(df=item_features_original)
        genres = (item_features.select("item_idx", sf.split("genres", "\|").alias("genres")))
        genres_list = (genres.select(sf.explode("genres").alias("genre")).distinct().filter('genre <> "(no genres listed)"').toPandas()["genre"].tolist())
        item_features = genres
        for genre in genres_list:
            item_features = item_features.withColumn(genre, sf.array_contains(sf.col("genres"), genre).astype(IntegerType()))

        self.item_features = item_features.drop("genres").cache()
        self.action_context = self.item_features.toPandas().drop(columns=['item_idx'],).to_numpy()

        #load user features and context
        user_features_original = preparator.transform(columns_mapping={'user_id': 'user_id'}, data=self.dataset.users)
        user_features = indexer.transform(df=user_features_original)
        user_features = user_features.toPandas()

        user_features = user_features.drop(columns=['zip_code'])
        bins = [0, 20, 30, 40, 50, 60, np.inf]
        names = ['<20', '20-29', '30-39','40-49', '51-60', '60+']

        user_features['agegroup'] = pd.cut(user_features['age'], bins, labels=names)
        user_features = user_features.drop(["age"], axis = 1)

        columnsToEncode = ["agegroup","gender","occupation"]
        myEncoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        myEncoder.fit(user_features[columnsToEncode])

        self.user_features = pd.concat([user_features.drop(columnsToEncode, 1), pd.DataFrame(myEncoder.transform(user_features[columnsToEncode]), 
                                                columns = myEncoder.get_feature_names(columnsToEncode))], axis=1).reindex()
        self.context = self.user_features.drop(columns=['user_idx'],).to_numpy()[log['user_idx'].to_numpy()]
        self.user_features = convert2spark(self.user_features)

    def pre_process(self) -> None:

        self.model = LogisticRegression(max_iter=100, random_state=12345, n_jobs=-1)
        print('fit started')
        self.model.fit(self.context, self.action)
        print('predict started')

        self.pscore = []
        for i in tqdm(range(self.n_rounds)):
            self.pscore.append(self.model.predict_proba([self.context[i]])[0][self.action[i]])

        self.pscore = np.array(self.pscore)

    def obtain_batch_bandit_feedback(
        self, test_size: float = 0.3, is_timeseries_split: bool = False
    ) -> Union[BanditFeedback, Tuple[BanditFeedback, BanditFeedback]]:

        if not isinstance(is_timeseries_split, bool):
            raise TypeError(
                f"`is_timeseries_split` must be a bool, but {type(is_timeseries_split)} is given"
            )

        if is_timeseries_split:
            check_scalar(
                test_size,
                name="target_size",
                target_type=(float),
                min_val=0.0,
                max_val=1.0,
            )

            train_spl = TimeSplitter(
                time_threshold=test_size,
                drop_cold_items=True,
                drop_cold_users=True,
                query_column="user_idx",
                item_column="item_idx",
            )

            train_log, test_log = train_spl.split(self.log)


            n_rounds_train = train_log.count()
            bandit_feedback_train = dict(
                log=train_log,
                item_features=self.item_features,
                user_features=self.user_features,
                n_rounds=n_rounds_train,
                n_actions=self.n_actions,
                action=self.action[:n_rounds_train],
                position=self.position[:n_rounds_train],
                reward=self.reward[:n_rounds_train],
                pscore=self.pscore[:n_rounds_train],
                context=self.context[:n_rounds_train],
                action_context=self.action_context,
            )

            test_log = test_log.toPandas().drop_duplicates(subset=["user_idx"], keep='first')
            test_action = np.array(test_log['item_idx'].tolist())
            test_reward = np.array(test_log['relevance'].tolist())
            test_context = self.user_features.toPandas().drop(columns=['user_idx'],).to_numpy()[test_log['user_idx'].to_numpy()]
            test_pscore = []
            for i in tqdm(range(test_log.shape[0])):
                test_pscore.append(self.model.predict_proba([test_context[i]])[0][test_action[i]])
            test_position = np.zeros(test_log.shape[0]).astype(int)

            test_pscore = np.array(test_pscore)


            bandit_feedback_test = dict(
                log=convert2spark(test_log),
                item_features=self.item_features,
                user_features=self.user_features,
                n_rounds=test_log.shape[0],
                n_actions=self.n_actions,
                action=test_action,
                position=test_position,
                reward=test_reward,
                pscore=test_pscore,
                context=test_context,
                action_context=self.action_context,
            )
            return bandit_feedback_train, bandit_feedback_test
        else:
            return dict(
                log=self.log,
                n_rounds=self.n_rounds,
                n_actions=self.n_actions,
                action=self.action,
                position=self.position,
                reward=self.reward,
                pscore=self.pscore,
                context=self.context,
                action_context=self.action_context,
            )

    def sample_bootstrap_bandit_feedback(
        self,
        sample_size: Optional[int] = None,
        test_size: float = 0.3,
        is_timeseries_split: bool = False,
        random_state: Optional[int] = None,
    ) -> BanditFeedback:
            
        if is_timeseries_split:
            bandit_feedback = self.obtain_batch_bandit_feedback(
                test_size=test_size, is_timeseries_split=is_timeseries_split
            )[0]
        else:
            bandit_feedback = self.obtain_batch_bandit_feedback(
                test_size=test_size, is_timeseries_split=is_timeseries_split
            )
        n_rounds = bandit_feedback["n_rounds"]
        if sample_size is None:
            sample_size = bandit_feedback["n_rounds"]
        else:
            check_scalar(
                sample_size,
                name="sample_size",
                target_type=(int),
                min_val=0,
                max_val=n_rounds,
            )
        random_ = check_random_state(random_state)
        bootstrap_idx = random_.choice(
            np.arange(n_rounds), size=sample_size, replace=True
        )
        for key_ in ["action", "position", "reward", "pscore", "context"]:
            bandit_feedback[key_] = bandit_feedback[key_][bootstrap_idx]
        bandit_feedback["n_rounds"] = sample_size
        return bandit_feedback
