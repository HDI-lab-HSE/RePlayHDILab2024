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

        # self.data: pd.DataFrame

        self.load_raw_data()
        self.pre_process()

    @property
    def n_rounds(self) -> int:
        """Size of the logged bandit data."""
        return self.data.shape[0]

    @property
    def n_actions(self) -> int:
        """Number of actions."""
        return int(self.action.max() + 1)

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
        """Load raw open bandit dataset."""


        preparator = DataPreparator()

        log = preparator.transform(columns_mapping={'user_id': 'user_id',
                                      'item_id': 'item_id',
                                      'relevance': 'rating',
                                      'timestamp': 'timestamp'
                                     }, data=self.dataset.ratings.iloc[:5000])
        
        only_positives_log = log.filter(sf.col('relevance') >= 3).withColumn('relevance', sf.lit(1))

        only_negatives_log = log.filter(sf.col('relevance') < 3).withColumn('relevance', sf.lit(0.))

        log = (only_positives_log.union(only_negatives_log))



        indexer = Indexer(user_col='user_id', item_col='item_id')

        indexer.fit(users=log.select('user_id'), items=log.select('item_id'))

        self.data = indexer.transform(df=log).toPandas()

        self.action = np.array(self.data['item_idx'].tolist())

        self.position = np.zeros(self.data.shape[0]).astype(int)


        item_features_original = preparator.transform(columns_mapping={'item_id': 'item_id'}, data=self.dataset.items)
        item_features = indexer.transform(df=item_features_original)
        year = item_features.withColumn('year', sf.substring(sf.col('title'), -5, 4).astype(st.IntegerType())).select('item_idx', 'year')
        genres = (item_features.select("item_idx", sf.split("genres", "\|").alias("genres")))
        genres_list = (genres.select(sf.explode("genres").alias("genre")).distinct().filter('genre <> "(no genres listed)"').toPandas()["genre"].tolist())
        item_features = genres
        for genre in genres_list:
            item_features = item_features.withColumn(
                genre,
                sf.array_contains(sf.col("genres"), genre).astype(IntegerType())
            )
        item_features = item_features.drop("genres").cache().toPandas()
        # item_features = item_features.join(year, on='item_idx', how='inner')
        # item_features.cache()

        # item_features = pd.concat([item_features.drop("year").toPandas(), pd.get_dummies(item_features.toPandas().year)], axis = 1)
        self.action_context = item_features.drop(columns=['item_idx'],).to_numpy()


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

        user_df = pd.concat([user_features.drop(columnsToEncode, 1), pd.DataFrame(myEncoder.transform(user_features[columnsToEncode]), 
                                                columns = myEncoder.get_feature_names(columnsToEncode))], axis=1).reindex()

        user_features = user_df.drop(columns=['user_idx'],).to_numpy()
        user_idxs = self.data['user_idx'].to_numpy()

        self.context = user_features[user_idxs]
        print(self.action.shape)
        print(self.context.shape)

        self.data.sort_values("timestamp", inplace=True)
        self.reward = np.array(self.data['relevance'].tolist())

    def pre_process(self) -> None:
        """Preprocess raw open bandit dataset.

        Note
        -----
        This is the default feature engineering and please override this method to
        implement your own preprocessing.
        see https://github.com/st-tech/zr-obp/blob/master/examples/examples_with_obd/custom_dataset.py for example.

        """

        model = LogisticRegression(max_iter=100, random_state=12345, n_jobs=-1)
        print('fit started')
        model.fit(self.context, self.action)
        print('predict started')

        self.pscore = []
        for i in tqdm(range(self.n_rounds)):
            self.pscore.append(model.predict_proba([self.context[i]])[0][self.action[i]])

        self.pscore = np.array(self.pscore)




        # user_cols = self.data.columns.str.contains("user_feature")
        # self.context = pd.get_dummies(
        #     self.data.loc[:, user_cols], drop_first=True
        # ).values
        # item_feature_0 = self.item_context["item_feature_0"].to_frame()
        # item_feature_cat = self.item_context.drop(
        #     columns=["item_id", "item_feature_0"], axis=1
        # ).apply(LabelEncoder().fit_transform)
        # self.action_context = pd.concat(
        #     objs=[item_feature_cat, item_feature_0], axis=1
        # ).values

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
            n_rounds_train = np.int32(self.n_rounds * (1.0 - test_size))
            bandit_feedback_train = dict(
                n_rounds=n_rounds_train,
                n_actions=self.n_actions,
                action=self.action[:n_rounds_train],
                position=self.position[:n_rounds_train],
                reward=self.reward[:n_rounds_train],
                pscore=self.pscore[:n_rounds_train],
                context=self.context[:n_rounds_train],
                action_context=self.action_context,
            )
            bandit_feedback_test = dict(
                n_rounds=(self.n_rounds - n_rounds_train),
                n_actions=self.n_actions,
                action=self.action[n_rounds_train:],
                position=self.position[n_rounds_train:],
                reward=self.reward[n_rounds_train:],
                pscore=self.pscore[n_rounds_train:],
                context=self.context[n_rounds_train:],
                action_context=self.action_context,
            )
            return bandit_feedback_train, bandit_feedback_test
        else:
            return dict(
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
