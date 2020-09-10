"""
Класс, реализующий стэккинг моделей.
"""
import logging
from functools import reduce
from operator import add
from typing import List, Optional

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.wrapper import JavaEstimator
from pyspark.sql import DataFrame
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.sql import functions as sf
import nevergrad as ng
import numpy as np
from tqdm import tqdm

from replay.constants import BASE_FIELDS, SCHEMA
from replay.metrics import NDCG
from replay.models.base_rec import Recommender
from replay.session_handler import State
from replay.splitters import k_folds
from replay.utils import get_top_k_recs


class Stack(Recommender):
    """Стэк базовых моделей возвращает свои скоры, которые взвешиваются,
    чтобы получить новое ранжирование."""

    def __init__(
        self,
        models: List[Recommender],
        n_folds: Optional[int] = 5,
        budget: Optional[int] = 30,
    ):
        """
        :param models: список инициализированных моделей
        :param n_folds: количество фолдов для обучения верхней модели,
            параметры смешения будут определены по среднему качеству на фолдах.
        :param budget: количество попыток найти вариант смешения моделей
        """
        self.models = models
        State()
        self.n_folds = n_folds
        self.budget = budget
        self._logger = logging.getLogger("replay")

    # pylint: disable=too-many-locals
    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        top_train = []
        top_test = []
        # pylint: disable=invalid-name
        df = log.withColumnRenamed("user_idx", "user_id").withColumnRenamed(
            "item_idx", "item_id"
        )
        for i, (train, test) in enumerate(k_folds(df, self.n_folds)):
            self._logger.info("Processing fold #%d", i)
            fold_train = State().session.createDataFrame(
                data=[], schema=SCHEMA
            )
            test_items = test.select("item_id").distinct()
            train_items = train.select("item_id").distinct()
            items_pos = test_items.join(train_items, on="item_id", how="inner")
            if items_pos.count() == 0:
                self._logger.info(
                    "Bad split, no positive examples, skipping..."
                )
                continue
            n_pos = (
                test.groupBy("user_id")
                .count()
                .agg({"count": "max"})
                .collect()[0][0]
            )
            for model in self.models:
                scores = model.fit_predict(train, k=n_pos * 2,)
                scores = scores.withColumnRenamed("relevance", str(model))
                fold_train = fold_train.join(
                    scores, on=["user_id", "item_id"], how="outer"
                ).fillna(0)
            top_train.append(fold_train)
            top_test.append(test)

        feature_cols = [str(model) for model in self.models]
        # pylint: disable=attribute-defined-outside-init
        self.top_train = top_train

        coefs = {
            model: ng.p.Scalar(lower=0, upper=1) for model in feature_cols
        }
        parametrization = ng.p.Instrumentation(**coefs)
        optimizer = ng.optimizers.OnePlusOne(
            parametrization=parametrization, budget=self.budget
        )
        base = [dict(zip(feature_cols, vals)) for vals in np.eye(len(feature_cols))]
        for one_model in base:
            optimizer.suggest(**one_model)
        for _ in tqdm(range(optimizer.budget)):
            weights = optimizer.ask()
            ranking = [
                NDCG()(rerank(pred, **weights.kwargs), true, 50)
                for pred, true in zip(top_train, top_test)
            ]
            loss = -np.mean(ranking)
            optimizer.tell(weights, loss)

        # pylint: disable=attribute-defined-outside-init
        self.params = optimizer.provide_recommendation().kwargs
        s = np.array(list(self.params.values()))
        if (s == 1).sum() == 1 and s.sum() == 1:
            name = [name for name in feature_cols if self.params[name] == 1][0]
            self._logger.warn("Could not find combination to improve quality, %s works best on its own", name)
        for model in self.models:
            model.fit(df)

    # pylint: disable=too-many-arguments
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
        top = State().session.createDataFrame(data=[], schema=SCHEMA)
        top = top.withColumnRenamed("user_id", "user_idx").withColumnRenamed(
            "item_id", "item_idx"
        )
        top = top.withColumn("user_idx", top["user_idx"].cast("integer"))
        top = top.withColumn("item_idx", top["item_idx"].cast("integer"))
        for model in self.models:
            # pylint: disable=protected-access
            scores = model._predict(
                log,
                k,
                users,
                items,
                user_features,
                item_features,
                filter_seen_items,
            )
            if filter_seen_items:
                scores = self._mark_seen_items(
                    scores, self._convert_index(log)
                )
            scores = scores.withColumn(
                "relevance",
                sf.when(scores["relevance"] < 0, None).otherwise(
                    scores["relevance"]
                ),
            )
            scores = scores.withColumnRenamed("relevance", str(model))
            top = top.join(scores, on=["user_idx", "item_idx"], how="outer").fillna(0)
        feature_cols = [str(model) for model in self.models]
        pred = rerank(top, **self.params)
        pred = pred.drop(*feature_cols)
        return pred


def rerank(df: DataFrame, **kwargs) -> DataFrame:
    """Добавляет колонку relevance линейной комбинацией колонок с весами из kwargs"""
    res = df.withColumn(
        "relevance",
        reduce(add, [sf.col(col) * weight for col, weight in kwargs.items()]),
    )
    res = res.orderBy("relevance", ascending=False)
    return res
