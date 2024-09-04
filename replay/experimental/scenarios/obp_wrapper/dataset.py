from dataclasses import dataclass
from obp.dataset import OpenBanditDataset
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

@dataclass
class ModifiedOpenBanditDataset(OpenBanditDataset):
    user_item_affinity_feat: bool = False 
    user_feat: bool = True
    @property
    def n_actions(self) -> int:
        """Number of actions."""
        return int(self.action.max() + 1)

    def pre_process(self) -> None:
        """Preprocess raw open bandit dataset."""
        if self.user_feat and self.user_item_affinity_feat:
            user_cols = self.data.columns.str.contains("user_feature")
            user_context = pd.get_dummies(
                self.data.loc[:, user_cols], drop_first=True
            ).values
            user_item_cols = self.data.columns.str.contains("user-item_affinity")
            user_item_context = self.data.loc[:,user_item_cols].values
            #user_item_context = np.column_stack((user_item_context, np.ones(user_item_context.shape[0])))
            self.context = np.c_[user_context, user_item_context]
            #self.context = np.einsum('nk,nl->nkl', user_context, user_item_context).reshape(user_context.shape[0],-1)
        elif self.user_feat:
            user_cols = self.data.columns.str.contains("user_feature")
            self.context = pd.get_dummies(
                self.data.loc[:, user_cols], drop_first=True
            ).values
        elif self.user_item_affinity_feat:
            user_item_cols = self.data.columns.str.contains("user-item_affinity")
            self.context = self.data.loc[:,user_item_cols].values
        else:
            raise ValueError("one of the `user_item_affinity_feat` or `user_feat` must be True.")
        item_feature_0 = self.item_context["item_feature_0"].values
        item_feature_cat = self.item_context.drop(
            columns=["item_id", "item_feature_0"], axis=1
        ).apply(LabelEncoder().fit_transform)
        item_feature_cat = OneHotEncoder(sparse=False, drop="first").fit_transform(item_feature_cat)
        item_feat = np.c_[item_feature_cat, item_feature_0]
        # self.action_context = np.zeros(((self.position.max()+1)* self.n_actions, item_feat.shape[1]+1))
        # for i in range(self.position.max()+1):
        #     self.action_context[i*self.n_actions:(i+1)*self.n_actions,1:] = item_feat
        #     self.action_context[i*self.n_actions:(i+1)*self.n_actions,0] = np.ones(self.n_actions)*i
        self.action_context = item_feat
            
        # pos = DataFrame(self.position)
        # self.action_context = (
        #     self.item_context.drop(columns=["item_id", "item_feature_0"], axis=1)
        #     .apply(LabelEncoder().fit_transform)
        #     .values
        # )
        # self.action_context = self.action_context[self.action]
        # self.action_context = np.c_[self.action_context, pos]

        # self.action = self.position * self.n_actions + self.action
        # self.position = np.zeros_like(self.position)
        # self.pscore /= 3