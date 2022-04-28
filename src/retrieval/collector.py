import pandas as pd
import numpy as np
from typing import List
from tqdm import tqdm
from .rules import (
    PersonalRetrieveRule,
    UserGroupRetrieveRule,
    GlobalRetrieveRule,
    FilterRule,
)
from sklearn.preprocessing import QuantileTransformer, StandardScaler


class RuleCollector:
    """Collect retrieval candidates by rules"""

    def collect(
        self,
        customer_list: np.ndarray,
        rules: List,
        filters: List = None,
        item_id: str = "article_id",
        compress=True,
    ) -> pd.DataFrame:
        """Collect retreival results

        Parameters
        ----------
        customer_list : np.ndarray
            Target customer list.
        rules : List
            List of rules to retrieve items.
        filters : List, optional
            Filters to remove some retrieved items, by default ``None``.
        item_id : str, optional
            Item unit, by default ``"article_id"``.
        compress : bool, optional
            Whether to compress the candidate items into a list, by default ``True``.

        Returns
        -------
        pd.DataFrame
            Dataframe of candidate items.
        """
        # * Check parameters
        customer_list = np.array(customer_list)
        self._check_rule(rules)
        if filters is not None:
            self._check_filter(filters)
        else:
            filters = []

        # * Get items to be removed.
        rm_items = []
        for filter in filters:
            rm_items += filter.retrieve()

        # * Retrieve items for each user.
        pred_df = None
        for rule in tqdm(rules, "Retrieve items by rules"):
            items = rule.retrieve()
            scaler = QuantileTransformer(output_distribution="normal")
            # scaler = StandardScaler()
            items["score"] = scaler.fit_transform(items["score"].values.reshape(-1, 1))
            items = items.loc[~items[item_id].isin(rm_items)].reset_index(drop=True)

            pred_df = pd.concat([pred_df, items], ignore_index=True)

        # pred_df = pred_df.sort_values(by=["customer_id", "score"]).reset_index(
        #     drop=True
        # )
        # pred_df = pred_df.drop_duplicates(["customer_id", item_id], keep="last")

        # * Compress the result.
        if compress:
            pred_df = pred_df.groupby("customer_id")[item_id].apply(list).reset_index()
            output_df = pd.DataFrame({"customer_id": customer_list})
            output_df = pd.merge(output_df, pred_df, on=["customer_id"], how="left")

            # * Impute with empty list.
            mask = output_df[item_id].isna()
            output_df.loc[mask, item_id] = pd.Series([[]] * mask.sum()).values

            return output_df[["customer_id", item_id]]

        return pred_df

    @staticmethod
    def _check_rule(rules: List) -> None:
        for rule in rules:
            if (
                not isinstance(rule, PersonalRetrieveRule)
                and not isinstance(rule, GlobalRetrieveRule)
                and not isinstance(rule, UserGroupRetrieveRule)
            ):
                raise TypeError(
                    "Rule must be `PersonalRetrieveRule` or `GlobalRetrieveRule`"
                )

    @staticmethod
    def _check_filter(filters: List) -> None:
        for filter in filters:
            if not isinstance(filter, FilterRule):
                raise TypeError("Filter must be `FilterRule`")
