import pandas as pd
import numpy as np
from typing import List
from tqdm import tqdm
from .rules import PersonalRetrieveRule, GlobalRetrieveRule, FilterRule


class RuleCollector:
    def collect(
        self,
        customer_list: np.ndarray,
        rules: List,
        filters: List = None,
        item_id: str = "article_id",
        compress=True,
    ) -> pd.DataFrame:
        """Use rules to Retrieve items for each user

        Args:
            customer_list (np.ndarray): target customer list
            rules (List): rules to Retrieve items
            filter (RetrieveRule, optional): filter to remove some Retrieveed items. Defaults to None.

        Returns:
            pd.DataFrame: prediction
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
            items = items.loc[~items[item_id].isin(rm_items)].reset_index(drop=True)

            if isinstance(rule, GlobalRetrieveRule):
                # * add `customer_id` to `GlobalRetrieveRule` results
                num_item = items.shape[0]
                num_user = customer_list.shape[0]

                tmp_user = np.repeat(customer_list, num_item)
                tmp_df = items.iloc[np.tile(np.arange(num_item), num_user)]
                tmp_df = tmp_df.reset_index(drop=True)
                tmp_df["customer_id"] = tmp_user

            else:  # * PersonalRetrieveRule
                tmp_df = items

            pred_df = pd.concat([pred_df, tmp_df], ignore_index=True)

        # pred_df = pred_df.sort_values(by=["customer_id", "rank"]).reset_index(drop=True)
        pred_df = pred_df.drop_duplicates(["customer_id", item_id], keep="first")

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
            if not isinstance(rule, PersonalRetrieveRule) and not isinstance(
                rule, GlobalRetrieveRule
            ):
                raise TypeError(
                    "Rule must be `PersonalRetrieveRule` or `GlobalRetrieveRule`"
                )

    @staticmethod
    def _check_filter(filters: List) -> None:
        for filter in filters:
            if not isinstance(filter, FilterRule):
                raise TypeError("Filter must be `FilterRule`")
