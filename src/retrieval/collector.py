import pandas as pd
import numpy as np
from typing import List
from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer


class RuleCollector:
    """Collect retrieval candidates by rules"""

    def collect(
        self,
        valid: pd.DataFrame,
        customer_list: np.ndarray,
        rules: List,
        filters: List = [],
        min_pos_rate: float = 0.01,
        item_id: str = "article_id",
        compress=True,
    ) -> pd.DataFrame:
        """Collect retreival results

        Parameters
        ----------
        valid : pd.DataFrame
            Valid set.
        customer_list : np.ndarray
            Target customer list.
        rules : List
            List of rules to retrieve items.
        filters : List, optional
            Filters to remove some retrieved items, by default ``[]``.
        min_pos_rate : float, optional
            Minimum positive rate of the generated candidates, by default ``0.01``.
        item_id : str, optional
            Item unit, by default ``"article_id"``.
        compress : bool, optional
            Whether to compress the candidate items into a list, by default ``True``.

        Returns
        -------
        pd.DataFrame
            Dataframe of candidate items.
        """
        customer_list = np.array(customer_list)

        # * prepare valid data to calculate positive rate of retrieved items
        label = valid[["customer_id", item_id]]
        label.columns = ["customer_id", "label_item"]

        # * Get items to be removed.
        rm_items = []
        for filter in filters:
            rm_items += filter.retrieve()

        # * Retrieve items for each user.
        pred_df = None
        for rule in tqdm(rules, "Retrieve items by rules"):
            items = rule.retrieve()
            scaler = QuantileTransformer(output_distribution="normal")
            items["score"] = scaler.fit_transform(items["score"].values.reshape(-1, 1))
            items = items.loc[~items[item_id].isin(rm_items)]

            # * Calculate positive rate
            items = items.sort_values(by="score", ascending=False).reset_index(
                drop=True
            )

            if label.shape[0] != 0:
                # * not test set, calculate the positive rate
                tmp_items = items.merge(label, on=["customer_id"], how="left")
                tmp_items = tmp_items[tmp_items["label_item"].notnull()]
                tmp_items["label"] = tmp_items.apply(
                    lambda x: 1 if x[item_id] in x["label_item"] else 0, axis=1
                )
                pos_rate = tmp_items["label"].mean()

                # * if the positive rate is too low, we need to find the `n` that
                # * has the highest positive rate.
                if pos_rate < min_pos_rate:
                    tmp_items["rank"] = tmp_items.groupby("customer_id")["score"].rank(
                        ascending=False
                    )

                    rank = tmp_items["rank"].max()
                    best_pos_rate = pos_rate
                    best_rank = rank
                    while rank > 0:
                        tmp_pos_rate = tmp_items.loc[
                            tmp_items["rank"] <= rank, "label"
                        ].mean()
                        if tmp_pos_rate > best_pos_rate:
                            best_pos_rate = tmp_pos_rate
                            best_rank = rank
                        if tmp_pos_rate > min_pos_rate:
                            break
                        rank -= 1

                    # * if the best positive rate is still too low, we have to skip this rule.
                    if best_pos_rate < min_pos_rate:
                        print("skip")
                        continue

                    print(f"TOP{best_rank} Positive rate: {best_pos_rate:.5f}")
                    items = tmp_items.loc[tmp_items["rank"] <= best_rank]
                    items.drop(["rank", "label_item", "label"], axis=1, inplace=True)
                else:
                    print(f"Positive rate: {pos_rate:.5f}")

            # * Merge with previous results
            pred_df = pd.concat([pred_df, items], ignore_index=True)

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
