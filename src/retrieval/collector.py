import pandas as pd
import numpy as np
from typing import List
from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from ..utils import calc_valid_date


class RuleCollector:
    """Collect retrieval candidates by rules"""

    def collect(
        self,
        week_num: int,
        trans_df: pd.DataFrame,
        customer_list: np.ndarray,
        rules: List,
        filters: List = [],
        min_pos_rate: float = 0.01,
        item_id: str = "article_id",
        norm: bool = True,
        norm_type: str = "quantile",
        compress=True,
    ) -> pd.DataFrame:
        """Collect retreival results

        Parameters
        ----------
        week_num : int
            Week number.
        trans_df : pd.DataFrame
            Transaction dataframe.
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
        norm : bool, optional
            Whether to normalize the score, by default ``True``.
        norm_type : str, optional
            Normalization method, by default ``"quantile"``.
        compress : bool, optional
            Whether to compress the candidate items into a list, by default ``True``.

        Returns
        -------
        pd.DataFrame
            Dataframe of candidate items.
        """
        customer_list = np.array(customer_list)

        # * prepare valid data to calculate positive rate of retrieved items
        start_date, end_date = calc_valid_date(week_num)
        mask = (start_date <= trans_df["t_dat"]) & (trans_df["t_dat"] < end_date)
        label = trans_df.loc[mask, ["customer_id", "article_id"]]
        label = label.drop_duplicates(["customer_id", "article_id"])
        label["label"] = 1

        # * Get items to be removed.
        rm_items = []
        for filter in filters:
            rm_items += filter.retrieve()

        # * Retrieve items for each user.
        pred_df = None
        for rule in tqdm(rules, "Retrieve items by rules"):
            items = rule.retrieve()
            if norm:
                if norm_type == "quantile":
                    scaler = QuantileTransformer(output_distribution="normal")
                elif norm_type == "minmax":
                    scaler = MinMaxScaler()
                items["score"] = scaler.fit_transform(
                    items["score"].values.reshape(-1, 1)
                )
            items = items.loc[~items[item_id].isin(rm_items)]

            if label.shape[0] != 0:
                # * not test set, calculate the positive rate
                label_customers = label["customer_id"].unique()
                tmp_items = items[items["customer_id"].isin(label_customers)]
                tmp_items = tmp_items.merge(
                    label, on=["customer_id", "article_id"], how="left"
                )
                tmp_items["label"] = tmp_items["label"].fillna(0)
                pos_rate = tmp_items["label"].mean()

                # * if the positive rate is too low, we need to find the `n` that
                # * has the highest positive rate.
                if pos_rate < min_pos_rate:
                    tmp_items = tmp_items.sort_values(by="score", ascending=False)
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
                    items.drop(["rank", "label"], axis=1, inplace=True)
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
