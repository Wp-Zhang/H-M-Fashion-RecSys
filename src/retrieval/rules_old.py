import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List
from tqdm import tqdm


class MatchRule(ABC):
    """Use rules to match items"""

    @abstractmethod
    def match(self) -> pd.DataFrame:
        """filter items

        Returns:
            pd.DataFrame: (customer,item) pairs
        """
        pass


class OrderHistoryMatch(MatchRule):
    """Match items from user history orders"""

    def __init__(
        self, trans_df, days: int = 7, n: int = None, item_id: str = "article_id"
    ):
        self.iid = item_id
        self.trans_df = trans_df[["t_dat", "customer_id", item_id]].copy()
        self.days = days
        self.n = n

    def match(self) -> pd.DataFrame:
        """Filter history purchase items in the
            last *days* days in bill history.

        Returns:
            pd.DataFrame: (customer, item) pairs
        """
        df = self.trans_df
        df = df.reset_index()
        df["t_dat"] = pd.to_datetime(df["t_dat"])

        tmp = df.groupby("customer_id").t_dat.max().reset_index()
        tmp.columns = ["customer_id", "max_dat"]
        res = df.merge(tmp, on=["customer_id"], how="left")
        res["diff_dat"] = (res.max_dat - res.t_dat).dt.days
        res = res.loc[res["diff_dat"] < self.days].reset_index(drop=True)

        res = (
            res[["customer_id", self.iid, "index"]]
            .groupby(["customer_id", self.iid], as_index=False)
            .last()
        )
        res = res.sort_values(by="index", ascending=False).reset_index(drop=True)
        res["rank"] = res.groupby(["customer_id"])["index"].rank(
            ascending=True, method="first"
        )
        if self.n is not None:
            res = res[res["rank"] <= self.n]
        res = res[["customer_id", self.iid]]

        res.columns = ["customer_id", "prediction"]

        return res


class TimeHistoryMatch(MatchRule):
    """Get popular items in the specified time window"""

    def __init__(self, trans_df, n: int = 12, item_id: str = "article_id"):
        self.iid = item_id
        self.trans_df = trans_df
        self.n = n  # top n items

    def match(self) -> List[int]:
        """Get popular items in the specified time window

        Returns:
            List[int]: top n popular items
        """
        df = self.trans_df
        tmp = df.drop_duplicates(["customer_id", self.iid])
        tmp["count"] = 1
        tmp = tmp.groupby([self.iid], as_index=False)["count"].sum()
        tmp = tmp.sort_values(by="count", ascending=False)

        return tmp[self.iid].values[: self.n]


class ItemPairMatch(MatchRule):
    def __init__(self, trans_df: pd.DataFrame, item_id: str = "article_id"):
        assert len(trans_df.columns) == 2, "Too Many Columns!"

        self.iid = item_id
        self.trans_df = trans_df.drop_duplicates()
        self.trans_df.columns = ["customer_id", self.iid]

    def _get_freq_pair(self) -> dict:
        """Generate dict of frequent item pairs in target time window

        Args:
            train_set (pd.DataFrame): transaction dataframe in target time window
            n (int, optional): search top *n* pairs for each article. Defaults to 3.

        Returns:
            dict: frequent item pairs
        """
        tmp = self.trans_df.copy()
        s = tmp[["customer_id", self.iid]].merge(
            tmp[["customer_id", self.iid]], on="customer_id"
        )
        s = s[s["article_id_x"] != s["article_id_y"]]
        s["count"] = 1
        s = s.groupby(["article_id_x", "article_id_y"], as_index=False)["count"].sum()
        s = s.sort_values("count", ascending=False).reset_index(drop=True)
        s = s.groupby("article_id_x")["article_id_y"].first()
        s = s.to_dict()

        return s

    def match(self) -> pd.DataFrame:
        pair = self._get_freq_pair()
        df = self.trans_df
        df[self.iid] = df[self.iid].map(pair)
        df = df.loc[df.article_id.notnull()]
        df = df.drop_duplicates(["customer_id", self.iid])
        df[self.iid] = df[self.iid].astype("int32")
        # df.columns = ['customer_id','ItemPairMatch']
        df.columns = ["customer_id", "prediction"]
        # df = df.groupby('customer_id')['ItemPairMatch'].apply(list).reset_index()
        return df


class SaleTrendMatch(MatchRule):
    def __init__(
        self, trans_df, days: int = 7, n: int = 12, item_id: str = "article_id"
    ):
        self.iid = item_id
        self.trans_df = trans_df[["customer_id", self.iid, "t_dat"]]
        self.days = days  # length of period
        self.n = n  # top n

    def match(self):
        item_sale = self.trans_df
        item_sale["t_dat"] = pd.to_datetime(item_sale["t_dat"])
        item_sale["dat_gap"] = (item_sale.t_dat.max() - item_sale.t_dat).dt.days
        item_sale = item_sale[item_sale["dat_gap"] <= 2 * self.days - 1]

        group_a = item_sale[item_sale["dat_gap"] > self.days - 1][[self.iid]]
        group_b = item_sale[item_sale["dat_gap"] <= self.days - 1][[self.iid]]

        group_a["count"] = 1
        group_b["count"] = 1
        group_a = group_a.groupby(self.iid)["count"].sum().reset_index()
        group_b = group_b.groupby(self.iid)["count"].sum().reset_index()

        log = pd.merge(group_b, group_a, on=[self.iid], how="left")
        log["count_x"] = log["count_x"].fillna(0)
        log["trend"] = (log["count_x"] - log["count_y"]) / log["count_y"]

        log = log[log["trend"] > 0.8]
        log = log.sort_values(by=["count_x", "trend"], ascending=False)

        return list(log[self.iid].values[: self.n])


class OrderHistoryDecayMatch(MatchRule):
    def __init__(
        self,
        trans_df,
        a=2.5e4,
        b=1.5e5,
        c=2e-1,
        d=1e3,
        n=12,
        item_id: str = "article_id",
    ):
        self.iid = item_id
        self.trans_df = trans_df[["customer_id", self.iid, "t_dat"]]
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.n = n

    def match(self):
        df = self.trans_df
        df["t_dat"] = pd.to_datetime(df["t_dat"])
        last_ts = df["t_dat"].max()
        df["dat_gap"] = (last_ts - df["t_dat"]).dt.days

        df["last_day"] = last_ts - (last_ts - df["t_dat"]).dt.floor(f"{self.n}D")
        period_sales = (
            df[["last_day", self.iid, "t_dat"]].groupby(["last_day", self.iid]).count()
        )
        period_sales = period_sales.rename(columns={"t_dat": "period_sale"})
        df = df.join(period_sales, on=["last_day", self.iid])

        period_sales = period_sales.reset_index().set_index(self.iid)
        last_day = last_ts.strftime("%Y-%m-%d")
        df = df.join(
            period_sales.loc[period_sales["last_day"] == last_day, ["period_sale"]],
            on=self.iid,
            rsuffix="_targ",
        )
        df["period_sale_targ"].fillna(0, inplace=True)
        df["quotient"] = df["period_sale_targ"] / df["period_sale"]
        del df["period_sale_targ"], df["period_sale"]

        df["dat_gap"][df["dat_gap"] < 1] = 1
        x = df["dat_gap"]
        df["value"] = self.a / np.sqrt(x) + self.b * np.exp(-self.c * x) - self.d
        df["value"][df["value"] < 0] = 0
        df["value"] = df["value"] * df["quotient"]

        df = df.groupby(["customer_id", self.iid], as_index=False)["value"].sum()
        df = df.sort_values(by="value", ascending=False).reset_index(drop=True)
        df = df.reset_index()

        df["rank"] = df.groupby(["customer_id"])["index"].rank(
            ascending=True, method="first"
        )
        df = df[df["rank"] <= self.n]
        df = df[["customer_id", self.iid]]
        df.columns = ["customer_id", "prediction"]

        return df


class TimeHistoryMatch2(MatchRule):
    def __init__(self, trans_df, n=12, item_id: str = "article_id"):
        self.iid = item_id
        self.trans_df = trans_df[["customer_id", self.iid, "t_dat"]]
        self.n = n

    def match(self):
        df = self.trans_df
        df["t_dat"] = pd.to_datetime(df["t_dat"])
        last_ts = df["t_dat"].max()
        df["dat_gap"] = (last_ts - df["t_dat"]).dt.days

        df["last_day"] = last_ts - (last_ts - df["t_dat"]).dt.floor(f"{self.n}D")
        period_sales = (
            df[["last_day", self.iid, "t_dat"]].groupby(["last_day", self.iid]).count()
        )
        period_sales = period_sales.rename(columns={"t_dat": "period_sale"})
        df = df.join(period_sales, on=["last_day", self.iid])

        period_sales = period_sales.reset_index().set_index(self.iid)
        last_day = last_ts.strftime("%Y-%m-%d")
        df = df.join(
            period_sales.loc[period_sales["last_day"] == last_day, ["period_sale"]],
            on=self.iid,
            rsuffix="_targ",
        )
        df["period_sale_targ"].fillna(0, inplace=True)
        df["quotient"] = df["period_sale_targ"] / df["period_sale"]
        del df["period_sale_targ"], df["period_sale"]

        df = df.groupby(self.iid)["quotient"].sum()
        df = df.sort_values(ascending=False)

        return df.index.tolist()[: self.n]


class FilterMatch(MatchRule):
    def __init__(self, trans_df: pd.DataFrame, item_id: str = "article_id"):
        self.iid = item_id
        mask = trans_df["t_dat"] >= "2020-08-01"
        self.trans_df = trans_df.loc[mask]

    def _off_stock_items(self) -> List[int]:
        """Get items that are no longer for sale

        Returns:
            List: list of off stock items
        """
        item_sale = self.trans_df[["customer_id", self.iid, "t_dat"]].copy()
        item_sale["t_dat"] = pd.to_datetime(item_sale["t_dat"])
        item_sale["year_month"] = (
            (item_sale["t_dat"].dt.year).astype(str)
            + "_"
            + (item_sale["t_dat"].dt.month).astype(str)
        )
        item_sale = (
            item_sale.groupby([self.iid, "year_month"])["customer_id"]
            .agg("count")
            .reset_index()
        )
        item_sale.rename(columns={"customer_id": "count"}, inplace=True)

        item_sale = pd.pivot_table(
            item_sale, values="count", index=self.iid, columns="year_month"
        )
        item_sale = item_sale.fillna(0)
        mask = (
            (item_sale["2020_9"] - item_sale["2020_8"]) / item_sale["2020_8"]
        ) < -0.8
        mask2 = item_sale["2020_9"] == 0

        return list(item_sale[mask | mask2].index)

    def match(self) -> np.ndarray:
        """Generate list of items that need to be filtered in the final prediction

        Returns:
            np.ndarray: item list
        """
        off_stock = self._off_stock_items()

        return off_stock


class MatchCollector:
    @staticmethod
    def collect(
        customer_list: np.ndarray,
        rules: List[MatchRule],
        filter: MatchRule = None,
        squeeze=True,
    ) -> pd.DataFrame:
        """Use rules to match items for each user

        Args:
            customer_list (np.ndarray): target customer list
            rules (List[MatchRule]): rules to match items
            filter (MatchRule, optional): filter to remove some matched items. Defaults to None.

        Returns:
            pd.DataFrame: prediction
        """
        output_df = pd.DataFrame(columns=["customer_id"])
        output_df["customer_id"] = customer_list

        rm_items = []
        if filter is not None:
            rm_items = filter.match()

        pred_df = None
        for i, rule in tqdm(enumerate(rules), "Collecting by rules"):
            feature_out = rule.match()
            if not isinstance(feature_out, pd.DataFrame):
                feature_out = [x for x in feature_out if x not in rm_items]
                tmp_np = np.zeros((len(customer_list) * len(feature_out), 2), dtype=int)
                for i, user in enumerate(customer_list):
                    tmp_np[i * len(feature_out) : (i + 1) * len(feature_out), 0] = user
                    tmp_np[
                        i * len(feature_out) : (i + 1) * len(feature_out), 1
                    ] = feature_out

                tmp_df = pd.DataFrame(tmp_np, columns=["customer_id", "prediction"])
                pred_df = pd.concat([pred_df, tmp_df])
            else:
                # if not type(rule).__name__ == 'OrderHistoryMatch':
                mask = feature_out["prediction"].isin(rm_items)
                feature_out = feature_out[~mask]
                pred_df = pd.concat([pred_df, feature_out])

        # pred_df = pred_df.reset_index(drop=True)
        pred_df = pred_df.drop_duplicates()

        if squeeze:
            pred_df = (
                pred_df.groupby("customer_id")["prediction"].apply(list).reset_index()
            )
            output_df = pd.merge(output_df, pred_df, on=["customer_id"], how="left")

            # * Impute
            mask = output_df["prediction"].isna()
            output_df.loc[mask, "prediction"] = pd.Series([[]] * mask.sum()).values

            return output_df[["customer_id", "prediction"]]
        else:
            return pred_df


