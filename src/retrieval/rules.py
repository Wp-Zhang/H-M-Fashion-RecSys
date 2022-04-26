import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List
from tqdm import tqdm


class PersonalRetrieveRule(ABC):
    """Use certain rules to respectively retrieve items for each customer."""

    @abstractmethod
    def retrieve(self) -> pd.DataFrame:
        """Retrieve items

        Returns:
            pd.DataFrame: (customer_id, article_id, method, rank)
        """


class GlobalRetrieveRule(ABC):
    """Use certain rules to retrieve items for all customers."""

    @abstractmethod
    def retrieve(self) -> pd.DataFrame:
        """Retrieve items

        Returns:
            pd.DataFrame: (article_id, method, rank)
        """


class FilterRule(ABC):
    """Use certain rules to remove some retrieved items."""

    @abstractmethod
    def retrieve(self) -> List:
        """Retrieve items

        Returns:
            List: items to be removed
        """


# * ======================= Personal Retrieve Rules ======================= *


class OrderHistory(PersonalRetrieveRule):
    """Retrieve recently bought items by the customer."""

    def __init__(
        self,
        trans_df: pd.DataFrame,
        days: int = 7,
        n: int = None,
        item_id: str = "article_id",
    ):
        """Initialize an `OrderHistory` instance.

        Parameters
        ----------
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        days : int, optional
            Length of time window when getting user buying history, by default ``7``.
        n : int, optional
            Get top `n` recently bought items, by default ``12``.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        """
        self.iid = item_id
        self.trans_df = trans_df[["t_dat", "customer_id", item_id]]
        self.days = days
        self.n = n

    def retrieve(self) -> pd.DataFrame:
        df = self.trans_df.reset_index()
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
            res = res.loc[res["rank"] <= self.n]
            res["method"] = f"OrderHistory_{self.days}_top{self.n}"
        else:
            res["method"] = f"OrderHistory_{self.days}"

        res = res[["customer_id", self.iid, "rank", "method"]]

        return res


class ItemPair(PersonalRetrieveRule):
    """Customers who bought this often bought this."""

    def __init__(self, trans_df: pd.DataFrame, item_id: str = "article_id"):
        """Initialize an `ItemPairRetrieve` instance.

        Parameters
        ----------
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        """
        self.iid = item_id
        self.trans_df = trans_df[["customer_id", self.iid]].drop_duplicates()

    def _get_freq_pair(self) -> pd.DataFrame:
        """Generate dict of frequent item pairs in target transaction dataframe.

        Returns
        -------
        pd.DataFrame
            Frequent item pairs.
        """
        df = self.trans_df
        df2 = df.rename(columns={self.iid: "pair"})

        pair = df.merge(df2, on="customer_id")
        pair = pair[pair[self.iid] != pair["pair"]]
        pair["count"] = 1
        pair = pair.groupby([self.iid, "pair"])["count"].sum().reset_index()
        pair = pair.sort_values("count", ascending=False).reset_index(drop=True)
        pair = pair.groupby(self.iid).first().reset_index()
        pair = pair.sort_values(by="count", ascending=False).reset_index(drop=True)
        pair["rank"] = pair.index + 1

        return pair

    def retrieve(self) -> pd.DataFrame:
        pair = self._get_freq_pair()

        df = self.trans_df
        df = df.merge(pair, on=self.iid, how="left")
        df = df.loc[df["pair"].notnull()]

        df = df.drop_duplicates(["customer_id", "pair"])
        df[self.iid] = df["pair"].astype("int32")
        df["method"] = "ItemPairRetrieve"

        df = df[["customer_id", self.iid, "method", "rank"]]
        return df


# * ======================== Global Retrieve Rules ======================== *


class TimeHistory(GlobalRetrieveRule):
    """Retrieve popular items in specified time window."""

    def __init__(
        self,
        trans_df: pd.DataFrame,
        n: int = 12,
        unique: bool = True,
        item_id: str = "article_id",
    ):
        """Initialize TimeHistory.

        Parameters
        ----------
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        n : int, optional
            Get top `n` popular items, by default ``12``.
        unique : bool, optional
            Whether to drop duplicate buying records, by default ``True``.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        """
        self.iid = item_id
        self.trans_df = trans_df[["customer_id", self.iid]]
        self.unique = unique
        self.n = n

    def retrieve(self) -> List[int]:
        """Get popular items in the specified time window

        Returns:
            List[int]: top n popular items
        """
        df = self.trans_df
        if self.unique:
            df = df.drop_duplicates(["customer_id", self.iid])
        df["count"] = 1
        df = df.groupby(self.iid, as_index=False)["count"].sum()
        df = df.sort_values(by="count", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        df["method"] = f"TimeHistory_{self.n}"

        df = df[df["rank"] <= self.n][[self.iid, "rank", "method"]]

        return df


# * ============================ Filter Rules ============================ *


class OutOfStock(FilterRule):
    """Filter items that are out of stock."""

    def __init__(self, trans_df: pd.DataFrame, item_id: str = "article_id"):
        """Initialize an `OutOfStock` instance.

        Parameters
        ----------
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        item_id : str, optional
           Name of item id, by default ``"article_id"``.
        """
        self.iid = item_id
        mask = trans_df["t_dat"] >= "2020-08-01"
        self.trans_df = trans_df.loc[mask, ["customer_id", self.iid, "t_dat"]]

    def _off_stock_items(self) -> List[int]:
        """Get items that are no longer for sale

        Returns:
            List: list of off stock items
        """
        sale = self.trans_df
        sale["t_dat"] = pd.to_datetime(sale["t_dat"])
        sale["year_month"] = (
            (sale["t_dat"].dt.year).astype(str)
            + "_"
            + (sale["t_dat"].dt.month).astype(str)
        )
        sale = (
            sale.groupby([self.iid, "year_month"])["customer_id"]
            .count()
            .reset_index(name="count")
        )

        sale = pd.pivot_table(
            sale, values="count", index=self.iid, columns="year_month"
        )
        sale = sale.fillna(0)
        mask = ((sale["2020_9"] - sale["2020_8"]) / sale["2020_8"]) < -0.8
        mask2 = sale["2020_9"] == 0

        return list(sale[mask | mask2].index)

    def retrieve(self) -> List:
        off_stock = self._off_stock_items()

        return off_stock


# * ======================= TBD ======================= *


class SaleTrendRetrieve(GlobalRetrieveRule):
    def __init__(
        self, trans_df, days: int = 7, n: int = 12, item_id: str = "article_id"
    ):
        self.iid = item_id
        self.trans_df = trans_df[["customer_id", self.iid, "t_dat"]]
        self.days = days  # length of period
        self.n = n  # top n

    def retrieve(self):
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


class OrderHistoryDecayRetrieve(PersonalRetrieveRule):
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

    def retrieve(self):
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


class TimeHistoryRetrieve2(GlobalRetrieveRule):
    def __init__(self, trans_df, n=12, item_id: str = "article_id"):
        self.iid = item_id
        self.trans_df = trans_df[["customer_id", self.iid, "t_dat"]]
        self.n = n

    def retrieve(self):
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
