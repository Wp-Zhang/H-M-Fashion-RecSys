import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict
from tqdm import tqdm

# * scores of rules are the bigger the better


class PersonalRetrieveRule(ABC):
    """Use certain rules to respectively retrieve items for each customer."""

    @abstractmethod
    def retrieve(self) -> pd.DataFrame:
        """Retrieve items

        Returns:
            pd.DataFrame: (customer_id, article_id, method, score)
        """


class UserGroupRetrieveRule(ABC):
    """Use certain rules to respectively retrieve items for each group of customers."""

    def merge(self, result: pd.DataFrame):
        result = result[[*self.cat_cols, self.iid, "method", "score"]]

        user = self.data["user"][[*self.cat_cols, "customer_id"]]
        tmp_df = pd.DataFrame({"customer_id": self.customer_list})
        tmp_df = tmp_df.merge(user, on="customer_id", how="left")
        tmp_df = tmp_df.merge(result, on=[*self.cat_cols], how="left")

        tmp_df = tmp_df[["customer_id", *self.cat_cols, self.iid, "score", "method"]]

        return tmp_df

    @abstractmethod
    def retrieve(self) -> pd.DataFrame:
        """Retrieve items

        Returns:
            pd.DataFrame: (*group_cols, article_id, method, score)
        """


class ItemGroupRetrieveRule(ABC):
    """Use certain rules to respectively retrieve items in each group of items."""

    @abstractmethod
    def retrieve(self) -> pd.DataFrame:
        """Retrieve items

        Returns:
            pd.DataFrame: (*group_cols, article_id, method, score)
        """


class GlobalRetrieveRule(ABC):
    """Use certain rules to retrieve items for all customers."""

    def merge(self, result: pd.DataFrame):
        result = result[[self.iid, "method", "score"]]

        num_item = result.shape[0]
        num_user = self.customer_list.shape[0]

        tmp_user = np.repeat(self.customer_list, num_item)
        tmp_df = result.iloc[np.tile(np.arange(num_item), num_user)]
        tmp_df = tmp_df.reset_index(drop=True)
        tmp_df["customer_id"] = tmp_user

        return tmp_df

    @abstractmethod
    def retrieve(self) -> pd.DataFrame:
        """Retrieve items

        Returns:
            pd.DataFrame: (article_id, method, score)
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
        name: str = "1",
        item_id: str = "article_id",
    ):
        """Initialize OrderHistory.

        Parameters
        ----------
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        days : int, optional
            Length of time window when getting user buying history, by default ``7``.
        n : int, optional
            Get top `n` recently bought items, by default ``None``.
        name : str, optional
            Name of the rule, by default ``1``.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        """
        self.iid = item_id
        self.trans_df = trans_df[["t_dat", "customer_id", item_id]]
        self.days = days
        self.n = n
        self.name = name

    def retrieve(self) -> pd.DataFrame:
        df = self.trans_df.reset_index()
        df["t_dat"] = pd.to_datetime(df["t_dat"])

        tmp = df.groupby("customer_id").t_dat.max().reset_index()
        tmp.columns = ["customer_id", "max_dat"]
        res = df.merge(tmp, on=["customer_id"], how="left")

        res["diff_dat"] = (res.max_dat - res.t_dat).dt.days
        res = res.loc[res["diff_dat"] < self.days].reset_index(drop=True)

        res = res.sort_values(by=["diff_dat"], ascending=True).reset_index(drop=True)
        res = res.groupby(["customer_id", self.iid], as_index=False).first()

        res = res.reset_index()
        res = res.sort_values(by="index", ascending=False).reset_index(drop=True)
        res["rank"] = res.groupby(["customer_id"])["index"].rank(
            ascending=True, method="first"
        )
        res["score"] = -res["diff_dat"]

        if self.n is not None:
            res = res.loc[res["rank"] <= self.n]

        res["method"] = f"OrderHistory_{self.name}"
        res = res[["customer_id", self.iid, "score", "method"]]

        return res


class OrderHistoryDecay(PersonalRetrieveRule):
    """Retrieve recently bought items by the customer with decay."""

    def __init__(
        self,
        trans_df: pd.DataFrame,
        days: int = 7,
        n: int = None,
        name: str = "1",
        item_id: str = "article_id",
    ):
        """Initialize OrderHistoryDecay.

        Parameters
        ----------
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        days : int, optional
            Length of time window when getting user buying history, by default ``7``.
        n : int, optional
            Get top `n` recently bought items, by default ``None``.
        name : str, optional
            Name of the rule, by default ``1``.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        """
        self.iid = item_id
        self.trans_df = trans_df[["customer_id", self.iid, "t_dat"]]
        self.name = name
        self.days = days
        self.n = n

    def retrieve(self):
        df = self.trans_df
        df["t_dat"] = pd.to_datetime(df["t_dat"])
        last_ts = df["t_dat"].max()
        df["dat_gap"] = (last_ts - df["t_dat"]).dt.days

        df["last_day"] = last_ts - (last_ts - df["t_dat"]).dt.floor(f"{self.days}D")
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

        a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
        df["value"] = a / np.sqrt(x) + b * np.exp(-c * x) - d
        df["value"][df["value"] < 0] = 0
        df["value"] = df["value"] * df["quotient"]

        df = df.groupby(["customer_id", self.iid], as_index=False)["value"].sum()
        df = df.sort_values(by="value", ascending=False).reset_index(drop=True)
        df = df.reset_index()

        df["rank"] = df.groupby(["customer_id"])["index"].rank(
            ascending=True, method="first"
        )
        df["score"] = df["value"]

        df = df[df["value"] > 150]

        if self.n is not None:
            df = df[df["rank"] <= self.n]
        df["method"] = f"OrderHistoryDecay_{self.name}"

        df = df[["customer_id", self.iid, "score", "method"]]

        return df


class ItemPair(PersonalRetrieveRule):
    """Customers who bought this often bought this."""

    def __init__(
        self, trans_df: pd.DataFrame, name: str = "1", item_id: str = "article_id"
    ):
        """Initialize ItemPair.

        Parameters
        ----------
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        name: str, optional
            Name of the rule, by default ``1``.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        """
        self.iid = item_id
        self.trans_df = trans_df[["customer_id", self.iid]].drop_duplicates()
        self.name = name

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
        pair["score"] = pair["count"]

        return pair

    def retrieve(self) -> pd.DataFrame:
        pair = self._get_freq_pair()

        df = self.trans_df
        df = df.merge(pair, on=self.iid, how="left")

        df = df.loc[df["pair"].notnull()]
        df = df.drop_duplicates(["customer_id", "pair"])

        df[self.iid] = df["pair"].astype("int32")
        df["method"] = "ItemPairRetrieve_" + self.name

        df = df[["customer_id", self.iid, "method", "score"]]
        return df


# * ====================== User Group Retrieve Rules ====================== *


class UserGroupTimeHistory(UserGroupRetrieveRule):
    """Retrieve popular items of each **user** group in specified time window."""

    def __init__(
        self,
        data: Dict,
        customer_list: List,
        trans_df: pd.DataFrame,
        cat_cols: List,
        n: int = 12,
        name: str = "1",
        unique: bool = True,
        item_id: str = "article_id",
    ):
        """Initialize TimeHistory.

        Parameters
        ----------
        data : Dict
            Data dictionary.
        customer_list : List
            List of target customer ids.
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        cat_cols: List
            Name of user group columns.
        n : int, optional
            Get top `n` popular items, by default ``12``.
        name : str, optional
            Name of the rule, by default ``1``.
        unique : bool, optional
            Whether to drop duplicate buying records, by default ``True``.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        """
        self.data = data
        self.customer_list = customer_list
        self.iid = item_id
        self.trans_df = trans_df[["customer_id", self.iid, *cat_cols]]
        self.cat_cols = cat_cols
        self.unique = unique
        self.n = n
        self.name = name

    def retrieve(self) -> List[int]:
        """Get popular items in the specified time window

        Returns:
            List[int]: top n popular items
        """
        df = self.trans_df
        if self.unique:
            df = df.drop_duplicates(["customer_id", self.iid])

        df["count"] = 1
        df = df.groupby([*self.cat_cols, self.iid], as_index=False)["count"].sum()
        df = df.sort_values(by="count", ascending=False).reset_index(drop=True)
        df = df.reset_index()

        df["rank"] = df.groupby([*self.cat_cols])["index"].rank(
            ascending=True, method="first"
        )

        df["score"] = df["count"]
        df["method"] = "UGTimeHistory_" + self.name
        df = df[df["rank"] <= self.n][[*self.cat_cols, self.iid, "score", "method"]]

        df = self.merge(df)

        return df[["customer_id", self.iid, "method", "score"]]


class UserGroupSaleTrend(UserGroupRetrieveRule):
    """Retrieve trending items in specified time window for each user group."""

    def __init__(
        self,
        data: Dict,
        customer_list: List,
        trans_df: pd.DataFrame,
        cat_cols: List,
        days: int = 7,
        n: int = 12,
        name: str = "1",
        t: float = 0.8,
        item_id: str = "article_id",
    ):
        """Initialize SaleTrend.

        Parameters
        ----------
        data : Dict
            Data dictionary.
        customer_list : List
            List of target customer ids.
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        cat_cols : List
            Name of user group columns.
        days : int, optional
            Length of time window when calculating sale trend, by default ``7``.
        n : int, optional
            Get top `n` recently bought items, by default ``12``.
        name : str, optional
            Name of the rule, by default ``1``.
        t : float, optional
            Sale trend ratio, by default ``0.8``.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        """
        self.iid = item_id
        self.data = data
        self.customer_list = customer_list
        self.trans_df = trans_df[["customer_id", self.iid, "t_dat", *cat_cols]]
        self.cat_cols = cat_cols
        self.days = days
        self.n = n
        self.t = t
        self.name = name

    def retrieve(self):
        item_sale = self.trans_df
        item_sale["t_dat"] = pd.to_datetime(item_sale["t_dat"])
        item_sale["dat_gap"] = (item_sale.t_dat.max() - item_sale.t_dat).dt.days

        item_sale = item_sale[item_sale["dat_gap"] <= 2 * self.days - 1]
        group_a = item_sale[item_sale["dat_gap"] > self.days - 1]
        group_b = item_sale[item_sale["dat_gap"] <= self.days - 1]

        group_a["count"] = 1
        group_b["count"] = 1
        group_a = (
            group_a.groupby([*self.cat_cols, self.iid])["count"].sum().reset_index()
        )
        group_b = (
            group_b.groupby([*self.cat_cols, self.iid])["count"].sum().reset_index()
        )

        log = pd.merge(group_b, group_a, on=[*self.cat_cols, self.iid], how="left")
        log["count_x"] = log["count_x"].fillna(0)
        log["trend"] = (log["count_x"] - log["count_y"]) / (log["count_y"] + 1)

        log = log[log["trend"] > self.t]
        log = log.sort_values(by=["count_x", "trend"], ascending=False)
        log = log.reset_index(drop=True).reset_index()
        log["rank"] = log.groupby([*self.cat_cols])["index"].rank(
            ascending=True, method="first"
        )
        log = log[log["rank"] <= self.n]

        log["method"] = f"UGSaleTrend_{self.name}"
        log["score"] = log["trend"]
        log = log[[*self.cat_cols, self.iid, "method", "score"]]

        log = self.merge(log)

        return log[["customer_id", self.iid, "method", "score"]]


# * ======================== Global Retrieve Rules ======================== *


class TimeHistory(GlobalRetrieveRule):
    """Retrieve popular items in specified time window."""

    def __init__(
        self,
        customer_list: List,
        trans_df: pd.DataFrame,
        n: int = 12,
        name: str = "1",
        unique: bool = True,
        item_id: str = "article_id",
    ):
        """Initialize TimeHistory.

        Parameters
        ----------
        customer_list : List
            List of target customer ids.
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
        self.customer_list = customer_list
        self.trans_df = trans_df[["customer_id", self.iid]]
        self.unique = unique
        self.n = n
        self.name = name

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
        df["score"] = df["count"]
        df["method"] = "TimeHistory_" + self.name

        df = df[df["rank"] <= self.n][[self.iid, "score", "method"]]
        df = self.merge(df)

        return df[["customer_id", self.iid, "method", "score"]]


class TimeHistoryDecay(GlobalRetrieveRule):
    """Retrieve popular items in specified time window with decay."""

    def __init__(
        self,
        customer_list: List,
        trans_df: pd.DataFrame,
        days: int = 7,
        n: int = 12,
        name: str = "1",
        item_id: str = "article_id",
    ):
        """Initialize TimeHistoryDecay.

        Parameters
        ----------
        customer_list : List
            List of target customer ids.
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        days : int, optional
            Length of time window, by default ``7``.
        n : int, optional
            Get top `n` recently bought items, by default ``12``.
        name : str, optional
            Name of the rule, by default ``1``.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        """
        self.iid = item_id
        self.customer_list = customer_list
        self.trans_df = trans_df[["customer_id", self.iid, "t_dat"]]
        self.days = days
        self.n = n
        self.name = name

    def retrieve(self):
        df = self.trans_df
        df["t_dat"] = pd.to_datetime(df["t_dat"])
        last_ts = df["t_dat"].max()
        df["dat_gap"] = (last_ts - df["t_dat"]).dt.days

        df["last_day"] = last_ts - (last_ts - df["t_dat"]).dt.floor(f"{self.days}D")
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

        a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
        df["value"] = a / np.sqrt(x) + b * np.exp(-c * x) - d
        df["value"][df["value"] < 0] = 0
        df["value"] = df["value"] * df["quotient"]

        df = df.groupby([self.iid], as_index=False)["value"].sum()
        df = df.sort_values(by="value", ascending=False).reset_index(drop=True)

        df["rank"] = df.index + 1
        df["score"] = df["value"]
        df = df[df["rank"] <= self.n]
        df["method"] = f"TimeHistoryDecay_{self.name}"

        df = self.merge(df)

        return df[["customer_id", self.iid, "score", "method"]]


class SaleTrend(GlobalRetrieveRule):
    """Retrieve trending items in specified time window."""

    def __init__(
        self,
        customer_list: List,
        trans_df: pd.DataFrame,
        days: int = 7,
        n: int = 12,
        name: str = "1",
        t: float = 0.8,
        item_id: str = "article_id",
    ):
        """Initialize SaleTrend.

        Parameters
        ----------
        customer_list : List
            List of target customer ids.
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        days : int, optional
            Length of time window when calculating sale trend, by default ``7``.
        n : int, optional
            Get top `n` recently bought items, by default ``12``.
        name : str, optional
            Name of the rule, by default ``1``.
        t : float, optional
            Sale trend ratio, by default ``0.8``.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        """
        self.iid = item_id
        self.customer_list = customer_list
        self.trans_df = trans_df[["customer_id", self.iid, "t_dat"]]
        self.days = days
        self.n = n
        self.t = t
        self.name = name

    def retrieve(self):
        item_sale = self.trans_df
        item_sale["t_dat"] = pd.to_datetime(item_sale["t_dat"])
        item_sale["dat_gap"] = (item_sale.t_dat.max() - item_sale.t_dat).dt.days

        item_sale = item_sale[item_sale["dat_gap"] <= 2 * self.days - 1]
        group_a = item_sale[item_sale["dat_gap"] > self.days - 1]
        group_b = item_sale[item_sale["dat_gap"] <= self.days - 1]

        group_a["count"] = 1
        group_b["count"] = 1
        group_a = group_a.groupby(self.iid)["count"].sum().reset_index()
        group_b = group_b.groupby(self.iid)["count"].sum().reset_index()

        log = pd.merge(group_b, group_a, on=[self.iid], how="left")
        log["count_x"] = log["count_x"].fillna(0)
        log["trend"] = (log["count_x"] - log["count_y"]) / log["count_y"]

        log = log[log["trend"] > self.t]
        log = log.sort_values(by=["count_x", "trend"], ascending=False)
        log = log.reset_index(drop=True).iloc[: self.n]

        log["method"] = f"SaleTrend_{self.name}"
        log["score"] = log["trend"]
        log = log[[self.iid, "method", "score"]]

        log = self.merge(log)

        return log[["customer_id", self.iid, "score", "method"]]


# * ============================ Filter Rules ============================ *


class OutOfStock(FilterRule):
    """Filter items that are out of stock."""

    def __init__(self, trans_df: pd.DataFrame, item_id: str = "article_id"):
        """Initialize OutOfStock.

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
