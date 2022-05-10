from typing import List
import pandas as pd
import numpy as np
from tqdm import tqdm
from ..utils import calc_valid_date


def full_sale(
    trans: pd.DataFrame,
    groupby_cols: List,
    unique=False,
    week_num: int = 6,
) -> np.ndarray:
    """Calculate cumulative sales of each item unit.

    Parameters
    ----------
    trans : pd.DataFrame
        Dataframe of transaction data.
    groupby_cols : List
        Item unit.
    unique : bool, optional
        Whether to drop duplicate customer-item pairs, by default ``False``.

    Returns
    -------
    np.ndarray
        Array of cumulative sales.
    """
    inter = trans[["customer_id", "week", "valid", *groupby_cols]]
    if unique:
        inter = inter.drop_duplicates(["customer_id", "week", *groupby_cols])

    tmp_l = []
    for week in range(1, week_num + 1):
        df = inter[inter["week"] >= week]
        df = df.groupby([*groupby_cols])["valid"].sum().reset_index(name="_SALE")
        df["week"] = week
        tmp_l.append(df)

    df = pd.concat(tmp_l, ignore_index=True)
    inter = trans[["customer_id", "week", *groupby_cols]].merge(
        df, on=["week", *groupby_cols], how="left"
    )
    inter["_SALE"] = inter["_SALE"].fillna(0).astype("int")

    return inter["_SALE"].values


def week_sale(
    trans: pd.DataFrame,
    groupby_cols: List,
    unique=False,
    step: int = 0,
    week_num: int = 6,
) -> np.ndarray:
    """Calculate week sales of each item unit.

    Parameters
    ----------
    trans : pd.DataFrame
        Dataframe of transaction data.
    groupby_cols : List
        Item unit.
    unique : bool, optional
        Whether to drop duplicate customer-item pairs, by default ``False``.
    step: int, optional
        Step of week, by default ``0``. 0 means current week sale, 1 means last week sale, etc.

    Returns
    -------
    np.ndarray
        Array of week sales.
    """

    tmp_inter = trans[["week", "customer_id", "valid", *groupby_cols]]
    tmp_inter = tmp_inter[tmp_inter["week"] <= week_num + step + 2]
    if unique:
        tmp_inter = tmp_inter.drop_duplicates(["customer_id", *groupby_cols])

    df = (
        tmp_inter.groupby(["week", *groupby_cols])["valid"]
        .sum()
        .reset_index(name="_SALE")
    )
    df["week"] -= step

    tmp_inter = trans[["week", "customer_id", *groupby_cols]].merge(
        df, on=["week", *groupby_cols], how="left"
    )
    tmp_inter["_SALE"] = tmp_inter["_SALE"].fillna(0).astype("int")

    return tmp_inter["_SALE"].values.astype(int)


def period_sale(
    trans: pd.DataFrame,
    groupby_cols: List,
    unique=False,
    days: int = 14,
    rank: bool = False,
    norm: bool = False,
    week_num: int = 6,
) -> np.array:
    """Calculate item unit sale in last n days.

    Parameters
    ----------
    trans : pd.DataFrame
        Dataframe of transaction data.
    groupby_cols : List
        Item unit.
    unique : bool, optional
        Whether to drop duplicate customer-item pairs, by default ``False``.
    days : int, optional
        Length of time window, by default ``14``.
    rank: bool, optional
        Whether to return rank of sale, by default ``False``.
    norm: bool, optional
        Whether to normalized count, by default ``False``.
    week_num : int, optional
        Number of weeks of data to calculate sale for, by default ``6``.

    Returns
    -------
    Tuple[np.array]
        Period sale (sale rank | normed sale).
    """
    df = trans[[*groupby_cols, "customer_id", "t_dat", "week", "valid"]]
    if unique:
        df = df.drop_duplicates(["customer_id", *groupby_cols])

    df["t_dat"] = pd.to_datetime(df["t_dat"])

    tmp_l = []
    name = "PERIOD_SALE"
    for week in range(1, week_num + 1):
        _, end_date = calc_valid_date(week)
        tmp_df = df[
            (pd.to_datetime(end_date) - pd.Timedelta(days=days + 1) <= df["t_dat"])
            & (df["t_dat"] < pd.to_datetime(end_date))
        ]
        sale = tmp_df.groupby(groupby_cols)["valid"].sum().reset_index(name=name)
        sale["week"] = week
        if rank:
            sale[name + "_rank"] = sale[name].rank(ascending=False)
        if norm:
            sale[name + "_norm"] = sale[name] / sale[name].sum()

        tmp_l.append(sale)

    sale_df = pd.concat(tmp_l, ignore_index=True)
    df = df.merge(sale_df, on=[*groupby_cols, "week"], how="left")

    if not rank and not norm:
        return df[name].values.astype(int)
    elif rank and not norm:
        return df[name].values.astype(int), df[name + "_rank"].values.astype(int)
    elif not rank and norm:
        return df[name].values.astype(int), df[name + "_norm"].values
    else:
        return (
            df[name].values.astype(int),
            df[name + "_rank"].values.astype(int),
            df[name + "_norm"].values,
        )


def repurchase_ratio(
    trans: pd.DataFrame, groupby_cols: List, week_num: int = 6
) -> np.ndarray:
    """Calculate repurchase ratio of item units.

    Parameters
    ----------
    trans : pd.DataFrame
        Dataframe of transaction data.
    groupby_cols : List
        Item unit.

    Returns
    -------
    np.ndarray
        Array of repurchase ratios.
    """
    tmp_l = []
    for week in tqdm(range(1, week_num + 1)):
        tmp_df = trans[trans["week"] >= week]
        # * Article re-purchase ratio
        item_user_sale = (
            tmp_df[tmp_df["valid"] == 1]
            .groupby(["customer_id", *groupby_cols])
            .size()
            .reset_index(name="_SALE")
        )
        item_sale = (
            item_user_sale.groupby(groupby_cols).size().reset_index(name="_I_SALE")
        )
        item_user_sale = (
            item_user_sale[item_user_sale["_SALE"] > 1]  # * purchase more than once
            .groupby(groupby_cols)
            .size()
            .reset_index(name="_MULTI_SALE")
        )
        item_sale = item_sale.merge(item_user_sale, on=groupby_cols, how="left")
        item_sale["_RATIO"] = item_sale["_MULTI_SALE"] / (item_sale["_I_SALE"] + 1e-6)
        item_sale = item_sale[[*groupby_cols, "_RATIO"]]
        item_sale["week"] = week
        tmp_l.append(item_sale)

    df = trans[["week", *groupby_cols]]
    item_sale = pd.concat(tmp_l, ignore_index=True)
    df = df.merge(item_sale, on=["week", *groupby_cols], how="left")
    df["_RATIO"] = df["_RATIO"].fillna(0)
    return df["_RATIO"].values


def popularity(
    trans: pd.DataFrame, item_id: str = "article_id", week_num: int = 6
) -> np.ndarray:
    """Calculate popularity factor of item unit.

    Parameters
    ----------
    trans : pd.DataFrame
        Dataframe of transaction data.
    item_id : str, optional
        item unit id, by default ``"article_id"``.
    week_num : int, optional
        Number of weeks of data to calculate sale for, by default ``6``.

    Returns
    -------
    np.ndarray
        Array of popularity factors.
    """
    df = trans[[item_id, "t_dat", "week", "valid"]]
    df["t_dat"] = pd.to_datetime(df["t_dat"])

    tmp_l = []
    name = "Popularity_" + item_id
    for week in range(1, week_num + 1):
        tmp_df = df[df["week"] >= week][df["valid"] == 1]
        last_day = tmp_df["t_dat"].max()
        tmp_df[name] = 1 / ((last_day - tmp_df["t_dat"]).dt.days + 1)
        tmp_df = tmp_df.groupby([item_id])[name].sum().reset_index()
        tmp_df["week"] = week
        tmp_l.append(tmp_df)

    info = pd.concat(tmp_l)[[item_id, name, "week"]]
    df = df.merge(info, on=[item_id, "week"], how="left")
    df[name] = df[name].fillna(0)
    return df[name].values
