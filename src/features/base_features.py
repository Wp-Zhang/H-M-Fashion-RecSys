from typing import List
import pandas as pd
import numpy as np


def cum_sale(trans: pd.DataFrame, groupby_cols: List, unique=False) -> np.ndarray:
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
    tmp_inter = trans[["t_dat", "customer_id", *groupby_cols]]
    if unique:
        tmp_inter = tmp_inter.drop_duplicates(["customer_id", *groupby_cols])

    df = tmp_inter.groupby(["t_dat", *groupby_cols]).size().reset_index(name="_SALE")
    df["_SALE"] = df.groupby(groupby_cols)["_SALE"].cumsum()
    df["_SALE"] = df.groupby(groupby_cols)["_SALE"].shift(1)
    df["_SALE"] = df["_SALE"].fillna(0)

    tmp_inter = trans[["t_dat", "customer_id", *groupby_cols]].merge(
        df, on=["t_dat", *groupby_cols], how="left"
    )
    tmp_inter["_SALE"] = tmp_inter["_SALE"].fillna(0).astype("int")

    return tmp_inter["_SALE"].values


def week_sale(trans: pd.DataFrame, groupby_cols: List, unique=False) -> np.ndarray:
    """Calculate week sales of each item unit.

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
        Array of week sales.
    """
    tmp_inter = trans[["week", "customer_id", *groupby_cols]]
    if unique:
        tmp_inter = tmp_inter.drop_duplicates(["customer_id", *groupby_cols])

    df = tmp_inter.groupby(["week", *groupby_cols]).size().reset_index(name="_SALE")
    # df["_SALE"] = df.groupby(groupby_cols)["_SALE"].cumsum()
    # df["_SALE"] = df.groupby(groupby_cols)["_SALE"].shift(1)
    # df["_SALE"] = df["_SALE"].fillna(0)

    tmp_inter = trans[["week", "customer_id", *groupby_cols]].merge(
        df, on=["week", *groupby_cols], how="left"
    )
    tmp_inter["_SALE"] = tmp_inter["_SALE"].fillna(0).astype("int")

    return tmp_inter["_SALE"].values


def sale_trend(
    trans: pd.DataFrame,
    groupby_cols: List,
    days: int = 7,
    item_id: str = "article_id",
    week_num: int = 6,
) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    trans : pd.DataFrame
        _description_
    groupby_cols : List
        _description_
    days : int, optional
        _description_, by default 7
    item_id : str, optional
        _description_, by default "article_id"

    Returns
    -------
    np.ndarray
        _description_
    """
    df = trans[[*groupby_cols, item_id, "t_dat", "week"]]
    df["t_dat"] = pd.to_datetime(df["t_dat"])

    tmp_l = []
    name = "SaleTrend_" + "|".join(groupby_cols) + "_" + item_id
    for week in range(week_num):
        tmp_df = df[df["week"] >= week]
        tmp_df["dat_gap"] = (tmp_df.t_dat.max() - tmp_df.t_dat).dt.days

        tmp_df = tmp_df[tmp_df["dat_gap"] <= 2 * days - 1]
        group_a = tmp_df[tmp_df["dat_gap"] > days - 1]
        group_b = tmp_df[tmp_df["dat_gap"] <= days - 1]

        group_a["count"] = 1
        group_b["count"] = 1
        group_a = group_a.groupby([*groupby_cols, item_id])["count"].sum().reset_index()
        group_b = group_b.groupby([*groupby_cols, item_id])["count"].sum().reset_index()

        log = pd.merge(group_b, group_a, on=[*groupby_cols, item_id], how="left")
        log["count_x"] = log["count_x"].fillna(0)
        log["count_y"] = log["count_y"].fillna(0)
        log[name] = (log["count_x"] - log["count_y"]) / (log["count_y"] + 1)

        log = log[[*groupby_cols, item_id, name]]
        res = df[df["week"] >= week].merge(log, on=[*groupby_cols, item_id], how="left")

        tmp_l.append(res)

    info = pd.concat(tmp_l)[[*groupby_cols, item_id, name, "week"]]
    info = info.drop_duplicates(groupby_cols + [item_id, "week"])

    df = df.merge(info, on=[*groupby_cols, item_id, "week"], how="left")

    return df[name].values


def repurchase_ratio(trans: pd.DataFrame, groupby_cols: List) -> np.ndarray:
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

    # * Article re-purchase ratio
    item_user_sale = (
        trans.groupby(["customer_id", *groupby_cols]).size().reset_index(name="_SALE")
    )
    item_sale = item_user_sale.groupby(groupby_cols).size().reset_index(name="_I_SALE")
    item_user_sale = (
        item_user_sale[item_user_sale["_SALE"] > 1]  # * purchase more than once
        .groupby(groupby_cols)
        .size()
        .reset_index(name="_MULTI_SALE")
    )
    item_sale = item_sale.merge(item_user_sale, on=groupby_cols, how="left")
    item_sale["_RATIO"] = item_sale["_MULTI_SALE"] / item_sale["_I_SALE"]
    item_sale = item_sale[[*groupby_cols, "_RATIO"]]

    tmp_trans = trans[groupby_cols]
    tmp_trans = tmp_trans.merge(item_sale, on=groupby_cols, how="left")

    return tmp_trans["_RATIO"].values


def purchased_before(trans: pd.DataFrame, groupby_cols: List) -> np.ndarray:
    df = (
        trans.groupby(["customer_id", *groupby_cols, "t_dat"])
        .size()
        .reset_index(name="_BOUGHT")
    )
    df["_BOUGHT"] = df.groupby(["customer_id", *groupby_cols])["_BOUGHT"].cumsum()
    df["_BOUGHT"] = df.groupby(["customer_id", *groupby_cols])["_BOUGHT"].shift(1)
    df["_BOUGHT"] = df["_BOUGHT"].fillna(0)

    tmp = trans[["customer_id", *groupby_cols, "t_dat"]]
    tmp = tmp.merge(df, on=["customer_id", *groupby_cols, "t_dat"], how="left")
    tmp["_BOUGHT"] = tmp["_BOUGHT"].fillna(0)
    tmp["_BOUGHT"][tmp["_BOUGHT"] > 1] = 1

    return tmp["_BOUGHT"].values
