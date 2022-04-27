from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
from tqdm import tqdm


def calc_valid_date(week_num: int, last_date: str = "2020-09-29") -> Tuple[str]:
    """Calculate start and end date of a given week number.

    Parameters
    ----------
    week_num : int
        Week number.
    last_date : str, optional
        The last day, by default ``"2020-09-22"``.

    Returns
    -------
    Tuple[str]
        Start and end date of the given week number.
    """
    end_date = pd.to_datetime(last_date) - pd.Timedelta(days=7 * week_num - 1)
    start_date = end_date - pd.Timedelta(days=7)

    end_date = end_date.strftime("%Y-%m-%d")
    start_date = start_date.strftime("%Y-%m-%d")
    return start_date, end_date


def re_encode_ids(
    data: Dict, user_features: List[str], item_features: List[str]
) -> Tuple[Dict]:
    """Rencode ids in the dataset to reduce embedding size.

    Parameters
    ----------
    data : Dict
        Dataset.
    user_features : List[str]
        List of user features.
    item_features : List[str]
        List of item features.

    Returns
    -------
    Tuple[Dict]
        Dataset with re-encoded ids, encode maps.
    """
    feat2idx_dict = {}
    user = data["user"]
    item = data["item"]
    inter = data["inter"]

    for feat in user_features:
        if feat in inter.columns:
            valid_ids = inter[feat].unique()
            user = user.loc[user[feat].isin(valid_ids)]
        else:
            valid_ids = user[feat].unique()

        id2idx_map = {x: i + 1 for i, x in enumerate(list(valid_ids))}
        user[feat] = user[feat].map(id2idx_map)

        if feat in inter.columns:
            inter[feat] = inter[feat].map(id2idx_map)
        feat2idx_dict[feat] = id2idx_map

    for feat in item_features:
        if feat in inter.columns:
            valid_ids = inter[feat].unique()
            item = item.loc[item[feat].isin(valid_ids)]
        else:
            valid_ids = item[feat].unique()

        id2idx_map = {x: i + 1 for i, x in enumerate(list(valid_ids))}
        item[feat] = item[feat].map(id2idx_map)

        if feat in inter.columns:
            inter[feat] = inter[feat].map(id2idx_map)
        feat2idx_dict[feat] = id2idx_map

    user = user.reset_index(drop=True)
    item = item.reset_index(drop=True)

    data["user"] = user
    data["item"] = item
    data["inter"] = inter

    return data, feat2idx_dict


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Rudce memory usage by changing feature dtype.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to reduce memory usage.
    verbose : bool, optional
        Whether to print the process, by defaults ``False``.

    Returns
    -------
    pd.DataFrame
        Reduced memory usage dataframe.

    References
    ----------
    .. [1] https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65

    """
    start_mem_usg = df.memory_usage().sum() / 1024**2
    if verbose:
        print("Memory usage of dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings

            # Print current column type
            if verbose:
                print("******************************")
                print("Column: ", col)
                print("dtype before: ", df[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all():
                NAlist.append(col)
                df[col].fillna(mn - 1, inplace=True)

            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = df[col] - asint
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)

            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)

            # Print new column type
            if verbose:
                print("dtype after: ", df[col].dtype)
                print("******************************")

    # Print final result
    mem_usg = df.memory_usage().sum() / 1024**2
    if verbose:
        print("___MEMORY USAGE AFTER COMPLETION:___")
        print("Memory usage is: ", mem_usg, " MB")
        print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return df, NAlist


def merge_week_data(
    data: Dict, week_num: int, candidates: pd.DataFrame, label: pd.DataFrame
) -> pd.DataFrame:
    """Merge transaction, user and item features with week data.

    Parameters
    ----------
    data : Dict
        Dataset.
    week_num : int
        Week number.
    candidates : pd.DataFrame
        Retrieval candidates.
    label : pd.DataFrame
        Valid set.

    Returns
    -------
    pd.DataFrame
        Merged data.
    """
    tqdm.pandas()

    trans = data["inter"]
    item = data["item"]
    user = data["user"]

    trans = trans.sort_values(by=["t_dat", "customer_id"]).reset_index(drop=True)
    trans_info = (
        trans[trans["week"] > week_num]
        .groupby(["article_id"], as_index=False)
        .last()
        .drop(columns=["customer_id"])
    )
    trans_info["week"] = week_num

    # * ======================================================================================================================

    if label is not None:  # * label is None means this is the test data
        label.columns = ["customer_id", "label_article"]
        candidates = candidates.merge(label, on=["customer_id"], how="left")

        candidates = candidates[candidates["label_article"].notnull()]
        candidates["label"] = candidates.progress_apply(
            lambda x: 1 if x["article_id"] in x["label_article"] else 0, axis=1
        )

        # candidates['label'] = 0
        # mask = candidates['label_article'].notnull()
        # candidates.loc[mask, 'label'] = candidates[mask].progress_apply(lambda x: 1 if x['article_id'] in x['label_article'] else 0, axis=1)

        del candidates["label_article"]

    # * ======================================================================================================================

    # Merge with features
    candidates = candidates.merge(trans_info, on="article_id", how="left")

    user_feats = ["FN", "Active", "club_member_status", "fashion_news_frequency", "age"]
    candidates = candidates.merge(
        user[["customer_id", *user_feats]], on="customer_id", how="left"
    )
    candidates[user_feats] = candidates[user_feats].astype("int8")

    item_feats = [
        "product_type_no",
        "product_group_name",
        "graphical_appearance_no",
        "colour_group_code",
        "perceived_colour_value_id",
        "perceived_colour_master_id",
    ]
    candidates = candidates.merge(
        item[["article_id", *item_feats]], on="article_id", how="left"
    )
    candidates[item_feats] = candidates[item_feats].astype("int8")

    candidates, _ = reduce_mem_usage(candidates)

    return candidates
