from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
from tqdm._tqdm_notebook import tqdm_notebook as tqdm


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
            if pd.api.types.is_integer_dtype(df[col]):
                IsInt = True
            else:
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
    data: Dict, trans: pd.DataFrame, week_num: int, candidates: pd.DataFrame
) -> pd.DataFrame:
    """Merge transaction, user and item features with week data.

    Parameters
    ----------
    data : Dict
        Dataset.
    trans : pd.DataFrame
        Dataframe of transaction records.
    week_num : int
        Week number.
    candidates : pd.DataFrame
        Retrieval candidates.

    Returns
    -------
    pd.DataFrame
        Merged data ready for training.
    """
    tqdm.pandas()

    user = data["user"]
    item = data["item"]

    trans_info = (
        trans[trans["week"] == week_num + 1]
        .sort_values(by=["t_dat", "customer_id"])
        .reset_index(drop=True)
        .groupby(["week", "article_id"], as_index=False)
        .last()
        .drop(columns=["customer_id"])
    )
    trans_info["week"] = week_num
    trans_info, _ = reduce_mem_usage(trans_info)

    # * ======================================================================================================================

    if week_num != 0:  # this is not test data
        start_date, end_date = calc_valid_date(week_num)
        mask = (start_date <= data["inter"]["t_dat"]) & (
            data["inter"]["t_dat"] < end_date
        )
        label = data["inter"].loc[mask, ["customer_id", "article_id"]]
        label = label.drop_duplicates(["customer_id", "article_id"])
        label["label"] = 1

        label_customers = label["customer_id"].unique()
        print(len(label_customers))
        candidates = candidates[candidates["customer_id"].isin(label_customers)]

        candidates = candidates.merge(
            label, on=["customer_id", "article_id"], how="left"
        )
        candidates["label"] = candidates["label"].fillna(0)

    # * ======================================================================================================================

    # Merge with features
    candidates = candidates.merge(trans_info, on="article_id", how="left")

    user_feats = [
        "FN",
        "Active",
        "club_member_status",
        "fashion_news_frequency",
        "age",
        "user_gender",
    ]
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
        "article_gender",
        "season_type",
    ]
    candidates = candidates.merge(
        item[["article_id", *item_feats]], on="article_id", how="left"
    )
    candidates[item_feats] = candidates[item_feats].astype("int8")

    candidates, _ = reduce_mem_usage(candidates)

    return candidates


def calc_embd_similarity(
    candidate: pd.DataFrame,
    user_embd: np.ndarray,
    item_embd: np.ndarray,
    sub: bool = True,
    item_id: str = "article_id",
) -> np.ndarray:
    """Calculate user-item embedding similarity.

    Parameters
    ----------
    candidate : pd.DataFrame
        DataFrame of candidate items for one week.
    user_embd : np.ndarray
        Pre-trained user embedding.
    item_embd : np.ndarray
        Pre-trained item embedding.
    sub : bool, optional
        Whether to subtract 1 from id values.
    item_id : str, optional
        Item id column name.

    Returns
    -------
    np.ndarray
        Similarity array.
    """
    # * maybe add embedding statistic info like std, mean, etc?
    
    sim = np.zeros(candidate.shape[0])
    batch_size = 10000
    for batch in tqdm(range(0, candidate.shape[0], batch_size)):
        if sub:
            tmp_users = (
                candidate.loc[batch : batch + batch_size - 1, "customer_id"].values - 1
            )
            tmp_items = (
                candidate.loc[batch : batch + batch_size - 1, item_id].values - 1
            )
        else:
            tmp_users = candidate.loc[
                batch : batch + batch_size - 1, "customer_id"
            ].values
            tmp_items = candidate.loc[batch : batch + batch_size - 1, item_id].values
        tmp_user_embd = np.expand_dims(user_embd[tmp_users], 1)  # (batch_size, 1, dim)
        tmp_item_embd = np.expand_dims(item_embd[tmp_items], 2)  # (batch_size, dim, 1)
        tmp_sim = np.einsum("ijk,ikj->ij", tmp_user_embd, tmp_item_embd)
        sim[batch : batch + batch_size] = tmp_sim.reshape(-1)
    return sim
