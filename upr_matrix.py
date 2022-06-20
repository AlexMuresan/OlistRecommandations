import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

SHORT_DF_ENTRIES = 400


def generate_upr_matrix(
    df: pd.DataFrame,
    row_label: str = "customer_unique_id",
    column_label: str = "product_id",
    values_label: str = "review_score",
):
    # creating matrix with customer_unique_id on rows and product_id on columns
    upr_matrix = np.full((df[row_label].max() + 1, df[column_label].max() + 1), np.NaN)

    # filling upr_matrix with available values
    for _, row in df.iterrows():
        upr_matrix[row[row_label]][row[column_label]] = row[values_label]

    upr_matrix_imputed = upr_matrix.copy()

    for idx in tqdm(range(df[row_label].max()), desc="Filling matrix with average"):
        matrix_row = upr_matrix[idx][:]
        if np.any(np.isnan(matrix_row)):
            upr_matrix[idx][
                np.random.randint(low=0, high=matrix_row.shape[0])
            ] = np.random.randint(low=1, high=5)
        upr_matrix_imputed[idx][:] = np.nan_to_num(matrix_row, (np.nanmean(matrix_row)))

    return upr_matrix, upr_matrix_imputed


if __name__ == "__main__":
    data_folder = "/Users/alex/Workspace/Datasets/OlistEcommercePublicDataset"

    orders_df = pd.read_csv(os.path.join(data_folder, "olist_orders_dataset.csv"))
    reviews_df = pd.read_csv(
        os.path.join(data_folder, "olist_order_reviews_dataset.csv")
    )
    products_df = pd.read_csv(os.path.join(data_folder, "olist_products_dataset.csv"))
    order_items_df = pd.read_csv(
        os.path.join(data_folder, "olist_order_items_dataset.csv")
    )
    customer_df = pd.read_csv(os.path.join(data_folder, "olist_customers_dataset.csv"))

    dfs = [orders_df, reviews_df, products_df, order_items_df, customer_df]

    # Converting ID columns from 'object' type to string
    for df in dfs:
        for column, ctype in zip(df.columns, df.dtypes):
            if ctype == "object":
                df[column] = df[column].astype("string")

    # Changing customer unique ID's from random string to integer
    customer_id_dict = customer_df["customer_unique_id"].to_dict()
    customer_id_dict_reversed = {v: k for k, v in customer_id_dict.items()}
    customer_df["customer_unique_id"] = customer_df["customer_unique_id"].map(
        customer_id_dict_reversed
    )
    unique_id_df = pd.merge(
        orders_df[["order_id", "customer_id"]],
        customer_df[["customer_id", "customer_unique_id"]],
        on=["customer_id"],
        how="inner",
    )

    # Changing product unique ID's from random string to integer
    product_id_dict = order_items_df["product_id"].to_dict()
    product_id_dict_reversed = {v: k for k, v in product_id_dict.items()}
    order_items_df["product_id"] = order_items_df["product_id"].map(
        product_id_dict_reversed
    )
    product_and_order_id_df = pd.merge(
        orders_df[["order_id", "customer_id"]],
        order_items_df[["order_id", "product_id"]],
        on=["order_id"],
        how="inner",
    )

    user_product_order_id_df = pd.merge(
        unique_id_df,
        product_and_order_id_df,
        on=["order_id", "customer_id"],
        how="inner",
    )
    final_df = pd.merge(
        user_product_order_id_df,
        reviews_df[["order_id", "review_score"]],
        on=["order_id"],
        how="inner",
    ).drop(["customer_id", "order_id"], axis=1)

    final_df["customer_unique_id"] = final_df["customer_unique_id"].astype(np.int32)
    final_df["product_id"] = final_df["product_id"].astype(np.int32)
    final_df["review_score"] = final_df["review_score"].astype(np.int8)

    tmp_short_df = final_df.head(SHORT_DF_ENTRIES)
    selection = tmp_short_df["customer_unique_id"].unique().tolist()
    final_df_short = final_df[
        pd.DataFrame(final_df["customer_unique_id"].tolist())
        .isin(selection)
        .any(1)
        .values
    ]

    print(len(selection), "unique ids selected")
    print(final_df_short.shape[0], "entries in new dataframe")

    upr_matrix_short, upr_matrix_short_imputed = generate_upr_matrix(final_df_short)
    print(upr_matrix_short.shape)
    print(upr_matrix_short_imputed.shape)
    print()

    print(upr_matrix_short[0][:])
    print()
    print(upr_matrix_short_imputed[0][:])
    print()
