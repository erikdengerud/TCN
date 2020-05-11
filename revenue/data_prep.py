import tarfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from functools import reduce


def f(a, b):
    if a and not b:
        return True
    else:
        return False


num_entries_req = 16
all_df = []
describe_dicts = []
comp_sect_dicts = []
all_companies = set()

sector_data = {
    path[path.find("____") + len("____") : -len(".parquet")]: path
    for path in glob.glob("revenue/data/fundamental_data/total_revenue/*")
}

print("Start loop.")
for sector in sector_data.keys():
    try:
        print(f"Treating sector : {sector}.")
        df_orig = pd.read_parquet(sector_data[sector], engine="pyarrow")
        if df_orig.empty:
            print(f"Sector {sector} is an empty df!")
            raise Exception
        df = df_orig.copy()
        num_companies_sector = df.shape[1]
        num_months_sector = df.shape[0]
        companies_sector = df.columns

        df_stats = df.copy()
        df_stats["reported"] = len(df_stats.columns) - df_stats.isnull().sum(axis=1)
        df_stats["month"] = df_stats.index.month
        df_months = (
            pd.concat((df_stats["month"], df_stats["reported"]), axis=1)
            .groupby("month")
            .sum()
        )
        print(df_months)

        drop_set = set()
        drop_months = [1, 2, 4, 5, 7, 8, 10, 11]
        for m in drop_months:
            fail_cols = df_stats.columns[
                np.where(
                    df_stats[df_stats["month"] == m].notna().apply(np.sum, axis=0) > 0
                )[0].tolist()
            ].tolist()
            drop_set.update(fail_cols)
        df_stats = df_stats.drop(columns=list(drop_set))
        drop_set.remove("month")
        drop_set.remove("reported")
        df = df.drop(columns=list(drop_set))
        print(
            f"""
            Dropping {len(drop_set)} columns that have non standard fiscal period. This is 
            {len(drop_set) / num_companies_sector * 100:.2f}% of the comanies. We 
            still have {len(df.columns) / num_companies_sector * 100:.2f}% of the 
            original companies.
            """
        )

        df["reported"] = len(df.columns) - df.isnull().sum(axis=1)
        df["month"] = df.index.month
        df_months = (
            pd.concat((df["month"], df["reported"]), axis=1).groupby("month").sum()
        )
        print(df_months)

        df = df[
            (df["month"] == 3)
            | (df["month"] == 6)
            | (df["month"] == 9)
            | (df["month"] == 12)
        ]

        df = df.drop(columns=["month", "reported"])
        tot = 0
        drop_cols = []
        for col in df.columns:
            df_test = pd.DataFrame([np.roll(df[col], 1), df[col].to_numpy()]).T
            df_test.columns = ["rolled", "original"]
            df_test = df_test.isna()
            df_test["susp"] = df_test.apply(
                lambda r: f(r["rolled"], r["original"]), axis=1
            )
            count_df = df_test["susp"][1:].value_counts()
            try:
                if count_df[True] > 1 or (
                    count_df[True] == 1 and df_test["original"][0].notna()
                ):
                    tot += 1
                    drop_cols.append(col)
            except:
                pass
        df = df.drop(columns=drop_cols)
        print(
            f"""
            Dropping {tot} columns because of missing reported earnings. The dataset 
            contains {len(df.columns) / len(df_orig.columns) * 100:.2f} % of the original 
            companies.
            """
        )

        drop_cols = (
            df.count().where(df.count() < num_entries_req).dropna(axis=0).index.tolist()
        )
        df = df.drop(columns=drop_cols)
        print(
            f"""
            Dropping {len(drop_cols)} columns because of not enough reported earnings. 
            The dataset contains {len(df.columns) / len(df_orig.columns)* 100:.2f}% of 
            the original companies.
            """
        )

        d = {}

        vals = df.values.flatten()
        vals = vals[np.logical_not(np.isnan(vals))]
        d["sector"] = sector
        d["count"] = len(df.columns)
        d["min"] = np.min(vals)
        d["max"] = np.max(vals)
        d["mean"] = np.mean(vals)
        d["std"] = np.std(vals)
        d["longest"] = max(df.count())
        d["shortest"] = min(df.count())
        describe_dicts.append(d)

        # save sector metadata - > company belonging to sector
        drop_cols = []
        for column in df.columns:
            comp_sect_dicts.append({"company": column, "sector": sector})
            if column in all_companies:
                # The comnpany has already been added
                drop_cols.append(column)

        df = df.drop(columns=drop_cols)

        all_companies.update(df.columns)

        all_df.append(df)
    except Exception as e:
        print(e)

print("Done loop.")

# dfs = [df.set_index("date") for df in all_df]
df_all = reduce(lambda df1, df2: pd.concat([df1, df2], axis=1), all_df)
df_all.head()
# save df
df_all.to_csv("revenue/data/processed_companies.csv")
# Total stats
d = {}
vals = df_all.values.flatten()
print(vals)
print(vals.shape)
print(vals.dtype)
vals = vals[np.logical_not(np.isnan(vals))]
d["sector"] = "Total"
d["count"] = len(df_all.columns)
d["min"] = np.min(vals)
d["max"] = np.max(vals)
d["mean"] = np.mean(vals)
d["std"] = np.std(vals)
d["longest"] = max(df_all.count())
d["shortest"] = min(df_all.count())
describe_dicts.append(d)

# describe df print
df_describe = pd.DataFrame(describe_dicts)
print(df_describe)
df_describe.to_csv("revenue/data/describe_data.csv")
# comp : sect data
df_comp_sect = pd.DataFrame(comp_sect_dicts)
df_comp_sect.to_csv("revenue/data/comp_sect_meta.csv")
