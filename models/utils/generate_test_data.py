import pandas as pd

def generate_test_data(df, size, balance, seed):
    if balance == "Original":
        return df.sample(n=size, random_state=seed)

    elif balance == "More YES cases":
        df_yes = df[df["y"] == "yes"]
        df_no = df[df["y"] == "no"]

        yes_size = int(size * 0.4)   # 40% YES
        no_size = size - yes_size

        return pd.concat([
            df_yes.sample(n=min(yes_size, len(df_yes)), random_state=seed),
            df_no.sample(n=min(no_size, len(df_no)), random_state=seed)
        ]).sample(frac=1, random_state=seed)  # shuffle