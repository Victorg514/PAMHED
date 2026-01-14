import pandas as pd

file_path = "data/final.parquet"

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

df = pd.read_parquet(file_path)
print(df.head(20))  # Show first 20 rows in full width
