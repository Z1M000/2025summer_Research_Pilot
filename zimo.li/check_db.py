import lancedb
import csv
import pandas as pd

DB_PATH = "/home/ubuntu/database"
TABLE_NAME = "task11test_zimo"
db = lancedb.connect(DB_PATH)
table = db.open_table(TABLE_NAME)
df = table.to_pandas()
df.to_csv("/home/ubuntu/openalex/workspace/zimo.li/view_db.csv", index=True)