import os
import os
import pandas as pd
print("Executing RFM Pipeline...")
# 1. Load the ledger
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
SALES_DATA_PATH = os.path.join(DATA_DIR, "raw_sales_data.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "rfm_segments.csv")
df = pd.read_csv(SALES_DATA_PATH)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalCost'] = df['UnitPrice'] * df['Quantity']
# Reference date is one day after the last transaction in the data, so this
# stays correct even if simulator.py's date window changes later.
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
# 2. Group by customer to calculate raw metrics
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency
    'InvoiceNo': 'nunique',                                  # Frequency
    'TotalCost': 'sum'                                       # Monetary
}).reset_index()
rfm.columns = ['CustomerID', 'recency', 'frequency', 'monetary']
# 3. Calculate quintile rankings using Pandas qcut (Quantile-based discretization)
# For Recency, smaller values get a higher score (5)
rfm['r_score'] = pd.qcut(rfm['recency'].rank(method='first', ascending=False), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
# 4. Map Segment Names using conditions
def assign_segment(row):
    r, f, m = row['r_score'], row['f_score'], row['m_score']
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3 and m >= 3:
        return 'Loyal Customers'
    elif r >= 3 and f <= 2:
        return 'Promising / New'
    elif r <= 2 and f >= 3:
        return 'At Risk / Can\'t Lose Them'
    elif r <= 2 and f <= 2:
        return 'Hibernating / Lost'
    else:
        return 'About to Sleep / Average'
rfm['customer_segment'] = rfm.apply(assign_segment, axis=1)
# 5. Output to a distinct file
rfm.to_csv(OUTPUT_PATH, index=False)
print("Successfully executed calculations!")
print(f"Output file '{OUTPUT_PATH}' generated with data science schema.")
