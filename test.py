import numpy as np
import pandas as pd

from pathlib import Path
from src.metrics.eval_csv_helpers import canonicalize, compute_confidence, assign_groups, compute_accuracy



csv_file = Path('./samples/uspto50k_test/epoch=479_top_5_accuracy=0.828_3_application_T=500_n=100_seed=1.csv')

#csv_file = Path('./round_samples/epoch=599_top_5_accuracy=0.812_3_application_T=500_n=100_seed=1.csv')
df = pd.read_csv(csv_file)
#df = df.iloc[:6400]
df_unique = df.drop_duplicates(subset=['pred'])
print(len(df_unique))
print(len(df)/100)
df['from_file'] = 'from_file'
df = assign_groups(df, samples_per_product_per_file=100)
df.loc[(df['product'] == 'C') & (df['true'] == 'C'), 'true'] = 'Placeholder'

df_processed = compute_confidence(df)
for key in ['product', 'pred']:
    df_processed[key] = df_processed[key].apply(canonicalize)

print(compute_accuracy(df_processed, top=[1, 3, 5, 10], scoring=lambda df: np.log(df['confidence'])))
