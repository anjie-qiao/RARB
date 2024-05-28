import numpy as np
import pandas as pd

from pathlib import Path
from src.metrics.eval_csv_helpers import canonicalize, compute_confidence, assign_groups, compute_accuracy


#csv_file = Path('./samples/uspto50k_test/epoch=479_top_5_accuracy=0.930_T=500_n=10_seedddddd=1.csv')
csv_file = Path('./samples/uspto50k_test/epoch=479_top_5_accuracy=0.930_T=500_n=10_seedddddd=1.csv')
#csv_file = Path('./samples/uspto50k_test/epoch=479_top_5_accuracy=0.930_T=500_n=10_seed=1.csv')
df = pd.read_csv(csv_file)
df['from_file'] = 'epoch=479_top_5_accuracy=0.930_T=500_n=10_seed22=1'
df = assign_groups(df, samples_per_product_per_file=10)
df.loc[(df['product'] == 'C') & (df['true'] == 'C'), 'true'] = 'Placeholder'

df_processed = compute_confidence(df)

for key in ['product', 'pred']:
    df_processed[key] = df_processed[key].apply(canonicalize)

print(compute_accuracy(df_processed, top=[1, 3, 5, 10], scoring=lambda df: np.log(df['confidence'])))