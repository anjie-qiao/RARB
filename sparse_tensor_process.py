#encoded USPTO application reactants has 969283, but only about 150K of them will be retrieved
#convert to sparse matrix storage:  1.85GB -> 350MB
import torch
import pandas as pd
from tqdm import tqdm

def save_sparse_tensor(tensor, indices_to_keep):
    indices = torch.tensor(indices_to_keep)
    values = tensor[indices_to_keep]
    sparse_tensor = torch.sparse_coo_tensor(indices.unsqueeze(0), values, tensor.size())
    return sparse_tensor

df_train= pd.read_csv("data/uspto50k/raw/uspto50k_train_application.csv")
df_val= pd.read_csv("data/uspto50k/raw/uspto50k_val_application.csv")
df_test= pd.read_csv("data/uspto50k/raw/uspto50k_test_application.csv")
df = pd.concat([df_test, df_train, df_val], ignore_index=True)
print(len(df))

indices_to_keep = set()
for i,row in enumerate(tqdm(df['retrieval_index'].values)):
    row_list = [int(item) for item in row.split(',')]
    indices_to_keep.update(row_list[:5])
sorted_indices_list = torch.tensor(sorted(list(indices_to_keep))).to("cuda")
print("retrirval_num: ",len(sorted_indices_list))

original_tensor = torch.load('data/uspto50k/raw/rxn_encoded_reac_uspto_full.pt')
print(original_tensor.shape)
sparse_tensor= save_sparse_tensor(original_tensor,sorted_indices_list)
torch.save(sparse_tensor, 'data/uspto50k/raw/rxn_encoded_reac_uspto_full_sparse.pt')


########################
#test whether dense_tensor and sparse_tesor is equal
df_train= pd.read_csv("data/uspto50k/raw/uspto50k_train_application.csv")
df_val= pd.read_csv("data/uspto50k/raw/uspto50k_val_application.csv")
df_test= pd.read_csv("data/uspto50k/raw/uspto50k_test_application.csv")
df = pd.concat([df_test, df_train, df_val], ignore_index=True)
densc_tensor = torch.load("data/uspto50k/raw/rxn_encoded_reac_uspto_full.pt")
sparse_tensor = torch.load("data/uspto50k/raw/rxn_encoded_reac_uspto_full_sparse.pt")
error_num = 0
for i,row in enumerate(tqdm(df['retrieval_index'].values)):
    row_list = [int(item) for item in row.split(',')][:5]
    for j in row_list:
        if not torch.equal(densc_tensor[j] ,sparse_tensor[j]) : error_num+=1
print(error_num) 
