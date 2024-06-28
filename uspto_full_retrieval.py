import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator
from tqdm import tqdm
import bisect
from concurrent.futures import ThreadPoolExecutor, as_completed

max_size = 10
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)

input_df = pd.read_csv('data/uspto50k/raw/uspto50k_test.csv', index_col=False)
retri_df = pd.read_csv("data/uspto50k/raw/USPTU_full_filter.csv")

retri_list = []
for i, reaction_smiles in enumerate(tqdm(retri_df['reactants>reagents>production'].values)):
    reaction_smiles = reaction_smiles.strip()
    reactants_smi, _, product_smi = reaction_smiles.split('>')
    retri_fp = morgan_gen.GetFingerprint(Chem.MolFromSmiles(reactants_smi))
    retri_list.append(retri_fp)

index_list = [None] * len(input_df)
sim_list = [None] * len(input_df)

def compute_similarity(i, reaction_smiles, retri_list, morgan_gen, max_size):
    reactants_smi, _, product_smi = reaction_smiles.split('>')
    input_fp = morgan_gen.GetFingerprint(Chem.MolFromSmiles(product_smi))
    result = []
    for j, retri_fp in enumerate(retri_list):
        similarity = DataStructs.TanimotoSimilarity(input_fp, retri_fp)
        if len(result) < max_size:
            bisect.insort(result, (similarity, j))
        elif similarity > result[0][0]:
            bisect.insort(result, (similarity, j))
            result.pop(0)
    result.reverse()
    str_index = ','.join(map(str, [index for _, index in result]))
    str_sim = ','.join(map(str, [sim for sim, _ in result]))
    return i, str_index, str_sim

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(compute_similarity, i, reaction_smiles, retri_list, morgan_gen, max_size): i for i, reaction_smiles in enumerate(input_df['reactants>reagents>production'].values)}
    with tqdm(total=len(futures)) as pbar:
        for future in as_completed(futures):
            i, str_index, str_sim = future.result()
            index_list[i] = str_index
            sim_list[i] = str_sim
            pbar.update(1)

input_df['retrieval_index'] = index_list
input_df['retrieval_similarity'] = sim_list
input_df.to_csv('data/uspto50k/raw/uspto50k_test_application.csv', index=False)