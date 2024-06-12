import numpy as np
from rdkit import Chem
import torch
import pandas as pd
from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator
import bisect
from tqdm import tqdm
import argparse
import re

def topological_fingerprint_retrieval(dataset):
    rank_list = []
    for i in range(100):
        rand_seed = np.random.randint(0,len(dataset))
        target_r= dataset.iloc[rand_seed]['reactants']
        target_p = dataset.iloc[rand_seed]['production']
        target_similarity = DataStructs.TanimotoSimilarity(Chem.RDKFingerprint(Chem.MolFromSmiles(target_r)), Chem.RDKFingerprint(Chem.MolFromSmiles(target_p)))
        target_num = 0
        for j, reaction_smiles in tqdm(dataset.iterrows(), total=dataset.shape[0]):
            reactants_smi = reaction_smiles[3]
            product_smi = reaction_smiles[4]
            if product_smi == target_p:
                continue
            similarity = DataStructs.TanimotoSimilarity(Chem.RDKFingerprint(Chem.MolFromSmiles(reactants_smi)), Chem.RDKFingerprint(Chem.MolFromSmiles(target_p)))
            if similarity> target_similarity: target_num = target_num+1
        rank_list.append(target_num)
    return rank_list


def morgan_fingerprint_retrieval(dataset):
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rank_list = []
    for i in range(100):
        rand_seed = np.random.randint(0,len(dataset))
        target_r= dataset.iloc[rand_seed]['reactants']
        target_p = dataset.iloc[rand_seed]['production']
        target_similarity = DataStructs.TanimotoSimilarity(morgan_gen.GetFingerprint(Chem.MolFromSmiles(target_r)),
                                                        morgan_gen.GetFingerprint(Chem.MolFromSmiles(target_p)))
        target_num = 0
        for j, reaction_smiles in tqdm(dataset.iterrows(), total=dataset.shape[0]):
            reactants_smi = reaction_smiles[3]
            product_smi = reaction_smiles[4]
            if product_smi == target_p:
                continue
            similarity = DataStructs.TanimotoSimilarity(morgan_gen.GetFingerprint(Chem.MolFromSmiles(reactants_smi)),
                                                    morgan_gen.GetFingerprint(Chem.MolFromSmiles(target_p)))
            if similarity> target_similarity: target_num = target_num+1
        rank_list.append(target_num)
    return rank_list

def embedding_retrieval(dataset):
    rank_list = []
    dataset['reactant_embedding'] = dataset['reactant_embedding'].apply(clean_and_convert_embedding)
    dataset['product_embedding'] = dataset['product_embedding'].apply(clean_and_convert_embedding)
    for i in range(100):
        rand_seed = np.random.randint(0,len(dataset))
        #(1,512)
        target_r= dataset.iloc[rand_seed]['reactant_embedding']
        target_p = dataset.iloc[rand_seed]['product_embedding']
        query_smi = dataset.iloc[rand_seed]['product']
        target_similarity = compute_similarity(target_r, target_p, metric='cosine') 
        target_num = 0
        for j, reaction_smiles in tqdm(dataset.iterrows(), total=dataset.shape[0]):
            reactants_emb = reaction_smiles[2] #reactant_embedding
            product_smi = reaction_smiles[1] #product
            if product_smi == query_smi:
                continue
            similarity = compute_similarity(reactants_emb, target_p, metric='cosine') 
            if similarity> target_similarity: target_num = target_num+1
        rank_list.append(target_num)
    return rank_list

def clean_and_convert_embedding(embedding_str):
    embedding_str = re.sub(r'[\n\[\]]', '', embedding_str)
    embedding = np.array([float(x) for x in embedding_str.split() if x.strip()])
    return embedding

def compute_similarity(query_embedding, dataset_embeddings, metric='dot'):
    if metric == 'dot':
        similarity = np.dot(query_embedding, dataset_embeddings)
    elif metric == 'cosine':
        similarity = np.dot(query_embedding, dataset_embeddings) / (np.linalg.norm(query_embedding) * np.linalg.norm(dataset_embeddings))
    return similarity

def main(args):
    dataset = pd.read_csv(args.dataset_dir,index_col=False)

    if args.retrieval_type == 'topological_fp':
        dataset['reactants'] = dataset['reactants>reagents>production'].str.split('>', expand=True)[0]
        dataset['production'] = dataset['reactants>reagents>production'].str.split('>', expand=True)[2]
        rank_list = topological_fingerprint_retrieval(dataset)

    elif args.retrieval_type == 'morgan_fp':
        dataset['reactants'] = dataset['reactants>reagents>production'].str.split('>', expand=True)[0]
        dataset['production'] = dataset['reactants>reagents>production'].str.split('>', expand=True)[2]
        rank_list = morgan_fingerprint_retrieval(dataset)

    elif args.retrieval_type == 'embedding':
        rank_list = embedding_retrieval(dataset)

    print(rank_list)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--retrieval_type', type=str, required=True)
    argparser.add_argument('--dataset_dir', type=str, required=True)
    argparser.add_argument('--retrieval_num', type=int, default=100)
    args = argparser.parse_args()
    main(args)