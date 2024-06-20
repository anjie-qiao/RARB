import argparse
from ctypes.wintypes import PINT
import os
import sys
from traceback import print_tb
from rxnmapper import RXNMapper

sys.path.append('.')
sys.path.append('./LocalTransform')

import pickle
import random
import sys
from os.path import join

import numpy as np
import torch
from tqdm import tqdm
import yaml
from easydict import EasyDict as edict
from train_module.Pretrain_Graph import Pretrain_Graph
from yaml import Dumper, Loader
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator
import bisect
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

rxn_mapper = RXNMapper()

def make_dummy_data(rxn, need_map=True):
    # if len(rxn) > 512:
    #     return None
    origin_reactants, _, origin_product = rxn.split('>')
    if need_map:
        rxn = rxn_mapper.get_attention_guided_atom_maps([rxn])[0]['mapped_rxn']
    line1, line2, line3 = rxn.split('>')
    line1 = line1.split('.')
    line2 = line2.split('.')
    reactant = line1 + line2
    product = line3
    reagent = [r for r in reactant if ':' not in r]
    reactant = [r for r in reactant if ':' in r]
    map_num = [len(r.split(':')) - 1 for r in reactant]
    main_reactant = reactant[map_num.index(max(map_num))]
    sub_reactant = [r for r in reactant if r != main_reactant]
    #pad to dummies
    sub_reactant = [r for r in sub_reactant if r != '']
    main_reactant = [main_reactant, []]
    sub_reactant = [[r, []] for r in sub_reactant]
    reagent = '.'.join(reagent)
    return [[main_reactant, sub_reactant], reagent, product, rxn, origin_reactants, origin_product]

## Morgan Fingerprint Retrieval 
def morgan_fingerprint_retrieval(dataset,query_list, max_size):
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    index_list = [None] * len(dataset)
    sim_list = [None] * len(dataset)
    for i, reaction_smiles in enumerate(tqdm(dataset['reactants>reagents>production'].values)):
        reactants_smi, _, product_smi = reaction_smiles.split('>')
        input_fp = morgan_gen.GetFingerprint(Chem.MolFromSmiles(product_smi))
        result = []
        for j, retri_fp in enumerate(query_list):
            if args.data_tpye == "train" and i==j: continue
            similarity = DataStructs.TanimotoSimilarity(input_fp,retri_fp)
            if len(result) < max_size:
                bisect.insort(result, (similarity,j))
            elif similarity > result[0][0]:
                bisect.insort(result, (similarity,j))
                result.pop(0)
        result.reverse()
        str_index = ','.join(map(str, [index for _, index in result]))
        str_sim = ','.join(map(str, [sim for sim, _ in result]))
        index_list[i] = str_index
        sim_list[i] = str_sim
    dataset['retrieval_index'] = index_list
    dataset['retrieval_similarity'] = sim_list
    return dataset
    
## embedding Fingerprint Retrieval 
def embedding_retrieval(dataset,encoded_query,query_list,max_size):
    index_list = []
    sim_list = []
    for i, query_p in enumerate(encoded_query):
        result = []
        for j, query_r in enumerate(query_list):
            if args.data_tpye == "train" and i==j: continue
            similarity = compute_similarity(query_p.flatten().cpu(), query_r.flatten().cpu())
            if len(result) < max_size:
                bisect.insort(result, (similarity,j))
            elif similarity > result[0][0]:
                bisect.insort(result, (similarity,j))
                result.pop(0)
        result.reverse()
        str_index = ','.join(map(str, [index for _, index in result]))
        str_sim = ','.join(map(str, [sim for sim, _ in result]))
        index_list.append(str_index)
        sim_list.append(str_sim) 
    dataset['retrieval_index'] = index_list
    dataset['retrieval_similarity'] = sim_list
    return dataset

def compute_similarity(query_embedding, dataset_embeddings, metric='cosine'):
    if metric == 'dot':
        similarity = np.dot(query_embedding, dataset_embeddings)
    elif metric == 'cosine':
        similarity = np.dot(query_embedding, dataset_embeddings) / (np.linalg.norm(query_embedding) * np.linalg.norm(dataset_embeddings))
    return similarity

def main(args):
    max_size = 10

    device = torch.device('cuda' if torch.cuda.is_available( ) else 'cpu')
    cfg = edict({
        'model':
        yaml.load(open(join(args.config_path, 'model/pretrain_graph.yaml')),
                Loader=Loader),
    })

    output_f = args.input_file.split('.')[0] + '_unirxnfp.csv'
 
    retri_list = []
    
    input_df = pd.read_csv(args.input_file)
    retri_df = pd.read_csv(args.retrieval_file)
    encoded_reactants = torch.load(args.embedding_file)

    if args.retrieval_type == 'morgan':
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
        for i, reaction_smiles in enumerate(tqdm(retri_df['reactants>reagents>production'].values)):
            reaction_smiles = reaction_smiles.strip()
            reactants_smi, _, product_smi = reaction_smiles.split('>')
            retri_fp = morgan_gen.GetFingerprint(Chem.MolFromSmiles(reactants_smi))
            retri_list.append(retri_fp)
        input_df = morgan_fingerprint_retrieval(input_df, retri_list, max_size)
    
    elif args.retrieval_type == 'rxn-embedding':
        model = Pretrain_Graph(cfg, stage='inference')
        model = model.load_from_checkpoint(args.model_dir)
        model = model.to(device)
        model.eval()
        for i, reaction_smiles in enumerate(tqdm(retri_df['reactants>reagents>production'].values)):
            reaction_smiles = reaction_smiles.strip()
            _, _, product_smi = reaction_smiles.split('>')
            reactant_fp = encoded_reactants[i]
            retri_list.append(reactant_fp)

        encoded_query = [] 
        for i, reaction_smiles in enumerate(tqdm(input_df['reactants>reagents>production'].values)):
            reaction_smiles = reaction_smiles.strip()
            _, _, product_smi = reaction_smiles.split('>')
            dummy_data = make_dummy_data(reaction_smiles, need_map=True)
            _, product_fp = model.generate_reaction_fp_mix(dummy_data, device, no_reagent=True)#set to True if you want to ignore reagents
            encoded_query.append(product_fp)
        input_df = embedding_retrieval(input_df, encoded_query, retri_list, max_size)

    input_df.to_csv(output_f, index=False)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_dir', type=str, default='ckpt/uni_rxn_base.ckpt', help='path to the pretrained base model checkpoint')
    argparser.add_argument('--input_file', type=str, help='path to the input file for query')
    argparser.add_argument('--retrieval_file', type=str, help='path to the retrieval_file for featurization')
    argparser.add_argument('--embedding_file', type=str,)
    argparser.add_argument('--data_tpye', type=str,)
    argparser.add_argument('--retrieval_type', type=str, help='select retrieval type for dataset')
    argparser.add_argument('--config_path', type=str, default='config/')
    args = argparser.parse_args()
    main(args)