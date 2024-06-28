import argparse
import os
import sys
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
import bisect
import re

rxn_mapper = RXNMapper()

def make_dummy_data(rxn, need_map=True):
    # if len(rxn) > 512:
    #     return None
    try:
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
    except Exception as e:
        print(f"Error processing rxn: {rxn}")
        print(f"Error: {e}")
        return "error"


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available( ) else 'cpu')
    cfg = edict({
        'model':
        yaml.load(open(join(args.config_path, 'model/pretrain_graph.yaml')),
                Loader=Loader),
        'dataset':
        yaml.load(open(join(args.config_path, 'dataset/pretrain.yaml')),
                Loader=Loader),
    })

    model = Pretrain_Graph(cfg, stage='inference')
    model = model.load_from_checkpoint(args.model_dir)
    model = model.to(device)
    model.eval()

    encoded_reactants = []
    encoded_products = []
    error_indices = []
    retri_df = pd.read_csv(args.retrieval_file)

    for i, reaction_smiles in enumerate(tqdm(retri_df['reactants>reagents>production'].values)):
        reaction_smiles = reaction_smiles.strip()
        dummy_data = make_dummy_data(reaction_smiles, need_map=True)
        #embedding shape: (1, 512)
        if dummy_data != "error":
            reactant_fp, product_fp = model.generate_reaction_fp_mix(dummy_data, device, no_reagent=True)#set to True if you want to ignore reagents
            encoded_reactants.append(reactant_fp)
            encoded_products.append(product_fp)
        else:
            error_indices.append(i)

    retri_df.drop(error_indices, inplace=True)
    retri_df.to_csv(args.retrieval_file.split('.')[0] + "_filter.csv", index=False)
    error_indices = pd.DataFrame(error_indices)
    error_indices.to_csv("error_indices.csv", index=False)
    
    encoded_reactants = torch.cat(encoded_reactants,dim=0)
    encoded_products = torch.cat(encoded_products,dim=0)
    print(encoded_reactants.shape)
    print(encoded_products.shape)
    torch.save(encoded_reactants, 'rxn_encoded_reac_uspto_full.pt')
    torch.save(encoded_products, 'rxn_encoded_prod_uspto_full.pt')
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_dir', type=str, default='ckpt/uni_rxn_base.ckpt', help='path to the pretrained base model checkpoint')
    argparser.add_argument('--retrieval_file', type=str, help='path to the retrieval_file for featurization')
    argparser.add_argument('--config_path', type=str, default='config/')
    args = argparser.parse_args()
    main(args)