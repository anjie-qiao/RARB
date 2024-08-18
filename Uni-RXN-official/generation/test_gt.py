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
from tqdm import tqdm
import pandas as pd



input_file = "/data/uspto50k/raw/uspto50k_train.csv"
output_f = input_file.split('.')[0] + '_unirxnGT.csv'
input_df = pd.read_csv(input_file)
index_list = [None] * len(input_df)
for i, reaction_smiles in enumerate(tqdm(input_df['reactants>reagents>production'].values)):
    index_list[i]  = str(i)
input_df['retrieval'] = index_list

input_df.to_csv(output_f, index=False)

