import argparse
import os
from time import time
import pandas as pd
import torch

from src.utils import disable_rdkit_logging, parse_yaml_config, set_deterministic

from src.data import utils
from src.analysis.rdkit_functions import build_molecule
from src.frameworks.discrete_diffusion import DiscreteDiffusion
from src.frameworks.markov_bridge import MarkovBridge  
from src.frameworks.one_shot_model import OneShotModel
from src.data.retrobridge_dataset import RetroBridgeDataModule, RetroBridgeDatasetInfos

from src.frameworks.rerto_classifier import RertoClassifier

from rdkit import Chem
from tqdm import tqdm
import time

from pdb import set_trace
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main(args):
    torch_device = 'cuda' if args.device == 'gpu' else 'cpu'
    data_root = os.path.join(args.data, args.dataset)
    checkpoint_name = args.checkpoint_decoder.split('/')[-1].replace('.ckpt', '')

    output_dir = os.path.join(args.samples, f'{args.dataset}_{args.mode}')
    table_name = f'{checkpoint_name}_T={args.n_steps}_n={args.n_samples}_seednew={args.sampling_seed}.csv'
    table_path = os.path.join(output_dir, table_name)

    skip_first_n = 0
    prev_table = pd.DataFrame()
    if os.path.exists(table_path):
        prev_table = pd.read_csv(table_path)
        skip_first_n = len(prev_table) // args.n_samples
        assert len(prev_table) % args.batch_size == 0

    print(f'Skipping first {skip_first_n} data points')

    os.makedirs(output_dir, exist_ok=True)
    print(f'Samples will be saved to {table_path}')

    # Loading model form checkpoint (all hparams will be automatically set)
    decoder_model = MarkovBridge
    classifier_model = RertoClassifier
    
    print('Model class:', decoder_model)

    classifier_model = classifier_model.load_from_checkpoint(args.checkpoint_classifier, map_location=torch_device)
    classifier_model.eval().to(torch_device)
    classifier_model.visualization_tools = None

    decoder_model = decoder_model.load_from_checkpoint(args.checkpoint_decoder, map_location=torch_device)
    
    datamodule = RetroBridgeDataModule(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        extra_nodes=args.extra_nodes,
        evaluation=False,
        swap=args.swap,
    )
    dataset_infos = RetroBridgeDatasetInfos(datamodule)

    set_deterministic(args.sampling_seed)
    decoder_model.eval().to(torch_device)
    decoder_model.visualization_tools = None
    decoder_model.T = args.n_steps

    
  
    group_size = args.n_samples

    ident = 0
    true_molecules_smiles = []
    pred_molecules_smiles = []
    product_molecules_smiles = []
    computed_scores = []
    true_atom_nums = []
    sampled_atom_nums = []
    computed_nlls = []
    computed_ells = []
    computed_nc = []

    dataloader = datamodule.test_dataloader() if args.mode == 'test' else datamodule.val_dataloader()

    for i, data in enumerate(tqdm(dataloader)):
        if i * args.batch_size < skip_first_n:
            continue

        bs = len(data.batch.unique())
        batch_groups = []
        batch_scores = []
        batch_nll = []
        batch_ell = []
        batch_node_label = []

        ground_truth = []
        input_products = []
        for sample_idx in range(group_size):
            data = data.to(torch_device)
            

            pred_molecule_list, true_molecule_list, products_list, scores, nlls, ells, node_correct = decoder_model.sample_batch(
                data=data,
                batch_id=ident,
                batch_size=bs,
                save_final=0,
                keep_chain=0,
                number_chain_steps_to_save=1,
                sample_idx=sample_idx,
                save_true_reactants=True,
                use_one_hot=args.use_one_hot,

                torch_device=torch_device,
                checkpoint_classifier = classifier_model,

            )

            batch_groups.append(pred_molecule_list)
            batch_scores.append(scores)
            batch_nll.append(nlls)
            batch_ell.append(ells)
            batch_node_label.append(node_correct)

            if sample_idx == 0:
                ground_truth.extend(true_molecule_list)
                input_products.extend(products_list)

        # Regrouping sampled reactants for computing top-N accuracy
        grouped_samples = []
        grouped_scores = []
        grouped_nlls = []
        grouped_ells = []
        for mol_idx_in_batch in range(bs):
            mol_samples_group = []
            mol_scores_group = []
            nlls_group = []
            ells_group = []

            for batch_group, scores_group, nll_gr, ell_gr in zip(batch_groups, batch_scores, batch_nll, batch_ell):
                mol_samples_group.append(batch_group[mol_idx_in_batch])
                mol_scores_group.append(scores_group[mol_idx_in_batch])
                nlls_group.append(nll_gr[mol_idx_in_batch])
                ells_group.append(ell_gr[mol_idx_in_batch])

            assert len(mol_samples_group) == group_size
            grouped_samples.append(mol_samples_group)
            grouped_scores.append(mol_scores_group)
            grouped_nlls.append(nlls_group)
            grouped_ells.append(ells_group)

        # Writing smiles
        for true_mol, product_mol, pred_mols, pred_scores, nlls, ells, node_corrects in zip(
                ground_truth, input_products, grouped_samples, grouped_scores, grouped_nlls, grouped_ells, batch_node_label
        ):
            true_mol, true_n_dummy_atoms = build_molecule(
                true_mol[0], true_mol[1], dataset_infos.atom_decoder, return_n_dummy_atoms=True
            )
            true_smi = Chem.MolToSmiles(true_mol) # type: ignore

            product_mol = build_molecule(product_mol[0], product_mol[1], dataset_infos.atom_decoder)
            product_smi = Chem.MolToSmiles(product_mol) # type: ignore

            for pred_mol, pred_score, nll, ell, node_correct in zip(pred_mols, pred_scores, nlls, ells, node_corrects):
                pred_mol, n_dummy_atoms = build_molecule(
                    pred_mol[0], pred_mol[1], dataset_infos.atom_decoder, return_n_dummy_atoms=True
                )
                pred_smi = Chem.MolToSmiles(pred_mol) # type: ignore
                true_molecules_smiles.append(true_smi)
                product_molecules_smiles.append(product_smi)
                pred_molecules_smiles.append(pred_smi)
                computed_scores.append(pred_score)
                true_atom_nums.append(RetroBridgeDatasetInfos.max_n_dummy_nodes - true_n_dummy_atoms)
                sampled_atom_nums.append(RetroBridgeDatasetInfos.max_n_dummy_nodes - n_dummy_atoms)
                computed_nlls.append(nll)
                computed_ells.append(ell)
                computed_nc.append(node_correct)
  
        table = pd.DataFrame({
            'product': product_molecules_smiles,
            'pred': pred_molecules_smiles,
            'true': true_molecules_smiles,
            'score': computed_scores,
            'true_n_dummy_nodes': true_atom_nums,
            'sampled_n_dummy_nodes': sampled_atom_nums,
            'nll': computed_nlls,
            'ell': computed_ells,
            'nc_label':computed_nc,
        }) 
        full_table = pd.concat([prev_table, table])
        full_table.to_csv(table_path, index=False)


if __name__ == '__main__':
    disable_rdkit_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=argparse.FileType(mode='r'), required=True)
    parser.add_argument('--checkpoint_decoder', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_classifier', action='store', type=str, required=True)
    parser.add_argument('--samples', action='store', type=str, required=True)
    parser.add_argument('--model', action='store', type=str, required=True)
    parser.add_argument('--mode', action='store', type=str, required=True)
    parser.add_argument('--n_samples', action='store', type=int, required=True)
    parser.add_argument('--n_steps', action='store', type=int, required=False, default=None)
    parser.add_argument('--sampling_seed', action='store', type=int, required=False, default=None)
    parser.add_argument('--use_one_hot', action='store_true', required=False, default=False)
    main(args=parse_yaml_config(parser.parse_args()))
