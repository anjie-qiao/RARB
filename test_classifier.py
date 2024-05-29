import argparse
import os
import pandas as pd

from src.utils import disable_rdkit_logging, parse_yaml_config, set_deterministic
from src.analysis.rdkit_functions import build_molecule
from src.data.retrobridge_dataset import RetroBridgeDataModule, RetroBridgeDatasetInfos

from rdkit import Chem
from tqdm import tqdm
from src.data import utils
from src.frameworks.rerto_classifier import RertoClassifier
from pdb import set_trace
import torch


def main(args):
    torch_device = 'cuda:6' if args.device == 'gpu' else 'cpu'
    data_root = os.path.join(args.data, args.dataset)




    classifier_model = RertoClassifier
    classifier_model = classifier_model.load_from_checkpoint(args.checkpoint, map_location=torch_device)
    classifier_model.eval().to(torch_device)
    classifier_model.visualization_tools = None

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

    dataloader = datamodule.test_dataloader() if args.mode == 'test' else datamodule.val_dataloader()

    for i, data in enumerate(tqdm(dataloader)):
        bs = len(data.batch.unique())
        data = data.to(torch_device)
        product, p_node_mask = utils.to_dense(data.p_x, data.p_edge_index, data.p_edge_attr, data.batch)
        product = product.mask(p_node_mask)
    
        context, c_node_mask = utils.to_dense(data.context_x, data.context_edge_index, data.context_edge_attr,
                                              data.batch)
        context = context.mask(c_node_mask)
        px_label = context.X[:,:,-1]
        pe_label = context.E[:,:,:,-1]

        # product_label, c_node_mask = utils.to_dense(data.node_label, data.edge_label_index, data.edge_label, data.batch)
        # product_label = product_label.mask(c_node_mask)

        node_mask = p_node_mask

        product_data = {'X_t': product.X, 'E_t': product.E, 'y_t': product.y, 't': None, 'node_mask': node_mask}
        product_extra_data = classifier_model.compute_extra_data(product_data)
        pred_label = classifier_model.forward(product, product_extra_data, node_mask)

        node_correct = torch.all(torch.argmax(pred_label['X'], dim=-1) == px_label.squeeze(-1), dim=1)
        edge_comparison = torch.argmax(pred_label['E'], dim=-1) == pe_label.squeeze(-1)
        edge_comparison_flat = edge_comparison.view(edge_comparison.size(0), -1)
        edge_correct = torch.all(edge_comparison_flat, dim=1)
        print("node_accuracy: ",node_correct.float().mean().item())
        print("edge_accuracy: ",edge_correct.float().mean().item())
        
if __name__ == '__main__':
    disable_rdkit_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=argparse.FileType(mode='r'), required=True)
    parser.add_argument('--checkpoint', action='store', type=str, required=True)
    parser.add_argument('--mode', action='store', type=str, required=True)
    parser.add_argument('--n_samples', action='store', type=int, required=True)
    parser.add_argument('--sampling_seed', action='store', type=int, required=False, default=None)
    parser.add_argument('--use_one_hot', action='store_true', required=False, default=False)
    main(args=parse_yaml_config(parser.parse_args()))
