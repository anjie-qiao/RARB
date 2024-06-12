import argparse
import os
import pandas as pd

from src.utils import disable_rdkit_logging, parse_yaml_config, set_deterministic
from src.analysis.rdkit_functions import build_molecule
from src.data.retrobridge_dataset import RetroBridgeDataModule, RetroBridgeDatasetInfos
import torch.nn.functional as F

from rdkit import Chem
from tqdm import tqdm
from src.data import utils
from src.frameworks.rerto_classifier import RertoClassifier
from src.analysis.visualization import ClassifierVisualization
from pdb import set_trace
import torch
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, roc_auc_score
from torch_geometric.utils import to_dense_adj, to_dense_batch, scatter
import time


def main(args):
    torch_device = 'cuda' if args.device == 'gpu' else 'cpu'
    data_root = os.path.join(args.data, args.dataset)

    classifier_model = RertoClassifier
    classifier_model = classifier_model.load_from_checkpoint(args.checkpoint, map_location=torch_device)
    classifier_model.eval().to(torch_device)
    
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
    visualization_tools = ClassifierVisualization(dataset_infos)



    set_deterministic(args.sampling_seed)

    dataloader = datamodule.test_dataloader() if args.mode == 'test' else datamodule.val_dataloader()

    #graph_correct_all = torch.tensor([]).to(torch_device)
    # node_correct_all = torch.tensor([]).to(torch_device)
    # edge_correct_all = torch.tensor([]).to(torch_device)
    # node_correct_nozero = torch.tensor([]).to(torch_device)
    # edge_correct_nozero = torch.tensor([]).to(torch_device)

    #Confusion Matrix
    X_true = torch.tensor([]).to("cpu")
    X_pred = torch.tensor([]).to("cpu")
    #X_pred_prob = torch.tensor([]).to("cpu")
    E_true = torch.tensor([]).to("cpu")
    E_pred = torch.tensor([]).to("cpu")
    #E_pred_prob = torch.tensor([]).to("cpu")
    
    for i, data in enumerate(tqdm(dataloader)):
        bs = len(data.batch.unique())
        data = data.to(torch_device)

        product, p_node_mask = utils.to_dense(data.p_x, data.p_edge_index, data.p_edge_attr, data.batch)
        product = product.mask(p_node_mask)
        context, c_node_mask = utils.to_dense(data.context_x, data.context_edge_index, data.context_edge_attr, data.batch)
        context = context.mask(c_node_mask)
        product_label = {
            'X':context.X[:,:,-1],
            'E':context.E[:,:,:,-1],
        }
        node_mask = p_node_mask
        # (BS, N)
        dummy_node_mask = product.X[...,-1] != 1
        new_node_mask = node_mask & dummy_node_mask
        #(compactN,)
        product_label['X_flat'] = product_label['X'][new_node_mask]
        X_true = torch.cat([X_true,product_label['X_flat'].to("cpu")], dim=-1)
        #Invalid edge and diag==0
        edge_mask = (product.E[...,0] != 1) & (torch.sum(product.E, dim=-1) != 0)  
        product_label['E_flat'] = product_label['E'][edge_mask]
        E_true = torch.cat([E_true,product_label['E_flat'].to("cpu")], dim=-1)

        product_data = {'X_t': product.X, 'E_t': product.E, 'y_t': product.y, 't': None, 'node_mask': node_mask}
        product_extra_data = classifier_model.compute_extra_data(product_data)
        pred_label = classifier_model.forward(product, product_extra_data, node_mask, edge_mask, new_node_mask)
        #(bs, n, )
        #node_predicted_labels = torch.argmax(pred_label['X'], dim=-1)
        probabilities = F.softmax(pred_label['X'], dim=-1)
        positive_class_prob = probabilities[:, :, 1]
        print(args.threshold)
        node_predicted_labels = (positive_class_prob >= args.threshold).long()

        #node_pred_prob =  F.softmax(pred_label['X'], dim=-1)
        #X_pred_prob = torch.cat([X_pred_prob, node_pred_prob[new_node_mask][:,1].to("cpu")], dim=-1)
        X_pred = torch.cat([X_pred,node_predicted_labels[new_node_mask].to("cpu")], dim=-1)
        #(bs, n, n, )
        edge_predicted_labels = torch.argmax(pred_label['E'], dim=-1)
        #edge_pred_prob =  F.softmax(pred_label['E'], dim=-1)
        #E_pred_prob = torch.cat([E_pred_prob, edge_pred_prob[edge_mask][:,1].to("cpu")], dim=-1)
        E_pred = torch.cat([E_pred,edge_predicted_labels[edge_mask].to("cpu")], dim=-1)
        
        node_predicted_labels[~new_node_mask] = 0
        edge_predicted_labels[~edge_mask] = 0

        # visualizaing predicted graph
        #(bs,n)
        correct_mask = node_predicted_labels == context.X[...,-1]
        #(bs)
        node_correct = torch.all(correct_mask, dim=1)
        
        random_index = torch.randint(0, (~node_correct).int().sum().item(), (1,))

        products_V, V_node_mask = utils.to_dense(data.p_x, data.p_edge_index, data.p_edge_attr, data.batch)
        products_V = products_V.mask(V_node_mask, collapse=True)
        nocorrect_grpah_node = products_V.X[~node_correct][random_index]
        nocorrect_grpah_edge = products_V.E[~node_correct][random_index]
        #(bs,n,) 
        node_labels = torch.zeros_like(correct_mask, dtype = torch.long)
        #(bs,n) pred==ture: 0;  pred!=true: 1; 
        correct_mask = (node_predicted_labels != product_label['X'].squeeze(-1)).bool()
        #(bs,n,)  pred==ture: 0;  pred=0,true=1: 1;  pred=1,true=0: 2 
        node_labels[correct_mask] = node_predicted_labels[correct_mask] + 1
        nocorrect_node_labels = node_labels[~node_correct][random_index]
        
        molecule_list = []
        n_nodes = scatter(torch.ones_like(data.batch), data.batch, reduce='sum')
        n = n_nodes[~node_correct][random_index]

        atom_types = nocorrect_grpah_node[:, :n].squeeze(0).cpu()
        nocorrect_node_labels = nocorrect_node_labels[:, :n].squeeze(0).cpu()
        edge_types = nocorrect_grpah_edge[:, :n, :n].squeeze(0).cpu()
        molecule_list.append([atom_types, edge_types, nocorrect_node_labels])

        visualization_tools.visualize_with_labels("logs/classifier/",
                                                   molecule_list,
                                                   num_molecules_to_visualize= 1,
                                                   prefix=f'pred_',
                                                   bs_num = i,
                                                    )

        #(compactN,)
        #node_comparison_nozero =  node_predicted_labels[new_node_mask] == product_label['X_flat']
        # node_comparison_single = node_predicted_labels == product_label['X'].squeeze(-1)
        # node_correct = torch.all(node_comparison_single, dim=1)
        # edge_comparison = edge_predicted_labels == product_label['E'].squeeze(-1)
        # edge_comparison_flat = edge_comparison.view(edge_comparison.size(0), -1)
        # edge_correct = torch.all(edge_comparison_flat, dim=1)

        # edge_com_nozero = edge_predicted_labels[edge_mask] == product_label['E_flat']

        #
        # graph_correct = node_correct & edge_correct
        # node_correct_all = torch.cat([node_correct_all,node_correct], dim = -1)
        # edge_correct_all = torch.cat([edge_correct_all,edge_correct], dim = -1)
        #graph_correct_all = torch.cat([graph_correct_all,graph_correct], dim = -1)
        # node_correct_nozero = torch.cat([node_correct_nozero,node_comparison_nozero.view(-1)], dim = -1)
        # edge_correct_nozero = torch.cat([edge_correct_nozero,edge_com_nozero.view(-1)], dim = -1)
    
    X_true = X_true.numpy()
    X_pred = X_pred.numpy()
    #X_pred_prob = X_pred_prob.numpy()
    X_cm = confusion_matrix(X_true, X_pred)
    X_tn, X_fp, X_fn, X_tp = X_cm.ravel()
    #X_auc = roc_auc_score(X_true, X_pred_prob)
    print(f"Node: (TP)={X_tp}; (TN)={X_tn}; (FP)={X_fp}; (FN)={X_fn}")
    X_precision, X_recall, X_f1_score, _ = precision_recall_fscore_support(X_true, X_pred, average='binary')
    print(f"Precision: {X_precision:.2f}; Recall: {X_recall:.2f}; F1 Score: {X_f1_score:.2f};")
    X_report = classification_report(X_true, X_pred, digits=2)


    print("\nClassification Report:\n", X_report)

    E_true = E_true.numpy()
    E_pred = E_pred.numpy()
    #E_pred_prob = E_pred_prob.numpy()
    E_cm = confusion_matrix(E_true, E_pred)
    E_tn, E_fp, E_fn, E_tp = E_cm.ravel()
    #E_auc = roc_auc_score(E_true, E_pred_prob)
    print(f"edge: (TP)={E_tp}; (TN)={E_tn}; (FP)={E_fp}; (FN)={E_fn}")
    E_precision, E_recall, E_f1_score, _ = precision_recall_fscore_support(E_true, E_pred, average='binary')
    print(f"Precision: {E_precision:.2f}; Recall: {E_recall:.2f}; F1 Score: {E_f1_score:.2f};")
    E_report = classification_report(E_true, E_pred, digits=2)
    print("\nClassification Report:\n", E_report)

    # print("single_node_accuracy: ",node_correct_nozero.float().mean().item())
    # print("single_edge_accuracy: ",edge_correct_nozero.float().mean().item())
    # print("graph_node_accuracy: ",node_correct_all.float().mean().item())
    # print("graph_edge_accuracy: ",edge_correct_all.float().mean().item())

if __name__ == '__main__':
    disable_rdkit_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=argparse.FileType(mode='r'), required=True)
    parser.add_argument('--checkpoint', action='store', type=str, required=True)
    parser.add_argument('--mode', action='store', type=str, required=True)
    parser.add_argument('--sampling_seed', action='store', type=int, required=False, default=None)
    parser.add_argument('--threshold', action='store', required=False, default=False)

    main(args=parse_yaml_config(parser.parse_args()))
