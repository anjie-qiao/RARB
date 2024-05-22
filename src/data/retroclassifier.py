from multiprocessing import dummy
from syslog import LOG_SYSLOG
from telnetlib import TM

from networkx import null_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl  # type: ignore
import os
import time

from src.data import utils
from src.metrics.train_metrics import TrainLossClassifier
from src.metrics.sampling_metrics import compute_retrosynthesis_metrics
from src.models.transformer_model import GraphTransformer
from src.models.classifier import Classifier
from src.data.utils import decode_no_edge

from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from pdb import set_trace


class RertoClassifier(pl.LightningModule):
    def __init__(
            self,
            experiment_name,
            checkpoints_dir,
            lr,
            weight_decay,
            n_layers,
            hidden_mlp_dims,
            hidden_dims,
            lambda_train,
            dataset_infos,
            extra_features,
            domain_features,
            enc_node_loss,
            enc_edge_loss,
            log_every_steps,
            sample_every_val,
    ):

        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims

        self.sample_every_val = sample_every_val
        self.name = experiment_name
        self.checkpoints_dir = checkpoints_dir

        self.model_dtype = torch.float32

        self.lr = lr
        self.weight_decay = weight_decay

        self.enc_node_loss = enc_node_loss
        self.enc_edge_loss = enc_edge_loss

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']

        self.dataset_info = dataset_infos
        self.train_loss = TrainLossClassifier(lambda_train)
        self.val_loss = TrainLossClassifier(lambda_train)
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.encoder_model = GraphTransformer(
            n_layers=n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU()
        )

        self.classifier = Classifier(
            dx=output_dims['X'],
            de=output_dims['E']
        )

        self.save_hyperparameters(ignore=[dataset_infos])

        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None

        self.log_every_steps = log_every_steps

        self.val_counter = 0

    def configure_optimizers(self):
        params = list(self.encoder_model.parameters()) + list(self.classifier.parameters())
        return torch.optim.AdamW(
            params=params,
            # params=self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            amsgrad=True,
        )

    def training_step(self, data, i):
        product, node_mask, pred_label, True_lable, product_edge_indices, dummy_node_mask = self.process_and_forward(
            data)

        return self.compute_training_loss(True_lable, pred_label, i, self.enc_node_loss, self.enc_edge_loss)

    def compute_training_loss(self, True_lable, pred_label, i, enc_node_loss, enc_edge_loss):
        loss = self.train_loss(True_lable, pred_label, enc_node_loss, enc_edge_loss)
        if i % self.log_every_steps == 0:
            self.log(f'train_loss/batch_CE', loss.detach())
        return {'loss': loss}

    def compute_validation_loss(self, True_lable, pred_label, i, enc_node_loss, enc_edge_loss):
        loss = self.val_loss(True_lable, pred_label, enc_node_loss, enc_edge_loss)
        if i % self.log_every_steps == 0:
            self.log(f'val_loss/batch_CE', loss.detach())
        return {'loss': loss}

    def validation_step(self, data, i):
        product, node_mask, pred_label, True_lable, product_edge_indices, dummy_node_mask = self.process_and_forward(
            data)
        return self.compute_validation_loss(True_lable, pred_label, i, self.enc_node_loss, self.enc_edge_loss)

    def on_validation_epoch_end(self):
        self.val_counter += 1
        if self.val_counter % self.sample_every_val == 0:
            self.val_accuracy()
            self.trainer.save_checkpoint(os.path.join(self.checkpoints_dir, 'last.ckpt'))

    @torch.no_grad()
    def val_accuracy(self):

        print(f'Sampling epoch={self.current_epoch}')

        dataloader = self.trainer.datamodule.val_dataloader()
        total_batch = len(dataloader.dataset) // dataloader.batch_size

        graph_correct_all = torch.tensor([]).to(self.device) # 整图正确率
        node_correct_all = torch.tensor([]).to(self.device)  # 整图正确率
        edge_correct_all = torch.tensor([]).to(self.device)  # 整图正确率
        node_correct_nozero = torch.tensor([]).to(self.device)  # 整图正确率
        edge_correct_nozero = torch.tensor([]).to(self.device)  # 整图正确率

        if len(dataloader.dataset) % dataloader.batch_size != 0:
            total_batch += 1
        for data in tqdm(dataloader, total=total_batch):
            data = data.to(self.device)

            product, node_mask, pred_label, True_lable, product_edge_indices, dummy_node_mask = self.process_and_forward(
                data)

            # 将node的预测标签还原回bs,n
            node_predicted_labels = torch.argmax(pred_label['X'], dim=-1)
            node_labels_expanded = torch.full_like(True_lable['X'][node_mask], fill_value=0, dtype=torch.long)
            node_labels_expanded[dummy_node_mask] = node_predicted_labels
            restored_node_labels = torch.zeros_like(True_lable['X'], dtype=torch.long)
            restored_node_labels[node_mask] = node_labels_expanded

            #将edge的预测标签还原回bs,n,n
            edge_predicted_labels = torch.argmax(pred_label['E'], dim=-1)
            edge_labels = torch.full(product.E.shape[:-1], fill_value=0, dtype=torch.long,device=self.device)
            start_idx = 0
            for idx, (batch_idx, row_indices, col_indices) in enumerate(product_edge_indices):
                end_idx = start_idx + row_indices.size(0)
                edge_labels[batch_idx, row_indices, col_indices] = edge_predicted_labels[start_idx:end_idx]
                edge_labels[batch_idx, col_indices, row_indices] = edge_predicted_labels[start_idx:end_idx]  # 确保对称性
                start_idx = end_idx

            node_comparison_nozero = node_predicted_labels == True_lable['X_flat'].squeeze(-1) #非填充节点正确率
            edge_com_nozero = edge_predicted_labels == True_lable['E_flat'].squeeze(-1) #非0边正确率

            node_comparison_all = restored_node_labels == True_lable['X'].squeeze(-1) #bs,n,图全部节点正确率
            node_correct = torch.all(node_comparison_all, dim=1)
            edge_comparison = edge_labels == True_lable['E'].squeeze(-1)  #bs,n,n 图全部边正确率

            edge_comparison_all = edge_comparison.view(edge_comparison.size(0), -1)
            edge_correct = torch.all(edge_comparison_all, dim=1)
            graph_correct = node_correct & edge_correct #整图正确率

            node_correct_all = torch.cat([node_correct_all, node_correct], dim=-1)
            edge_correct_all = torch.cat([edge_correct_all, edge_correct], dim=-1)
            graph_correct_all = torch.cat([graph_correct_all, graph_correct], dim=-1)
            node_correct_nozero = torch.cat([node_correct_nozero, node_comparison_nozero.view(-1)], dim=-1)
            edge_correct_nozero = torch.cat([edge_correct_nozero, edge_com_nozero.view(-1)], dim=-1)

        accuracy = graph_correct_all.float().mean().item()
        accuracy_node = node_correct_all.float().mean().item()
        accuracy_edge = edge_correct_all.float().mean().item()
        accuracy_node_single = node_correct_nozero.float().mean().item()
        accuracy_edge__single = edge_correct_nozero.float().mean().item()

        self.log(f'val_accuracy', accuracy)
        self.log(f'val_accuracy/node', accuracy_node)
        self.log(f'val_accuracy/edge', accuracy_edge)
        self.log(f'val_accuracy/single_node', accuracy_node_single)
        self.log(f'val_accuracy/single_edge', accuracy_edge__single)

    def process_and_forward(self, data):
        product, p_node_mask = utils.to_dense(data.p_x, data.p_edge_index, data.p_edge_attr, data.batch)
        product = product.mask(p_node_mask)

        context, c_node_mask = utils.to_dense(data.context_x, data.context_edge_index, data.context_edge_attr,
                                              data.batch)
        context = context.mask(c_node_mask)
        True_lable = {
            'X': context.X[:, :, -1],
            'E': context.E[:, :, :, -1],
        }
        node_mask = p_node_mask
        dummy_node_mask = product.X[node_mask][:, -1] != 1   #dummy node one-hot = [0,0,...,0,1]

        True_lable['X_flat'] = True_lable['X'][node_mask]    # X.shape : bs,n
        True_lable['X_flat'] = True_lable['X_flat'][dummy_node_mask]  # X_flat.shape : m

        triu_indices = torch.triu_indices(product.X.size(1), product.X.size(1), offset=1).to(self.device)
        product_edge_indices = [] #用来保存E中product边的索引
        new_E = decode_no_edge(product.E)    #Markovbridge将edge_attr=[0,0,0,0,0] -> [1,0,0,0,0],现在转换回来
        for i in range(new_E.size(0)):
            adj_matrix = new_E[i, :, :, :]
            edge_features = adj_matrix[triu_indices[0], triu_indices[1], :]
            nonzero_mask = edge_features.any(dim=-1)    # 只保留非零边
            nonzero_indices = nonzero_mask.nonzero(as_tuple=False).view(-1)  # 记录非零边的索引
            product_edge_indices.append((i, triu_indices[0][nonzero_indices], triu_indices[1][nonzero_indices]))

        edge_label_list = []
        for batch_idx, row_indices, col_indices in product_edge_indices:
            label = True_lable['E'][batch_idx, row_indices, col_indices].squeeze()
            edge_label_list.append(label)
        True_lable['E_flat'] = torch.cat(edge_label_list)   # E shape: bs,n,n; E_flat: m

        product_data = {'X_t': product.X, 'E_t': product.E, 'y_t': product.y, 't': None, 'node_mask': node_mask}
        product_extra_data = self.compute_extra_data(product_data)
        pred_label = self.forward(product, product_extra_data, node_mask, product_edge_indices, dummy_node_mask)
        return product, node_mask, pred_label, True_lable, product_edge_indices, dummy_node_mask

    def forward(self, product, product_extra_data, node_mask, product_edge_indices, dummy_node_mask):
        p_X = torch.cat((product.X, product_extra_data.X), dim=2).float()
        p_E = torch.cat((product.E, product_extra_data.E), dim=3).float()
        p_y = torch.hstack((product.y, product_extra_data.y)).float()
        enc_input = self.encoder_model(p_X, p_E, p_y, node_mask)

        Y_classifier_list = []
        for batch_idx, row_indices, col_indices in product_edge_indices:  # 从E中提取product的边
            features = enc_input.E[batch_idx, row_indices, col_indices, :]
            Y_classifier_list.append(features)
        Y_classifier = torch.cat(Y_classifier_list)

        X_calssifier = enc_input.X[node_mask]
        X_calssifier = X_calssifier[dummy_node_mask]

        pred_label = self.classifier(X_calssifier, Y_classifier)

        return pred_label

    def compute_extra_data(self, noisy_data, context=None):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        if context is not None:
            extra_X = torch.cat((extra_X, context.X), dim=-1)
            extra_E = torch.cat((extra_E, context.E), dim=-1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)
