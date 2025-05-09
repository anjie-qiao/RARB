import os
import argparse

from datetime import datetime

from src.utils import disable_rdkit_logging, parse_yaml_config, set_deterministic
from src.data.retrieval_dataset import RetroBridgeDataModule, RetroBridgeDatasetInfos
from src.features.extra_features import DummyExtraFeatures, ExtraFeatures
from src.features.extra_features_molecular import ExtraMolecularFeatures
from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from src.metrics.sampling_metrics import SamplingMolecularMetrics
from src.analysis.visualization import MolecularVisualization
from src.frameworks.markov_bridge import MarkovBridge
from src.frameworks.discrete_diffusion import DiscreteDiffusion
from src.frameworks.one_shot_model import OneShotModel
import torch

from pytorch_lightning import Trainer, callbacks, loggers

from pdb import set_trace


def find_last_checkpoint(checkpoints_dir):
    if 'last.ckpt' in os.listdir(checkpoints_dir):
        return os.path.join(checkpoints_dir, 'last.ckpt')

    top_5_checkpoints_dir = os.path.join(checkpoints_dir, 'top_5_accuracy')
    epoch2fname = [
        (int(fname.split('_')[0].split('=')[1]), fname)
        for fname in os.listdir(top_5_checkpoints_dir)
        if fname.endswith('.ckpt')
    ]
    latest_fname = max(epoch2fname, key=lambda t: t[0])[1]
    return os.path.join(top_5_checkpoints_dir, latest_fname)


def main(args):
    start_time = datetime.now().strftime('%d_%m_%H_%M_%S')
    run_name = f'{args.experiment_name}_k={args.retrieval_k}_{args.retrieval_dataset}_emb{args.embedding}_{start_time}'

    experiment = run_name if args.resume is None else args.resume
    print(f'EXPERIMENT: {experiment}')

    data_root = os.path.join(args.data, args.dataset)
    checkpoints_dir = os.path.join(args.checkpoints, experiment)
    graphs_dir = os.path.join(args.logs, 'graphs', experiment)
    chains_dir = os.path.join(args.logs, 'chains', experiment)

    os.makedirs(args.logs, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(chains_dir, exist_ok=True)

    set_deterministic(args.seed)

    datamodule = RetroBridgeDataModule(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        extra_nodes=args.extra_nodes,
        swap=args.swap,
        evaluation=False,
        retrieval_dataset=args.retrieval_dataset,
        augmented_graphfeature=args.augmented_graphfeature,
        use_cluster=args.use_cluster,
    )
    dataset_infos = RetroBridgeDatasetInfos(datamodule)

    extra_features = (
        ExtraFeatures(args.extra_features, dataset_info=dataset_infos)
        if args.extra_features is not None
        else DummyExtraFeatures()
    )
    domain_features = (
        ExtraMolecularFeatures(dataset_infos=dataset_infos)
        if args.extra_molecular_features
        else DummyExtraFeatures()
    )
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features,
        use_context=args.use_context,
        retrieval_k=args.retrieval_k,

    )
    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    sampling_metrics = SamplingMolecularMetrics(dataset_infos, datamodule.train_smiles)
    if args.visualization:
        visualization_tools = MolecularVisualization(dataset_infos)
    else:
        visualization_tools = None

    
    if args.retrieval_dataset == "50k":
        #(40008,512)
        encoded_reactants = torch.load("data/uspto50k/raw/rxn_encoded_react_tensor.pt")
    elif args.retrieval_dataset == "application":
        #(969283,512) sparse matrix storage
        encoded_reactants = torch.load("data/uspto50k/raw/rxn_encoded_reac_uspto_full.pt")
    else: encoded_reactants = None

    if args.model == 'RetroBridge':
        model = MarkovBridge(
            experiment_name=experiment,
            chains_dir=chains_dir,
            graphs_dir=graphs_dir,
            checkpoints_dir=checkpoints_dir,
            diffusion_steps=args.diffusion_steps,
            diffusion_noise_schedule=args.diffusion_noise_schedule,
            transition=args.transition,
            lr=args.lr,
            weight_decay=args.weight_decay,
            n_layers=args.n_layers,
            hidden_mlp_dims=args.hidden_mlp_dims,
            hidden_dims=args.hidden_dims,
            lambda_train=args.lambda_train,
            dataset_infos=dataset_infos,
            train_metrics=train_metrics,
            sampling_metrics=sampling_metrics,
            visualization_tools=visualization_tools,
            extra_features=extra_features,
            domain_features=domain_features,
            use_context=args.use_context,
            log_every_steps=args.log_every_steps,
            sample_every_val=args.sample_every_val,
            samples_to_generate=args.samples_to_generate,
            samples_to_save=args.samples_to_save,
            samples_per_input=args.samples_per_input,
            chains_to_save=args.chains_to_save,
            number_chain_steps_to_save=args.number_chain_steps_to_save,
            fix_product_nodes=args.fix_product_nodes,
            loss_type=args.loss_type,
            retrieval_k=args.retrieval_k,
            encoded_reactants=encoded_reactants,
            augmented_graphfeature=args.augmented_graphfeature,
        )
    elif args.model == 'DiGress':
        model = DiscreteDiffusion(
            experiment_name=experiment,
            chains_dir=chains_dir,
            graphs_dir=graphs_dir,
            checkpoints_dir=checkpoints_dir,
            diffusion_steps=args.diffusion_steps,
            diffusion_noise_schedule=args.diffusion_noise_schedule,
            transition=args.transition,
            lr=args.lr,
            weight_decay=args.weight_decay,
            n_layers=args.n_layers,
            hidden_mlp_dims=args.hidden_mlp_dims,
            hidden_dims=args.hidden_dims,
            lambda_train=args.lambda_train,
            dataset_infos=dataset_infos,
            train_metrics=train_metrics,
            sampling_metrics=sampling_metrics,
            visualization_tools=visualization_tools,
            extra_features=extra_features,
            domain_features=domain_features,
            log_every_steps=args.log_every_steps,
            sample_every_val=args.sample_every_val,
            samples_to_generate=args.samples_to_generate,
            samples_to_save=args.samples_to_save,
            samples_per_input=args.samples_per_input,
            chains_to_save=args.chains_to_save,
            number_chain_steps_to_save=args.number_chain_steps_to_save,
            fix_product_nodes=args.fix_product_nodes,
            use_context=args.use_context,
        )
    elif args.model == 'OneShot':
        model = OneShotModel(
            experiment_name=experiment,
            chains_dir=chains_dir,
            graphs_dir=graphs_dir,
            checkpoints_dir=checkpoints_dir,
            lr=args.lr,
            weight_decay=args.weight_decay,
            n_layers=args.n_layers,
            hidden_mlp_dims=args.hidden_mlp_dims,
            hidden_dims=args.hidden_dims,
            lambda_train=args.lambda_train,
            dataset_infos=dataset_infos,
            train_metrics=train_metrics,
            sampling_metrics=sampling_metrics,
            visualization_tools=visualization_tools,
            extra_features=extra_features,
            domain_features=domain_features,
            log_every_steps=args.log_every_steps,
            sample_every_val=args.sample_every_val,
            samples_to_generate=args.samples_to_generate,
            samples_to_save=args.samples_to_save,
            samples_per_input=args.samples_per_input,
        )

    top_1_checkpoints_dir = os.path.join(checkpoints_dir, 'top_1_accuracy')
    top_5_checkpoints_dir = os.path.join(checkpoints_dir, 'top_5_accuracy')
    os.makedirs(top_1_checkpoints_dir, exist_ok=True)
    os.makedirs(top_5_checkpoints_dir, exist_ok=True)

    checkpoint_callbacks = [
        callbacks.ModelCheckpoint(
            dirpath=top_1_checkpoints_dir,
            filename='{epoch:03d}_{top_1_accuracy:.3f}',
            save_top_k=5,
            monitor=f'top_1_accuracy',
            mode='max',
            every_n_epochs=args.sample_every_val,
        ),
        callbacks.ModelCheckpoint(
            dirpath=top_5_checkpoints_dir,
            filename='{epoch:03d}_{top_5_accuracy:.3f}',
            save_top_k=5,
            monitor=f'top_5_accuracy',
            mode='max',
            every_n_epochs=args.sample_every_val,
        )
    ]

    wandb_logger = None if args.disable_wandb else loggers.WandbLogger(
        save_dir=args.logs,
        project='RetrievalBridge',
        group=args.dataset,
        name=experiment,
        id=experiment,
        resume='must' if args.resume is not None else 'allow',
        entity=args.wandb_entity,
    )
    trainer = Trainer(
        max_epochs=args.n_epochs,
        logger=wandb_logger,
        callbacks=checkpoint_callbacks,
        accelerator=args.device,
        devices=1,
        num_sanity_val_steps=0,
        enable_progress_bar=args.enable_progress_bar,
        log_every_n_steps=args.log_every_steps,
    )

    if args.resume is None:
        last_checkpoint = None
        print(f'No checkpoint was passed – training from scratch')
    else:
        last_checkpoint = find_last_checkpoint(checkpoints_dir)
        print(f'Training will be resumed from the latest checkpoint {last_checkpoint}')

    print('Start training')
    trainer.fit(model=model,  datamodule=datamodule, ckpt_path=last_checkpoint)


if __name__ == '__main__':
    disable_rdkit_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=argparse.FileType(mode='r'), required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--disable_wandb', action='store_true', required=False, default=False)
    main(args=parse_yaml_config(parser.parse_args()))
