import os
import argparse

#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from datetime import datetime



from src.utils import disable_rdkit_logging, parse_yaml_config, set_deterministic
from src.data.classifier_dataset import RetroBridgeDataModule, RetroBridgeDatasetInfos
from src.features.extra_features import DummyExtraFeatures, ExtraFeatures
from src.features.extra_features_molecular import ExtraMolecularFeatures
from src.frameworks.rerto_classifier import RertoClassifier


from pytorch_lightning import Trainer, callbacks, loggers # type: ignore

from pdb import set_trace


def find_last_checkpoint(checkpoints_dir):
    
    if 'last.ckpt' in os.listdir(checkpoints_dir):
        return os.path.join(checkpoints_dir, 'last.ckpt')

    top_5_checkpoints_dir = os.path.join(checkpoints_dir, 'accuracy')
    epoch2fname = [
        (int(fname.split('_')[0].split('=')[1]), fname)
        for fname in os.listdir(top_5_checkpoints_dir)
        if fname.endswith('.ckpt')
    ]
    latest_fname = max(epoch2fname, key=lambda t: t[0])[1]
    return os.path.join(top_5_checkpoints_dir, latest_fname)


def main(args):
    start_time = datetime.now().strftime('%d_%m_%H_%M_%S')
    run_name = f'{args.experiment_name}_{start_time}'
    experiment = run_name if args.resume is None else args.resume
    print(f'EXPERIMENT: {experiment}')

    data_root = os.path.join(args.data, args.dataset)
    checkpoints_dir = os.path.join(args.checkpoints, experiment)

    os.makedirs(args.logs, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    set_deterministic(args.seed)

    datamodule = RetroBridgeDataModule(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        extra_nodes=args.extra_nodes,
        swap=args.swap,
        evaluation=False,
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
        use_context = False,
        use_positional_encoding = False,
    )

    model = RertoClassifier(
        experiment_name=experiment,
        checkpoints_dir=checkpoints_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_layers=args.n_layers,
        hidden_mlp_dims=args.hidden_mlp_dims,
        hidden_dims=args.hidden_dims,
        lambda_train=args.lambda_train,
        dataset_infos=dataset_infos,
        extra_features=extra_features,
        domain_features=domain_features,
        enc_node_loss=args.enc_node_loss,
        enc_edge_loss=args.enc_edge_loss,
        log_every_steps=args.log_every_steps,
        sample_every_val=args.sample_every_val,
        use_positional_encoding=args.use_positional_encoding,
        pos_enc_dim=args.pos_enc_dim,
        threshold = args.threshold,
        class_weights = args.class_weights,
    )

    checkpoints_dir = os.path.join(checkpoints_dir, 'accuracy')
    os.makedirs(checkpoints_dir, exist_ok=True)
  

    checkpoint_callbacks = [
        callbacks.ModelCheckpoint(
            dirpath=checkpoints_dir,
            filename='{epoch:03d}_{val_accuracy:.3f}',
            save_top_k=5,
            monitor=f'val_accuracy',
            mode='max',
            every_n_epochs=args.sample_every_val,
        ),
    ]

    wandb_logger = None if args.disable_wandb else loggers.WandbLogger(
        save_dir=args.logs,
        project='retrodiff',
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
    parser.add_argument('--disable_wandb', action='store_true', required=False, default=False)
    main(args=parse_yaml_config(parser.parse_args()))
