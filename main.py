import pytorch_lightning as pl
from addict import Dict
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from cli import default_parser
from loggers.wandb import WandbLogger
from loggers.visdom import VisdomLogger


def main():
    experiment, args = create_experiment()
    checkpoints = ModelCheckpoint(
        filepath='/tmp/%s/{epoch:d}_{avg_val_loss:.2f}' % args.experiment,
        monitor='avg_val_loss',
        save_top_k=3,
        mode='min'
    )
    early_stopping = EarlyStopping(
        monitor='avg_val_loss',
        patience=3,
        mode='min'
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        amp_level='O1',
        num_sanity_val_steps=0,
        precision=16 if args.half_precision else 32,
        early_stop_callback=early_stopping,
        checkpoint_callback=checkpoints,
        log_gpu_memory='all',
        # logger=WandbLogger(),
        logger=VisdomLogger(),
        gpus=args.gpus
    )
    net = experiment(args)
    assert trainer.fit(net) == 1, 'Training failed!'


def create_experiment() -> Dict:
    parser = pl.Trainer.add_argparse_args(default_parser())
    config = Dict(vars(parser.parse_args()))
    experiment = config.pop('experiment')
    del config['kwargs']
    return experiment, config


if __name__ == '__main__':
    main()
