import pytorch_lightning as pl
from addict import Dict
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from cli import default_parser
from logger import VisdomLogger


def main():
    args = pl.Trainer.add_argparse_args(default_parser()).parse_args()
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
        logger=VisdomLogger(delete_env_on_start=True),
        gpus=args.gpus
    )
    net = args.experiment(args)
    assert trainer.fit(net) == 1, 'Training failed!'


if __name__ == '__main__':
    main()
