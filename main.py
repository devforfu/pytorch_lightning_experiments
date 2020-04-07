import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from cli import default_parser


def main():
    args = pl.Trainer.add_argparse_args(default_parser()).parse_args()
    checkpoints = ModelCheckpoint(
        filepath='/tmp/%s/{epoch:d}_{val_loss:.2f}' % args.experiment,
        verbose=True,
        save_top_k=3,
        mode='min'
    )
    early_stopping = EarlyStopping(patience=3)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        amp_level='O1',
        precision=16 if args.half_precision else 32,
        early_stop_callback=early_stopping,
        checkpoint_callback=checkpoints,
        log_gpu_memory='all',
        gpus=args.gpus
    )
    net = args.experiment(args)
    assert trainer.fit(net) == 1, 'Training failed!'


if __name__ == '__main__':
    main()
