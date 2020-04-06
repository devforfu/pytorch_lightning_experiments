import pytorch_lightning as pl

import mnist_experiments
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from cli import default_parser


def main():
    args = pl.Trainer.add_argparse_args(default_parser()).parse_args()
    exp = load_experiment(args.experiment)
    checkpoints = ModelCheckpoint(filepath='/tmp/mnist/{epoch:d}_{val_loss:.2f}',
                                  verbose=True,
                                  save_top_k=3,
                                  mode='min')
    early_stopping = EarlyStopping(patience=3)
    trainer = pl.Trainer(max_epochs=args.epochs,
                         amp_level='O1',
                         precision=16 if args.half_precision else 32,
                         early_stop_callback=early_stopping,
                         checkpoint_callback=checkpoints,
                         log_gpu_memory='all',
                         gpus=args.gpus)
    net = exp.create_module(args)
    assert trainer.fit(net) == 1, 'Training failed!'


def load_experiment(name: str):
    if not hasattr(mnist_experiments, name):
        raise ValueError(f'unknown experiment: {name}')
    return getattr(mnist_experiments, name)


if __name__ == '__main__':
    main()
