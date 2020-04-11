import pytorch_lightning as pl
import scipy.special
from addict import Dict
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from cli import default_parser
from loggers.visdom import VisdomLogger


def main():
    experiment, args = create_experiment()
    checkpoints = ModelCheckpoint(
        filepath='/tmp/%s/{epoch:d}_{avg_valid_loss:.2f}' % args.experiment,
        monitor='avg_valid_loss',
        save_top_k=3,
        mode='min'
    )
    early_stopping = EarlyStopping(
        monitor='avg_valid_loss',
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
        logger=VisdomLogger(),
        gpus=args.gpus
    )
    net = experiment(args, metrics=[recall, accuracy])
    assert trainer.fit(net) == 1, 'Training failed!'


def create_experiment() -> Dict:
    parser = pl.Trainer.add_argparse_args(default_parser())
    config = Dict(vars(parser.parse_args()))
    experiment = config.pop('experiment')
    return experiment, config


def softmax_logits(f):
    from functools import wraps

    @wraps(f)
    def wrapped(logits, targets):
        predictions = scipy.special.softmax(logits, axis=1).argmax(axis=1)
        return f(targets, predictions)

    return wrapped


@softmax_logits
def recall(predictions, targets):
    from sklearn.metrics import recall_score
    score = recall_score(targets, predictions, average='macro')
    return score


@softmax_logits
def accuracy(predictions, targets):
    from sklearn.metrics import accuracy_score
    score = accuracy_score(targets, predictions)
    return score


if __name__ == '__main__':
    main()
