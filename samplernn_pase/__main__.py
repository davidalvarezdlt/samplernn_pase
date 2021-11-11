import argparse

import pytorch_lightning as pl

from .data import SampleRNNPASEData
from .model import SampleRNNModel


def main(args):
    data = SampleRNNPASEData(**vars(args))
    data.prepare_data()

    trainer = pl.Trainer.from_argparse_args(
        args, gradient_clip_val=1, gradient_clip_algorithm='value'
    )

    if args.test:
        model = SampleRNNModel.load_from_checkpoint(
            args.test_checkpoint,
            n_speakers=len(data.info['speakers']),
            ling_features_size=data.get_conds_linguistic_size(),
            **vars(args)
        )
        trainer.test(model, data)
    else:
        model = SampleRNNModel(
            n_speakers=len(data.info['speakers']),
            ling_features_size=data.get_conds_linguistic_size(),
            **vars(args)
        )
        trainer.fit(model, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_checkpoint')

    parser = pl.Trainer.add_argparse_args(parser)
    parser = SampleRNNPASEData.add_data_specific_args(parser)
    parser = SampleRNNModel.add_model_specific_args(parser)

    main(parser.parse_args())
