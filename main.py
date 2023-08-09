import os
import pytorch_lightning as pl
import torch
import warnings
import argparse
import yaml
from train.train_utils import configure_experiment, load_model
from dataset.taskonomy_constants import SEMSEG_CLASSES
from dataset.dataloader_factory import generate_support_data


def run(config):
    # set monitor name and postfix
    if config.stage == 0:
        config.monitor = 'summary/mtrain_valid_pred'
    else:
        if config.task == 'segment_semantic':
            config.monitor = f'mtest_support/segment_semantic_{config.channel_idx}_pred'
            if config.save_postfix == '':
                config.save_postfix = f'_task:segment_semantic_{config.channel_idx}'
        else:
            config.monitor = f'mtest_support/{config.task}_pred'
            if config.save_postfix == '':
                config.save_postfix = f'_task:{config.task}{config.save_postfix}'

    # environmental settings
    logger, log_dir, callbacks, precision, strategy, plugins = configure_experiment(config, model)
    if config.stage == 2:
        model.config.result_dir = log_dir

    # create pytorch lightning trainer.
    trainer = pl.Trainer(
        logger=logger,
        default_root_dir=log_dir,
        accelerator='gpu',
        max_epochs=((config.n_steps // config.val_iter) if (not config.no_eval) and config.stage <= 1 else 1),
        log_every_n_steps=-1,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        benchmark=True,
        devices=-1,
        strategy=strategy,
        precision=precision,
        plugins=plugins,
    )

    # validation at start
    if config.stage == 1:
        trainer.validate(model, verbose=False)
    # start training or fine-tuning
    elif config.stage <= 1:
        trainer.fit(model)
    # start evaluation
    else:
        trainer.test(model)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=pl.utilities.warnings.PossibleUserWarning)
    IS_RANK_ZERO = int(os.environ.get('LOCAL_RANK', 0)) == 0

    from args import config  # parse arguments

    # Call the generate_support_data function directly for segmentation task
    if config.stage >= 1 and config.task == 'segment_semantic' and config.channel_idx < 0:
        save_postfix = config.save_postfix
        for channel_idx in SEMSEG_CLASSES:
            config.save_postfix = save_postfix
            config.channel_idx = channel_idx

            support_data = generate_support_data(config, data_path="path_to_save_support_data.pt", split="train",
                                                 verbose=True)

            # Load the model and get the checkpoint path
            model, load_path = load_model(config)

            # Set other environmental settings
            logger, log_dir, callbacks, precision, strategy, plugins = configure_experiment(config, model)
            if config.stage == 2:
                model.config.result_dir = log_dir

            # Create pytorch lightning trainer.
            trainer = pl.Trainer(
                logger=logger,
                default_root_dir=log_dir,
                accelerator='gpu',
                max_epochs=((config.n_steps // config.val_iter) if (not config.no_eval) and config.stage <= 1 else 1),
                log_every_n_steps=-1,
                num_sanity_val_steps=0,
                callbacks=callbacks,
                benchmark=True,
                devices=-1,
                strategy=strategy,
                precision=precision,
                plugins=plugins,
            )

            # Validation at start
            if config.stage == 1:
                trainer.validate(model, verbose=False)
            # Start training or fine-tuning
            elif config.stage <= 1:
                trainer.fit(model)
            # Start evaluation
            else:
                trainer.test(model)
    else:
        # Call the generate_support_data function directly for other tasks
        support_data = generate_support_data(config, data_path="path_to_save_support_data.pt", split="train",
                                             verbose=True)

        # Load the model and get the checkpoint path
        model, load_path = load_model(config)

        # Set other environmental settings
        logger, log_dir, callbacks, precision, strategy, plugins = configure_experiment(config, model)
        if config.stage == 2:
            model.config.result_dir = log_dir

        # Create pytorch lightning trainer.
        trainer = pl.Trainer(
            logger=logger,
            default_root_dir=log_dir,
            accelerator='gpu',
            max_epochs=((config.n_steps // config.val_iter) if (not config.no_eval) and config.stage <= 1 else 1),
            log_every_n_steps=-1,
            num_sanity_val_steps=0,
            callbacks=callbacks,
            benchmark=True,
            devices=-1,
            strategy=strategy,
            precision=precision,
            plugins=plugins,
        )

        # Validation at start
        if config.stage == 1:
            trainer.validate(model, verbose=False)
        # Start training or fine-tuning
        elif config.stage <= 1:
            trainer.fit(model)
        # Start evaluation
        else:
            trainer.test(model)
            
def parse_arguments():
    parser = argparse.ArgumentParser(description='Your script description here.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()
    return args

def main():

    run(config)

if __name__ == "__main__":
    main()
