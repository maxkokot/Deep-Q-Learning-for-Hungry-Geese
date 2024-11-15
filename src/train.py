import click
import logging

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from models.model import DQNLightning, save_model
from utils import load_config, add_n_dims


def train(model_config_path, env_config_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = load_config(model_config_path)
    env_config = load_config(env_config_path)

    warm_start_size = model_config['warm_start_size']
    replay_size = model_config['replay_size']
    batch_size = model_config['batch_size']
    plc_sync_rate = model_config['plc_sync_rate']
    tgt_sync_rate = model_config['tgt_sync_rate']
    num_epochs = model_config['num_epochs']
    model_dir = model_config['model_dir']
    lr = model_config['lr']
    random_state = model_config['random_state']

    env_config = add_n_dims(env_config)

    pl.seed_everything(random_state)
    tb_logger = TensorBoardLogger("tb_logs", name="conv_nn")
    model = DQNLightning(batch_size=batch_size,
                         lr=lr,
                         replay_size=replay_size,
                         warm_start_size=warm_start_size,
                         tgt_sync_rate=tgt_sync_rate,
                         plc_sync_rate=plc_sync_rate,
                         device=device,
                         config=env_config)

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=num_epochs,
        logger=tb_logger,
    )

    trainer.fit(model)
    save_model(trainer, model_dir)


@click.command(name="train")
@click.option('--model_config_path', default='../config/model_config.yaml')
@click.option('--env_config_path', default='../config/env_config.yaml')
def train_command(model_config_path, env_config_path):
    logger = logging.getLogger(__name__)
    logger.info('Training DQN')
    train(model_config_path, env_config_path)
    logger.info('Model has been fitted and saved')


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    train_command()
