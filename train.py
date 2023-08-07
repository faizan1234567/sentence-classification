import torch
import wandb
import pandas as pd
import pytorch_lightning as pl
import hydra
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from omegaconf.omegaconf import OmegaConf
from dataset import Dataset
from model import colaModel
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(filename= "logs/logger.log")
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt= "%(asctime)s: %(message)s", datefmt= '%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

class visualizationLogger(pl.Callback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    
    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.dataset.val_dataloader()))
        sentences = val_batch["sentence"]

        outputs = pl_module(val_batch["input_ids"], val_batch["attention_mask"])
        predictions = torch.argmax(outputs, 1)
        labels = val_batch["label"]

        df = pd.DataFrame(
            {"sentence": sentences, "Label": labels.numpy(), "Predictions": predictions.numpy()
            })
        wrong_df = df[df["Label"] != df["Predictions"]]
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )


@hydra.main(config_path = "./configs", config_name = "configs", version_base = None)
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg, resolve = True))
    logger.info(f"Using model: {cfg.model.name}")
    logger.info(f"using the tokenizer: {cfg.model.tokenizer}")
    # instantiate two instance like the dataset and model
    cola_dataset = Dataset(cfg.model.tokenizer, cfg.preprocess.batch, 
                           cfg.preprocess.max_length)
    cola_model = colaModel(cfg.model.name)

    # add model checkpoint and model early stoppoing callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath = "./models", monitor = "valid/loss", mode = "min",
        filename = "best-checkpoint"
    )

    early_stoppoing_callback = EarlyStopping(
        monitor = "valid/loss", patience = 3, verbose = True, mode = "min"
    )

    # adding wandb logger
    wandb_logger = WandbLogger(project = "MLOps - logging and experiment tracking",
                               entity = "engrfaizan-ai")
    

    # now create a trainer object
    trainer = pl.Trainer(
        gpus = (1 if torch.cuda.is_available() else 0),
        logger = wandb_logger,
        max_epochs = cfg.training.max_epochs,
        fast_dev_run = False,
        log_every_n_steps = cfg.training.log_every_n_steps,
        deterministic = cfg.training.deterministic,
        callbacks = [checkpoint_callback, visualizationLogger(cola_dataset), early_stoppoing_callback],
        limit_train_batches = cfg.training.limit_train_batches,
        limit_val_batches = cfg.training.limit_val_batches
    ) 

    # start trainer
    trainer.fit(cola_model, cola_dataset)
    wandb.finish()


if __name__ == "__main__":
    main()