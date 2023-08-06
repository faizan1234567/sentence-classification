import torch
import wandb
import pandas as pd
import pytorch_lightning as pl
import hydra
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from dataset import Dataset
from model import colaModel

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



def main():
    # instantiate two instance like the dataset and model
    cola_dataset = Dataset()
    cola_model = colaModel()

    # add model checkpoint and model early stoppoing callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath = "./models", monitor = "val_loss", mode = "min"
    )

    early_stoppoing_callback = EarlyStopping(
        monitor = "val_loss", patience = 3, verbose = True, mode = "min"
    )

    # adding wandb logger
    wandb_logger = WandbLogger(project = "MLOps - logging and experiment tracking",
                               entity = "engrfaizan-ai")
    

    # now create a trainer object
    trainer = pl.Trainer(
        default_root_dir = "logs",
        gpus = (1 if torch.cuda.is_available() else 0),
        logger = wandb_logger,
        max_epochs = 5,
        fast_dev_run = False,
        log_every_n_steps = 10,
        deterministic = True,
        callbacks = [checkpoint_callback, visualizationLogger(cola_dataset), early_stoppoing_callback]
    ) 

    # start trainer
    trainer.fit(cola_model, cola_dataset)


if __name__ == "__main__":
    main()