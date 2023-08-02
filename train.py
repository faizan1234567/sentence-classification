import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from dataset import Dataset
from model import colaModel


def main():
    # instantiate two instance like the dataset and model
    cola_dataset = Dataset()
    cola_model = colaModel()

    # add model checkpoint and model early stoppoing callbacks
    checkpoint_callback = ModelCheckpoint(
        dir_path = "./models", monitor = "val_loss", mode = "min"
    )

    early_stoppoing_callback = EarlyStopping(
        monitor = "val_loss", patience = 3, verbose = True, mode = "min"
    )

    # now create a trainer object
    trainer = pl.Trainer(
        default_root_dir = "logs",
        gpus = (1 if torch.cuda.is_available() else 0),
        max_epochs = 5,
        fast_dev_run = False,
        logger = pl.loggers.TensorBoardLogger("logs/", name = "cola", version = 1)
        callbacks = [checkpoint_callback, early_stoppoing_callback]
    ) 

    # start trainer
    trainer.fit(cola_model, cola_dataset)


if __name__ == "__main__":
    main()