import torch
import pytorch_lightning as pl
from model import NN
from dataset import MNISTDataModule
import config
from pytorch_lightning.callbacks import EarlyStopping


if __name__ == '__main__':
  model = NN(input_size=config.input_size, learning_rate=config.learning_rate, num_classes=config.num_classes)
  dm = MNISTDataModule(data_dir=config.data_dir, batch_size=config.batch_size, num_workers=config.num_workers)
  trainer = pl.Trainer(accelerator=config.accelerator, devices=config.devices, min_epochs=config.min_epochs,
    max_epochs=config.max_epochs, precision=config.precision, callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=5)])
  trainer.fit(model, dm)
  trainer.validate(model, dm)
  trainer.test(model, dm)