import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
from model import SpeechRecognition
from dataset import Data, collate_fn_pad


class SpeechModule(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.criterion = nn.CTCLoss(blank=28,zero_infinity=True)
        self.val_outputs = []  # To store validation outputs

    def forward(self, x, hidden):
        out = self.model(x,hidden)
        return out

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.model.parameters()) 
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,mode='min',factor=0.5,patience=6
        )
        
        # return [self.optimizer], [self.scheduler]
        return {
            "optimizer" : self.optimizer,
            "lr_scheduler" : {
                "scheduler" : self.scheduler,
                "monitor" : "val_loss"
            }
        }
    
    def step(self,batch):
        spectrograms, labels, spec_lengths, label_lengths = batch
        batch_size = spectrograms.shape[0]
        hn, cn = self.model._init_hidden(batch_size=batch_size)
        output, _ = self.model(spectrograms, (hn,cn))
        output = F.softmax(output,dim=2)

        loss = self.criterion(output,labels,spec_lengths,label_lengths)
        return loss

    def training_step(self,batch):
        loss = self.step(batch)
        # logs = {"loss" : loss, "lr" : self.optimizer.param_groups[0]['lr']}
        self.log("train_loss", loss, on_step=False, on_epoch=True,prog_bar=True,logger=True)
        self.log("lr", self.optimizer.param_groups[0]['lr'],prog_bar=False,logger=True)
        return loss

    def train_dataloader(self):
        data_parameters = Data.parameter
        #can update data_parameters by overriding them in argument parser
        train_dataset = Data(json_path=self.args.train_file,**data_parameters)
        train_loader = DataLoader(dataset=train_dataset,batch_size=self.args.batch_size,collate_fn=collate_fn_pad)
        return train_loader
    

    def validation_step(self, batch):
        val_loss = self.step(batch=batch)
        self.val_outputs.append(val_loss)
        self.log("val_loss_step",val_loss,on_step=True,on_epoch=False,logger=False,prog_bar=False)
        # return {"val_loss" : val_loss}
    
    def on_validation_epoch_start(self):
        self.val_outputs = [] # reseting the validation outputs at the start of epoch
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_outputs).mean()
        self.scheduler.step(avg_loss)
        # tensorboard_logs = {'val_loss' : avg_loss}
        # return {'val_loss' : avg_loss, 'log' : tensorboard_logs}
        self.log("val_loss", avg_loss,on_epoch=True, on_step=False,prog_bar=True,logger=True)
    
    def val_dataloader(self):
        data_parameters = Data.parameter
        #can update data_parameters by overriding them in argument parser
        validation_dataset = Data(json_path=self.args.test_file,**data_parameters)
        validation_loader = DataLoader(dataset=validation_dataset,batch_size=self.args.batch_size,collate_fn=collate_fn_pad)
        return validation_loader
        

def checkpoint_callback(args):
    return ModelCheckpoint(
            dirpath=args.save_model_path,
            save_top_k=True,
            verbose=True,
            monitor='val_loss',
            mode='min'
        )


def main(args):
    hyper_parameters = SpeechRecognition.hyper_parameters 
    # hyper_parameters.update(args.hparams_override)   can update the hyper_parameters if required
    model = SpeechRecognition(**hyper_parameters)

    if args.load_model_from:
        speech_module = SpeechModule.load_from_checkpoint(checkpoint_path=args.load_model_from,model = model,args=args)
    else:
        speech_module = SpeechModule(model=model,args=args)

    logger = TensorBoardLogger(save_dir=args.logdir, name='speech_recognition')

    trainer = pl.Trainer(
        max_epochs=100, logger=logger, 
        gradient_clip_val=1.0, callbacks=checkpoint_callback(args=args),
        enable_checkpointing= True
        )
    
    trainer.fit(model=speech_module)



if __name__=="__main__":
    parser = ArgumentParser()

    parser.add_argument("--train_file", default=None, required=True, type=str,help="Path to your train file")
    parser.add_argument("--test_file", default=None, required=True,type=str, help="path to your test file")
    
    parser.add_argument("--batch_size", default=32, required=False,type=int, help="Batch size for training")
    parser.add_argument("--save_model_path", default=r"saved_model/",required=False,type=str,help="path to save the models")
    parser.add_argument("--load_model_from", default=None,required=False,type=str,help="path to save the models")
    parser.add_argument("--logdir", default=r"TensorBoardLogs/",required=False,type=str,help="path to save logs")

    args = parser.parse_args()
    
    main(args)





