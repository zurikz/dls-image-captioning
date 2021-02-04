import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from image_captioning.utils import pad_collate_fn

import time
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import clear_output
matplotlib.rcParams.update({'figure.figsize': (12, 7),
                            'font.size': 11})


class Trainer:
    def __init__(self, model, datasets: dict, criterion,
                 optimizer, scheduler, clip=5,
                 checkpoints_folder='../checkpoints/'):
        self.checkpoints_folder = checkpoints_folder
        self.model = model
        self.train = datasets['train']
        self.val = datasets['val']
        self.test = datasets['test']
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip = clip
        self.start_epoch = 1
        self.train_history = []
        self.val_history = []
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    def load_checkpoint(self, path_to_checkpoint):
        checkpoint = torch.load(path_to_checkpoint)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        self.start_epoch = checkpoint['epoch'] + 1

    def save_checkpoint(self, epoch, path_to_checkpoint):
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history
        }, path_to_checkpoint)

    def train(self, batch_size, epochs, path_to_checkpoint=None):
        """
        Train loop. Provide path to checkpoint for training from checkpoint.        
        """
        if path_to_checkpoint is not None:
            self.load_checkpoint(path_to_checkpoint)

        train_loader = DataLoader(self.train, batch_size, num_workers=4,
                                  collate_fn=pad_collate_fn)

        val_loader = DataLoader(self.val, batch_size, num_workers=4,
                                collate_fn=pad_collate_fn)

        best_val_metrics = {'loss': float('inf'), 
                            'cider': float('-inf'),
                            'bleu4': float('-inf')}

        for epoch in range(self.start_epoch, epochs):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics['loss']
            self.scheduler.step(val_loss)

            end_time = time.time()
            epoch_mins = int((end_time - start_time) / 60)

            self.train_history.append(train_loss)
            self.val_history.append(val_loss)
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\t  Val Loss: {val_loss:.3f}')

            for metric in ['cider', 'bleu-4']:
                if val_metrics[metric] > best_val_metrics[metric]:
                    best_val_metrics[metric] = val_metrics[metric]
                    filename = f'BEST_{metric}_{val_metrics[metric]:.3f}.pth.tar'
                    self.save_checkpoint(epoch, self.checkpoints_folder + filename)

            if val_loss < best_val_metrics['loss']:
                best_val_metrics['loss'] = val_loss
                filename = f'BEST_loss_{val_loss:.3f}.pth.tar'
                self.save_checkpoint(epoch, self.checkpoints_folder + filename)

    def train_epoch(self, train_loader):
        self.model.train()

        current_epoch_history = []
        for i, (imgs, captions, lenghts) in enumerate(train_loader):
            imgs = imgs.to(self.device)
            captions = captions.to(self.device)

            logits = self.model(imgs, captions, lenghts)
            predictions = logits[:, :-1, :]
            targets = captions[:, 1:]
            loss = self.criterion(predictions.permute(0, 2, 1), targets)
            current_epoch_history.append(loss.cpu().data.numpy())

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

            if (i + 1) % 10 == 0:
                self.plot_history(current_epoch_history)

        return sum(current_epoch_history) / len(current_epoch_history)

    def validate(self, val_loader):
        self.model.eval()

        references = [] # true captions for calculating CIDEr and BLEU-4
        hypotheses = [] # predictions
        val_metrics = {'loss': [], 'cider': [], 'bleu-4': []}

        with torch.no_grad():
            for i, (imgs, captions, lenghts) in enumerate(val_loader):
                imgs = imgs.to(self.device)
                captions = captions.to(self.device)

                logits = self.model(imgs, captions, lenghts)
                predictions = logits[:, :-1, :]
                targets = captions[:, 1:]
                loss = self.criterion(predictions.permute(0, 2, 1), targets)
                val_metrics['loss'].append(loss.cpu().data.numpy())
                val_metrics['cider'].append(cider_score(imgs, predictions, targets))
                val_metrics['bleu-4'].append(bleu_score(imgs, predictions, targets))

        for metric, scores in val_metrics:
            val_metrics[metric] = sum(scores) / len(val_loader)

        return val_metrics

    def plot_history(self, current_epoch_history):
        """
        Plots two types of plots:
            1) Left is a loss history of the current epoch.
            2) Right is a loss history of the entire training process.
        """
        _, ax = plt.subplots(nrows=1, ncols=2)
        clear_output(True)
        ax[0].plot(current_epoch_history, label='Epoch train loss')
        ax[0].set_xlabel('Batch')
        ax[0].set_title('Epoch train loss')
        ax[1].plot(self.train_history, label='Train loss')
        ax[1].plot(self.valid_history, label='Valid loss')
        ax[1].set_xlabel('Epoch')
        ax[1].set_title('Entire history')
        plt.legend()
        plt.show()
