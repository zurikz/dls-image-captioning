import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from image_captioning.utils import pad_collate_fn
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu

import time
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import clear_output
matplotlib.rcParams.update({'figure.figsize': (12, 12),
                            'font.size': 11})


cider = Cider(df='corpus')
bleu = Bleu(n=4)
torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(self, model, datasets: dict, vocab,
                 criterion, optimizer, scheduler, device,
                 clip=5, teacher_forcing_ratio=0.5,
                 checkpoints_folder='../checkpoints/'):
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.checkpoints_folder = checkpoints_folder
        self.model = model
        self.train = datasets['train']
        self.val = datasets['val']
        self.test = datasets['test']
        self.vocab = vocab
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip = clip
        self.start_epoch = 1
        self.train_history = []
        self.val_history = []
        self.cider_history = []
        self.bleu4_history = []
        self.device = device

    def load_checkpoint(self, path_to_checkpoint):
        checkpoint = torch.load(path_to_checkpoint)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        self.cider_history = checkpoint['cider_history']
        self.bleu4_history = checkpoint['bleu4_history']
        self.start_epoch = checkpoint['epoch'] + 1

    def save_checkpoint(self, epoch, path_to_checkpoint):
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history,
            'cider_history': self.cider_history,
            'bleu4_history': self.bleu4_history
        }, path_to_checkpoint)

    def fit(self, batch_size, epochs, path_to_checkpoint=None):
        """
        Train loop. Provide path to checkpoint for training from checkpoint.        
        """
        self.batch_size = batch_size
        if path_to_checkpoint is not None:
            self.load_checkpoint(path_to_checkpoint)

        train_loader = DataLoader(self.train, batch_size, num_workers=4,
                                  collate_fn=pad_collate_fn, pin_memory=True)

        val_loader = DataLoader(self.val, batch_size, num_workers=4, 
                                collate_fn=pad_collate_fn,
                                pin_memory=True, drop_last=True)

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
            self.cider_history.append(val_metrics['cider'])
            self.bleu4_history.append(val_metrics['bleu4'])
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\t  Val Loss: {val_loss:.3f}')

            for metric in ['cider', 'bleu4']:
                if val_metrics[metric] > best_val_metrics[metric]:
                    best_val_metrics[metric] = val_metrics[metric]
                    filename = f'BEST_{metric}.pth.tar'
                    self.save_checkpoint(epoch, self.checkpoints_folder + filename)

            if val_loss < best_val_metrics['loss']:
                best_val_metrics['loss'] = val_loss
                filename = 'BEST_loss.pth.tar'
                self.save_checkpoint(epoch, self.checkpoints_folder + filename)

    def train_epoch(self, train_loader):
        self.model.train()

        current_epoch_history = []
        for i, (imgs, captions, lenghts) in enumerate(train_loader):
            imgs = imgs.to(self.device)
            captions = captions.to(self.device)

            lenghts, sorted_idx = lenghts.sort(descending=True)
            imgs = imgs[sorted_idx]
            captions = captions[sorted_idx]

            # No need to decode at <eos>
            decode_lenghts = lenghts - 1
            teacher_forcing = True if random.random() < self.teacher_forcing_ratio \
                                   else False

            loss = 0
            hidden = imgs
            if teacher_forcing:
                for step in range(max(decode_lenghts)):
                    batch_size = sum([lenght > step for lenght in decode_lenghts])
                    decoder_input = captions[:batch_size, step]
                    targets = captions[:batch_size, step + 1]
                    logits, hidden = self.model(
                        imgs[:batch_size], decoder_input, hidden[:batch_size])
                    loss += self.criterion(logits, targets)
            else:
                decoder_input = captions[:, 0] # <sos>
                for step in range(max(decode_lenghts)):
                    logits, hidden = self.model(imgs, decoder_input, hidden)
                    decoder_input = logits.argmax(dim=1)
                    # Drop <eos> from the batch, no need to decode it further
                    non_eos_idx = (
                        decoder_input != self.vocab.stoi('<eos>')
                    ).nonzero(as_tuple=False).view(-1)
                    if non_eos_idx.shape[0] == 0:
                        break
                    decoder_input = decoder_input[non_eos_idx]
                    imgs = imgs[non_eos_idx]
                    targets = captions[non_eos_idx][:, step + 1]
                    hidden = hidden[non_eos_idx]
                    loss += self.criterion(logits[non_eos_idx], targets)

            current_epoch_history.append(loss.cpu().detach().numpy())
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

            if (i + 1) % 10 == 0:
                self.plot_history(current_epoch_history)

        return sum(current_epoch_history) / len(current_epoch_history)

    def validate(self, val_loader):
        self.model.eval()

        imgs_ids = list(range(self.batch_size * len(val_loader)))
        # true captions for calculating CIDEr and BLEU-4
        references = {img_id: caps for (img_id, caps) 
                      in zip(imgs_ids, self.val.captions)}
        # here we will store predictions
        hypotheses = {img_id: [] for img_id in imgs_ids} 
        val_metrics = {'loss': [], 'cider': 0, 'bleu4': 0}

        with torch.no_grad():
            for iteration, (imgs, captions, lenghts) in enumerate(val_loader):
                imgs = imgs.to(self.device)
                captions = captions.to(self.device)
                decode_lenght = max(lenghts - 1)

                loss = 0
                hidden = imgs
                decoder_input = captions[:, 0] # <sos>
                for step in range(decode_lenght):
                    logits, hidden = self.model(imgs, decoder_input, hidden)
                    decoder_input = logits.argmax(dim=1)

                    # put predictions in hypotheses dict
                    imgs_ids = [id + iteration * self.batch_size for id
                                in range(self.batch_size)]
                    output = decoder_input.cpu().numpy()
                    for img_id, token in zip(imgs_ids, output):
                        string = self.vocab.itos(int(token))
                        if string not in ['<eos>', '<pad>']:
                            hypotheses[img_id].append(string)

                    targets = captions[:, step + 1]
                    loss += self.criterion(logits, targets)

                val_metrics['loss'].append(loss.cpu().detach().numpy())
        
        # prepare hypotheses for evaluation
        for img_id in hypotheses.keys():
            str_list = []
            str_list.append(' '.join(hypotheses[img_id]))
            hypotheses[img_id] = str_list
        
        val_metrics['loss'] = sum(val_metrics['loss']) / len(val_loader)
        val_metrics['cider'] = cider.compute_score(references, hypotheses)[0]
        val_metrics['bleu4'] = bleu.compute_score(references, hypotheses)[0][3]

        return val_metrics

    def plot_history(self, current_epoch_history):
        """
        Plots 4 types of plots:
            1) 1 is a loss history of the current epoch.
            2) 2 is a loss history of the entire training process.
            3) 3 and 4 are CIDEr and BLEU-4 respectively.
        """
        _, ax = plt.subplots(nrows=2, ncols=2)
        clear_output(True)

        ax[0, 0].plot(current_epoch_history, label='Epoch train loss')
        ax[0, 0].set_xlabel('Batch')
        ax[0, 0].set_title('Epoch train loss')

        ax[0, 1].plot(self.train_history, label='Train loss')
        ax[0, 1].plot(self.val_history, label='Valid loss')
        ax[0, 1].set_xlabel('Epoch')
        ax[0, 1].set_title('Entire history')

        ax[1, 0].plot(self.cider_history, label='CIDEr')
        ax[1, 0].set_xlabel('Epoch')
        ax[1, 0].set_title('CIDEr')

        ax[1, 1].plot(self.bleu4_history, label='BLEU-4')
        ax[1, 1].set_xlabel('Epoch')
        ax[1, 1].set_title('BLEU-4')

        plt.legend()
        plt.show()
