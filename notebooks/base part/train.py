import torch
import torch.nn as nn

embedding_dim = 512
hidden_size = 512
lstm_dropout = 0.3
fc_dropout = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, loader, criterion, optimizer, clip, device):
    model.train()

    for i, (imgs, captions, lenghts) in enumerate(loader):
        imgs = imgs.to(device)
        captions = captions.to(device)
        lenghts = lenghts.to(device)

        logits = model(imgs, captions, lenghts)


