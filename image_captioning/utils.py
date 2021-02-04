import json
import numpy as np
from collections import Counter

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class Vocab:
    def __init__(self, captions_path: str, min_freq=5):
        self.captions = json.load(open(captions_path))
        self.word2idx, self.idx2word = self.build_vocab(min_freq)
        self.unk_idx = self.word2idx['<unk>']
        self.pad_idx = self.word2idx['<pad>']

    def build_vocab(self, min_freq):    
        word_freq = Counter()

        for idx in range(len(self.captions)):
            for sent_idx in range(len(self.captions[idx])):
                word_freq.update((self.captions[idx][sent_idx]).split())

        words = [w for w in word_freq.keys() if word_freq[w] > min_freq]
        special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        word2idx = {word: idx for idx, word in enumerate(special_tokens + words)}
        idx2word = {idx: word for idx, word in enumerate(special_tokens + words)}
        return word2idx, idx2word

    def stoi(self, words):
        if isinstance(words, str):
            word = words
            return self.word2idx[word] if word in self.word2idx \
                                       else self.unk_idx
        elif isinstance(words, list):
            return [self.word2idx[word] if word in self.word2idx
                    else self.unk_idx for word in words]

    def itos(self, idx):
        if isinstance(idx, int):
            return self.idx2word[idx]
        elif isinstance(idx, list):
            return [self.idx2word[i] for i in idx]
        
    def __len__(self):
        return len(self.word2idx)


class CaptionDataset(Dataset):
    def __init__(self, split: str, vocab: Vocab,
                 img_codes_path: str, captions_path: str):
        self.img_codes = np.load(img_codes_path)
        self.captions = json.load(open(captions_path))

        assert split in {'train', 'val', 'test'}
        self.split = split

        idx = len(self.img_codes) - 10000
        if split == 'train':
            self.img_codes = self.img_codes[:idx]
            self.captions = self.captions[:idx]
            self.encode_sentences(vocab)
        if split == 'val':
            self.img_codes = self.img_codes[idx:(idx + 5000)]
            self.captions = self.captions[idx:(idx + 5000)]
        if split == 'test':
            self.img_codes = self.img_codes[(idx + 5000):]
            self.captions = self.captions[(idx + 5000):]

        self.sort_dataset()

    def encode_sentences(self, vocab, max_len=20):
        for idx in range(len(self.captions)):
            for sent_idx in range(len(self.captions[idx])):
                sent = '<sos> ' + self.captions[idx][sent_idx] + ' <eos>'
                encoded_sent = vocab.stoi(sent.split())
                if len(encoded_sent) > max_len:
                    encoded_sent = encoded_sent[:max_len]
                    encoded_sent[-1] = vocab.stoi('<eos>')
                self.captions[idx][sent_idx] = encoded_sent

    def sort_dataset(self):
        self.captions, self.img_codes = zip(
            *sorted(
                zip(self.captions, self.img_codes),
                key=lambda cap_img: sum([len(cap) for cap in cap_img[0]])
            )
        )

    def __getitem__(self, idx):
        img = self.img_codes[idx]
        if self.split == 'train':
            random = np.random.randint(low=0, high=len(self.captions[idx]))
            caption = self.captions[idx][random]
            return (torch.tensor(img), 
                    torch.tensor(caption), 
                    torch.tensor(len(caption)))
        else:
            return (torch.tensor(img), 
                    self.captions[idx])

    def __len__(self):
        return len(self.captions)


def pad_collate_fn(batch):
    image_vectors, captions, lenghts = zip(*batch)
    padded_captions = pad_sequence(captions, batch_first=True)
    image_vectors = torch.stack(image_vectors)
    lenghts = torch.stack(lenghts)
    return image_vectors, padded_captions, lenghts


def create_datasets(vocab: Vocab, img_codes_path, captions_path):
    train = CaptionDataset('train', vocab, img_codes_path, captions_path)
    val = CaptionDataset('val', vocab, img_codes_path, captions_path)
    test = CaptionDataset('test', vocab, img_codes_path, captions_path)
    return {'train': train, 'val': val, 'test': test}
