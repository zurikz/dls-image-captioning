import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class AttentionDecoder(nn.Module):
    def __init__(self, vocab, num_layers, hidden_size=512, 
                 dropout=0.1, cnn_feature_size=2048):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_dim = hidden_size
        self.cnn_feature_size = cnn_feature_size
        self.vocab = vocab

        self.bn = nn.BatchNorm1d(cnn_feature_size)
        self.fc_h = nn.Linear(cnn_feature_size, num_layers * hidden_size)
        self.fc_c = nn.Linear(cnn_feature_size, num_layers * hidden_size)

        self.attn = nn.Linear(2 * hidden_size, cnn_feature_size)
        self.attn_combine = nn.Linear(2 * hidden_size, hidden_size)

        self.embedding = nn.Embedding(num_embeddings=len(vocab), 
                                      embedding_dim=hidden_size,
                                      padding_idx=vocab.pad_idx)

        self.lstm = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)

        self.fc_out = nn.Linear(hidden_size, len(vocab))
        self.dropout = nn.Dropout(dropout)

    def init_hidden_states(self, image_vectors):
        """
        (bs, cnn_feature_size) -> (bs, hidden_size) 
        """
        if self.batch_size > 1:
            image_vectors = self.bn(image_vectors)
        h = self.fc_h(image_vectors)
        c = self.fc_c(image_vectors)
        return (h.view(self.num_layers, -1, self.hidden_size),
                c.view(self.num_layers, -1, self.hidden_size))

    def forward(self, img, input_token, hidden):
        """
        :param img: (bs, cnn_feature_size)
        :param input_token: (bs, )
        :param hidden: on the first decoding step: (bs, cnn_feature_size)
                       then: (bs, num_layers, hidden_size)
        """
        self.batch_size = input_token.shape[0]
        if hidden.shape[1] == self.cnn_feature_size:
            hidden, _ = self.init_hidden_states(img)

        # (bs, num_layers, hidden) -> (num_layers, bs, hidden)
        hidden = hidden.contiguous().view(
            self.num_layers, -1, self.hidden_size
        )
        
        # embedded: (bs, embedding_dim)
        embedded = self.embedding(input_token)
        embedded = self.dropout(embedded)

        # print(f'emb {embedded.shape}')
        # print(f'hid {hidden.shape}')
        # print(torch.cat((embedded, hidden), dim=1).shape)

        # attn_weights: (bs, cnn_feature_size)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden[0]), dim=1)),
            dim=1
        )
        attn_applied = attn_weights * img
        attn_hidden, attn_c = self.init_hidden_states(attn_applied)

        output = torch.cat((embedded, attn_hidden[0]), dim=1)
        output = self.attn_combine(output)
        output = F.relu(output)

        # output: (bs, 1, hidden_size)
        output, _ = self.lstm(output.unsqueeze(1), (hidden, attn_c))
        logits = self.fc_out(self.dropout(output.squeeze(1)))

        return logits, hidden.permute(1, 0, 2)
