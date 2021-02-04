import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class Attention(nn.Module):
    def __init__(self, cnn_feature_size, hidden_size, attention_dim):
        super().__init__()
        self.encoder_attn = nn.Linear(cnn_feature_size, attention_dim)
        self.decoder_attn = nn.Linear(hidden_size, attention_dim)
        self.full_attn = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, img, hidden_state):
        """
        :param img: encoded images - (batch_size, cnn_feature_size)
        :param hidden_state: on the t-step - (batch_size, hidden_size)
        """
        attn1 = self.encoder_attn(img) # (batch_size, attention_dim)
        attn2 = self.decoder_attn(hidden_state) # (batch_size, attention_dim)
        attn = self.full_attn(self.relu(attn1 + attn2))

class AttentionDecoder(nn.Module):
    def __init__(self, vocab, hidden_size, num_layers, lstm_dropout=0.3,
                 fc_dropout=0.3, embedding_dim=300, cnn_feature_size=2048):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.fc_h = nn.Linear(cnn_feature_size, hidden_size * num_layers)
        self.fc_c = nn.Linear(cnn_feature_size, hidden_size * num_layers)
        self.bn = nn.BatchNorm1d(cnn_feature_size)

        self.embedding = nn.Embedding(num_embeddings=len(vocab), 
                                      embedding_dim=embedding_dim,
                                      padding_idx=vocab.pad_idx)

        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=lstm_dropout,
                            batch_first=True)

        self.fc_out = nn.Linear(hidden_size, len(vocab))
        self.dropout = nn.Dropout(fc_dropout)

    def init_hidden_states(self, image_vectors):
        """
        (bs, cnn_feature_size) -fc-> (bs, hidden_size * num_layers)
                          -reshape-> (num_layers, bs, hidden_size)
        """
        image_vectors = self.bn(image_vectors)
        h = self.fc_h(image_vectors)
        c = self.fc_c(image_vectors)
        h = h.reshape((self.num_layers, -1, self.hidden_size))
        c = c.reshape((self.num_layers, -1, self.hidden_size))
        return h, c

    def forward(self, image_vectors, padded_captions, lenghts):
        """
        :param image_vectors: (bs, cnn_feature_size)
        :param padded_captions: (bs, max_seq_len)
        :param lenghts: (bs)
        """
        # embedded_captions: (bs, max_seq_len, embedding_dim)
        embedded_captions = self.embedding(padded_captions)
        packed_captions = pack_padded_sequence(
            embedded_captions,
            lengths=lenghts,
            batch_first=True,
            enforce_sorted=False
        )

        h0, c0 = self.init_hidden_states(image_vectors)
        output, _ = self.lstm(packed_captions, (h0, c0))
        output = pad_packed_sequence(output, batch_first=True)

        # output[0]: (bs, max_seq_len, hidden_size)
        logits = self.fc_out(self.dropout(output[0]))

        return logits
