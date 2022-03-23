import torch.nn as nn
import torch


class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, word_embeddings=None):
        """
        :param vocab_size: the number of different embeddings to make (need one embedding for every unique word).
        :param embedding_dim: the dimension of each embedding vector.
        :param num_classes: the number of target classes.
        :param word_embeddings: optional pre-trained word embeddings. If not given word embeddings are trained from
        random initialization. If given then provided word_embeddings are used and the embeddings are not trained.
        """
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if word_embeddings is not None:
            self.embeddings = self.embeddings.from_pretrained(word_embeddings, freeze=True, padding_idx=0)
        self.W = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        """
        :param x: a LongTensor of shape [batch_size, max_sequence_length]. Each row is one sequence (movie review),
        the i'th element in a row is the (integer) ID of the i'th token in the original text.
        :return: a FloatTensor of shape [batch_size, num_classes]. Predicted class probabilities for every sequence
        in the batch.
        """
        # TODO perform embed, aggregate, and linear, then return the predicted class probabilities.
        # embed
        o1 = self.embeddings(x)
        # sum aggregate
        # o2 = torch.sum(o1, dim=1)
        # mean aggregate
        o2 = torch.mean(o1, dim=1)
        # max aggregation
        # o2, _ = torch.max(o1, dim=1)
        # linear
        y = self.W(o2)
        return y


class FastText_mlp(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, l1_dim=32, l2_dim=8, word_embeddings=None):
        """
        :param vocab_size: the number of different embeddings to make (need one embedding for every unique word).
        :param embedding_dim: the dimension of each embedding vector.
        :param l1_dim: the input dimension of the extra non linear layer.
        :param num_classes: the number of target classes.
        :param word_embeddings: optional pre-trained word embeddings. If not given word embeddings are trained from
        random initialization. If given then provided word_embeddings are used and the embeddings are not trained.
        """
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if word_embeddings is not None:
            self.embeddings = self.embeddings.from_pretrained(word_embeddings, freeze=True, padding_idx=0)
        self.relu = nn.ReLU()
        self.W1 = nn.Linear(embedding_dim, l1_dim)
        self.W2 = nn.Linear(l1_dim, l2_dim)
        self.W3 = nn.Linear(l2_dim, num_classes)

    def forward(self, x):
        """
        :param x: a LongTensor of shape [batch_size, max_sequence_length]. Each row is one sequence (movie review),
        the i'th element in a row is the (integer) ID of the i'th token in the original text.
        :return: a FloatTensor of shape [batch_size, num_classes]. Predicted class probabilities for every sequence
        in the batch.
        """
        # embed
        o1 = self.embeddings(x)
        # mean aggregate
        o2 = torch.mean(o1, dim=1)
        # mlp part
        y = self.W1(o2)
        y = self.relu(y)
        y = self.W2(y)
        y = self.relu(y)
        y = self.W3(y)
        return y