import os
from data_loader import TextDS
from data_loader import LabelledTextDS
import pickle
import torch


def save_datasets(vocab_size, device):
    dataset = TextDS(os.path.join('data', 'unlabelled_movie_reviews.csv'), num_terms=vocab_size, dev=device)
    filename = os.path.join('./datasets', 'vocab_size_'+str(vocab_size))
    outfile = open(filename, 'wb')
    pickle.dump(dataset, outfile)
    outfile.close()


def get_dataset(path):
    infile = open(path, 'rb')
    dataset = pickle.load(infile)
    infile.close()
    return dataset


dev = 'cuda' if torch.cuda.is_available() else 'cpu'
num_words = [2500, 5000, 7500, 10000]

# for labelled dataset
dataset = LabelledTextDS(os.path.join('data', 'labelled_movie_reviews.csv'), dev=dev)
filename = os.path.join('./datasets', 'Labelled_vocab_size_7500')
outfile = open(filename, 'wb')
pickle.dump(dataset, outfile)
outfile.close()

# for un-labelled set
# for vocab_size in num_words:
#     save_datasets(vocab_size, dev)