import os
import torch.optim

from data_loader import TextDS
from models import FastText
from plotting import *
from training import train_model

num_epochs = 5
num_hidden = 32
num_words = 10000

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = TextDS(os.path.join('data', 'labelled_movie_reviews.csv'), num_terms=num_words, dev=dev)
model = FastText(len(dataset.token_to_id) + 2, num_hidden, len(dataset.token_to_id) + 2).to(dev)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)
losses, accuracies = train_model(dataset, model, optimizer, num_epochs, scheduler)
torch.save(model, os.path.join('saved_models', 'w5L_word_embeddings_h' + str(num_hidden) + '_' + str(num_words) + '.pth'))
print(num_hidden, '\t', num_words)
print_accuracies(accuracies)
plot_losses(losses)
