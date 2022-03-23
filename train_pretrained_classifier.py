import os
import torch.optim

from data_loader import LabelledTextDS
from models import FastText
from plotting import *
from training import train_model
from save_datasets import get_dataset


def test_embedding_h(contest_window, dataset, hs, dev, num_epochs=50, lr=0.01):
    loss =[]
    for h in hs:
        embeddings = torch.load(os.path.join('saved_models', 'w'+str(contest_window)+'_word_embeddings_h'+str(h)+'_10000.pth')).embeddings.weight.data
        model = FastText(len(dataset.token_to_id) + 2, h, len(dataset.class_to_id), word_embeddings=embeddings).to(
            dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        losses, accuracies = train_model(dataset, model, optimizer, num_epochs,scheduler)
        print_accuracies(accuracies)
        loss.append(losses)
        torch.save(model, os.path.join('saved_models', 'classifier.pth'))

    for losses in loss:
        print('')
        plot_losses(losses)


num_epochs = 50
num_hidden = 32  # Number of hidden neurons in model
hs =[128, 256]
vocab_size = 10000
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

if os.path.isfile('./datasets/Labelled_vocab_size_'+str(vocab_size)):
    print("loading stored dataset")
    dataset = get_dataset('./datasets/Labelled_vocab_size_'+str(vocab_size))
else:
    dataset = LabelledTextDS(os.path.join('data', 'labelled_movie_reviews.csv'), dev=dev)

embeddings = torch.load(os.path.join('saved_models', 'w5L_word_embeddings_h32_10000.pth')).embeddings.weight.data
model = FastText(len(dataset.token_to_id)+2, num_hidden, len(dataset.class_to_id), word_embeddings=embeddings).to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold=1e-3, factor=0.2, patience=5, verbose=True)

losses, accuracies = train_model(dataset, model, optimizer, num_epochs, scheduler)
torch.save(model, os.path.join('saved_models', 'classifier.pth'))

print('')
print_accuracies(accuracies)
plot_losses(losses)

# test_embedding_h(5, dataset, hs, dev)