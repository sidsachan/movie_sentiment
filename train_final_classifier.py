import os
import torch.optim

from data_loader import LabelledTextDS
from models import FastText, FastText_mlp
from plotting import *
from training import train_model


def pretrained_mlp_final(dataset, dev, num_epochs=15, lr=0.01, num_hidden=256):
    embeddings = torch.load(os.path.join('saved_models', 'w5L_word_embeddings_h256_10000.pth')).embeddings.weight.data

    model = FastText_mlp(vocab_size=len(dataset.token_to_id) + 2, embedding_dim=num_hidden, l1_dim=64,
                         num_classes=len(dataset.class_to_id), word_embeddings=embeddings).to(dev)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total trainable params: ', pytorch_total_params)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1, verbose=True)

    losses, accuracies = train_model(dataset, model, optimizer, num_epochs, scheduler)
    torch.save(model, os.path.join('saved_models', 'classifier.pth'))

    print('')
    print_accuracies(accuracies)
    plot_losses(losses)


def testl1_dim(dataset, dev, l1_ds, num_epochs=15, num_hidden=256):
    loss = []
    for l1_d in l1_ds:
        embeddings = torch.load(os.path.join('saved_models', 'w5L_word_embeddings_h256_10000.pth')).embeddings.weight.data

        model = FastText_mlp(vocab_size=len(dataset.token_to_id) + 2, embedding_dim=num_hidden, l1_dim=l1_d,
                             num_classes=len(dataset.class_to_id), word_embeddings=embeddings).to(dev)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Total trainable params: ', pytorch_total_params)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1, verbose=True)

        losses, accuracies = train_model(dataset, model, optimizer, num_epochs, scheduler)
        loss.append(losses)
        torch.save(model, os.path.join('saved_models', 'classifier.pth'))

        print('')
        print_accuracies(accuracies)

    for losses in loss:
        plot_losses(losses)


num_epochs = 30
num_hidden = 256  # Number of hidden neurons in model
l1_ds = [32, 64]

dev = 'cuda' if torch.cuda.is_available() else 'cpu'  # If you have a GPU installed, use that, otherwise CPU
dataset = LabelledTextDS(os.path.join('data', 'labelled_movie_reviews.csv'), dev=dev)


# testing different dimension for mlp
# testl1_dim(dataset, dev, l1_ds)
pretrained_mlp_final(dataset, dev)