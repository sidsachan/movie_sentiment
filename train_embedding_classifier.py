import os
import torch.optim

from data_loader import LabelledTextDS
from models import FastText
from plotting import *
from training import train_model
from save_datasets import get_dataset


def test_lr_m(dataset, model, eps, lrs, ms):
    loss = []
    for lr in lrs:
        for mom in ms:
            # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
            losses, accuracies = train_model(dataset, model, optimizer, eps)
            torch.save(model, os.path.join('saved_models', 'classifier.pth'))
            loss.append(losses)
            print('m: ', mom, 'lr: ', lr)
            print_accuracies(accuracies)

    for losses in loss:
        plot_losses(losses)


def test_hidden(dataset, hidden_units, dev, lr=0.003, eps=10):
    loss = []
    for h in hidden_units:
        model = FastText(len(dataset.token_to_id) + 2, h, len(dataset.class_to_id)).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        losses, accuracies = train_model(dataset, model, optimizer, eps)
        loss.append(losses)
        print('h: ', h)
        print_accuracies(accuracies)

    for losses in loss:
        plot_losses(losses)


def test_with_scheduler(dataset, hidden_units, dev, lr=0.003, eps=10):
    loss = []
    for h in hidden_units:
        model = FastText(len(dataset.token_to_id) + 2, h, len(dataset.class_to_id)).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1,
                                                               verbose=True)
        losses, accuracies = train_model(dataset, model, optimizer, eps, scheduler)
        loss.append(losses)
        print('h: ', h)
        print_accuracies(accuracies)

    for losses in loss:
        plot_losses(losses)


num_epochs = 30
num_hidden = 1 # Number of hidden neurons in model
lr = 0.003
m = 0.95
vocab_size = 10000

dev = 'cuda' if torch.cuda.is_available() else 'cpu'  # If you have a GPU installed, use that, otherwise CPU
if os.path.isfile('./datasets/Labelled_vocab_size_'+str(vocab_size)):
    print("loading stored dataset")
    dataset = get_dataset('./datasets/Labelled_vocab_size_'+str(vocab_size))
else:
    dataset = LabelledTextDS(os.path.join('data', 'labelled_movie_reviews.csv'), dev=dev)

model = FastText(len(dataset.token_to_id) + 2, num_hidden, len(dataset.class_to_id)).to(dev)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total trainable params: ', pytorch_total_params)
print('Vocab size: ', len(dataset.token_to_id) + 2)
print('Number of classes: ', len(dataset.class_to_id))

lrs = [0.008, 0.005, 0.001]
ms = [0.8, 0.9, 0.95]
num_hiddens = [32, 64, 128]
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.95)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1,
                                                       verbose=True)
losses, accuracies = train_model(dataset, model, optimizer, num_epochs, scheduler)
torch.save(model, os.path.join('saved_models', 'classifier.pth'))
print('')
print_accuracies(accuracies)
plot_losses(losses)

# for testing different lr and momentum
# test_lr_m(dataset, model, num_epochs, lrs, ms)
# for testing different hidden units
# test_hidden(dataset, num_hiddens, dev)
# test_with_scheduler(dataset, num_hiddens, dev)