import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_losses(losses):
    plt.plot(losses)
    plt.legend(['Training', 'Validation'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, len(losses)+1, len(losses)//10))
    plt.grid()
    plt.show()


def plot_embeddings(embeddings, names, word_list=None, max_num_words=20):
    if word_list is not None:
        word_list = set(word_list)
        inds = [i for i in range(len(names)) if names[i] in word_list]
        embeddings = embeddings[inds]
        names = np.array(names)[inds]
    else:
        embeddings = embeddings[:max_num_words]
        names = names[:max_num_words]
    embeddings = PCA(2).fit_transform(embeddings.cpu())
    plt.scatter(embeddings[:, 0], embeddings[:, 1])
    for i in range(len(embeddings)):
        plt.annotate(names[i], embeddings[i])
    plt.show()


def print_accuracies(accuracies):
    print(f'training accuracy: {accuracies[0]}')
    print(f'validation accuracy: {accuracies[1]}')
    print(f'testing accuracy: {accuracies[2]}')


def plot_val_acc(data_dict):
    list = np.arange(0.1, 1.1, 0.1)
    for i in range(10):
        plt.plot(list, data_dict[(i + 1) * 50], label=str((i + 1) * 50), linestyle='dashed', marker='o')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.xlabel('max_df')
    plt.ylabel('Validation Accuracy')
    plt.ylim(0.6, 0.85)
    plt.grid()
    plt.show()


def plotxy(x_list, y_list, xlabel):
    min = np.min(y_list)
    max = np.max(y_list)
    range =max - min
    plt.plot(x_list, y_list, label='Validation accuracy', linestyle='dashed', marker='o')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('Validation Accuracy')
    plt.ylim(min - 0.3*range, max + 0.3*range)
    plt.grid()
    plt.show()