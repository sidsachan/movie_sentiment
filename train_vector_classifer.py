import os
from sklearn.linear_model import LogisticRegression

from data_loader import LabelledTextDS
from plotting import *


def train_linear_simple(model, train, valid, test):
    model.fit(train[0], train[1])
    train_accuracy = (model.predict(train[0]) == train[1]).astype(float).mean()
    valid_accuracy = (model.predict(valid[0]) == valid[1]).astype(float).mean()
    test_accuracy = (model.predict(test[0]) == test[1]).astype(float).mean()
    return train_accuracy, valid_accuracy, test_accuracy


# Function to test with different max_feature
def test_max_features(dataset, feature_list, verbose=False):
    val_list = []
    for f in feature_list:
        train, valid, test = dataset.get_vector_representation(maxdf=0.4, maxfeatures=f)
        train_accuracy, valid_accuracy, test_accuracy = train_linear_simple(model, train, valid, test)
        val_list.append(valid_accuracy)
        if verbose:
            print('\n')
            print_accuracies((train_accuracy, valid_accuracy, test_accuracy))
    # plot val accuracy vs max feature
    plotxy(feature_list, val_list, xlabel="max_features")


# Function to test with the max_df of Count Vectorizer
def test_max_df(dataset, verbose=False):
    f_list = np.arange(50, 550, 50)
    dfs = np.arange(0.1, 1.1, 0.1)
    val_dict = {}
    for f in f_list:
        val_list = []
        for df in dfs:
            train, valid, test = dataset.get_vector_representation(maxdf=df, maxfeatures=f)
            train_accuracy, valid_accuracy, test_accuracy = train_linear_simple(model, train, valid, test)
            val_list.append(valid_accuracy)
            if verbose:
                print('(features, df) = ', f, '\t', df, '\n')
                print_accuracies((train_accuracy, valid_accuracy, test_accuracy))
        val_dict[f] = val_list
    plot_val_acc(val_dict)


# function to test with different models for convergence quality
def test_logistic(model_list, train_set, valid_set, test_set, verbose=False):
    val_list = []
    for model in model_list:
        train_accuracy, valid_accuracy, test_accuracy = train_linear_simple(model, train_set, valid_set, test_set)
        val_list.append(valid_accuracy)
        if verbose:
            print('\n')
            print_accuracies((train_accuracy, valid_accuracy, test_accuracy))


# test result with different regularization strengths for (saga solver, L2 error)
def test_regularization(dataset, df, features, verbose=False):
    train, valid, test = dataset.get_vector_representation(maxdf=df, maxfeatures=features)
    val_list = []
    Cs = np.arange(0.1, 1.1, 0.1)
    for c in Cs:
        model = LogisticRegression(solver='saga', C=c)
        train_accuracy, valid_accuracy, test_accuracy = train_linear_simple(model, train, valid, test)
        val_list.append(valid_accuracy)
        if verbose:
            print('C=', 0.2 * c, '\n')
            print_accuracies((train_accuracy, valid_accuracy, test_accuracy))

    plotxy(Cs, val_list, xlabel="Parameter C")


dataset = LabelledTextDS(os.path.join('data', 'labelled_movie_reviews.csv'))
train, valid, test = dataset.get_vector_representation(maxdf=0.4, maxfeatures=10000)

model = LogisticRegression()  # You can change the hyper-parameters of the model by passing args here
model1 = LogisticRegression(solver='saga')
model2 = LogisticRegression(solver='saga', penalty='l1')
model3 = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.1)
model4 = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.5)
model5 = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.9)
models = [model, model1, model2, model3, model4, model5]

features_list = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

# Run the following function to get the plots reported for part 1
# test_max_features(dataset, features_list)
# test_max_df(dataset)
# test_logistic(models, train, valid, test, verbose=True)
# test_regularization(dataset, 0.4, 10000)

train_accuracy, valid_accuracy, test_accuracy = train_linear_simple(model1, train, valid, test)
print('\n')
print_accuracies((train_accuracy, valid_accuracy, test_accuracy))
