import os
import pickle
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda
import numpy as np
import random


def load_data_unsupervised(batch_size=64, validation_split = None, dataset="MNIST"):
    # Download training data from open datasets.

    if dataset == "MNIST":
        data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        # Download test data from open datasets.
        test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )

    elif dataset == "SVHN":
        data = datasets.SVHN(
            root="data",
            split="train",
            download=True,
            transform=ToTensor(),
        )

        test_data = datasets.SVHN(
            root="data",
            split="test",
            download=True,
            transform=ToTensor(),
        )

    elif dataset == "CIFAR10":
        data = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        # Download test data from open datasets.
        test_data = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )


    n = data.data.shape[0]

    # splitting between train and validation
    if validation_split:
        num_validation_samples = round(n * validation_split)
        num_training_samples = (n - num_validation_samples)
        training_data,  validation_data = torch.utils.data.random_split(data, [num_training_samples,num_validation_samples])
    else:
        training_data = data

    # wrap an iterable over the datasets
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    if validation_split:
        validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    else:
        validation_dataloader = None

    return train_dataloader, validation_dataloader, test_dataloader


def load_supervised_data_as_generator(batch_size=64,num_training_samples=100,cheat_validation=False,dataset="MNIST"):

    # Download training data from open datasets.
    if dataset == "MNIST":
        training_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        # Download test data from open datasets.
        test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )
    elif dataset == "SVHN":
        training_data = datasets.SVHN(
            root="data",
            split="train",
            download=True,
            transform=ToTensor(),
        )

        # Download test data from open datasets.
        test_data = datasets.SVHN(
            root="data",
            split="test",
            download=True,
            transform=ToTensor(),
        )
    elif dataset == "CIFAR10":
        training_data = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        # Download test data from open datasets.
        test_data = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )

    if dataset in "MNIST":
        num_classes = max(training_data.targets).item() + 1
    elif dataset == "SVHN":
        num_classes = max(training_data.labels).item() + 1
    elif dataset == "CIFAR10":
        num_classes = max(training_data.targets) + 1

    assert num_training_samples % num_classes == 0, "The number of training samples ("  + str(num_training_samples) \
                                                  +") must be divisible by the number of classes (" + str(num_classes)\
                                                  +")"

    num_samples_per_class = num_training_samples//num_classes

    num_validation_samples = num_training_samples//10

    # if we want many validation samples 
    if cheat_validation:
        num_validation_samples = max(num_validation_samples,1000)

    num_validation_samples_per_class = max(num_validation_samples//num_classes,1)


    # separate the data per class
    data_divided_per_class = [[] for _ in range(num_classes)]
    for (img,label) in training_data:
        data_divided_per_class[label].append((img,label))

    # shuffle all lists and select a subset
    selected_training_data = []
    selected_validation_data = []
    for j in range(num_classes):
        random.shuffle(data_divided_per_class[j])
        selected_training_data.extend(data_divided_per_class[j][:num_samples_per_class])
        selected_validation_data.extend(data_divided_per_class[j][num_samples_per_class:num_samples_per_class+num_validation_samples_per_class])

    # shuffle the new dataset once last time to avoid the classes between clustered together
    random.shuffle(selected_training_data)
    random.shuffle(selected_validation_data)

    # wrap an iterable over the datasets
    train_dataloader = DataLoader(selected_training_data, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(selected_validation_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


    # get img_width and height
    num_channels, img_height, img_width = train_dataloader.dataset[0][0].shape[0], train_dataloader.dataset[0][0].shape[1], train_dataloader.dataset[0][0].shape[2]
    return train_dataloader, validation_dataloader, test_dataloader, img_height, img_width, num_channels, num_classes

def load_joint_data_as_generator(batch_size=64,num_labelled_samples=100, validation_split=None, percentage_unlabelled_set=1, data_version=0, dataset="MNIST"):

    file_path = "data/" + dataset + "/pre_saved/save_data_%sS_v%s.pckl" % (num_labelled_samples, data_version)

    if os.path.isfile(file_path):

        with open(file_path,'rb') as f:
            dic_saved = pickle.load(f)

        pre_saved_shuffling = dic_saved["pre_saved_shuffling"]
        seed = dic_saved["seed"]

    else:

        pre_saved_shuffling = None
        pre_saved_shuffling_to_save = []
        seed = 1234 + data_version

    if dataset == "MNIST":
        # Download training data from open datasets.
        data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        # Download test data from open datasets.
        test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )
    elif dataset == "SVHN":
        data = datasets.SVHN(
            root="data",
            split="train",
            download=True,
            transform=ToTensor(),
        )

        # Download test data from open datasets.
        test_data = datasets.SVHN(
            root="data",
            split="test",
            download=True,
            transform=ToTensor(),
        )

    elif dataset == "CIFAR10":
        data = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        # Download test data from open datasets.
        test_data = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )

    n = data.data.shape[0]

    # splitting between train and validation
    if validation_split:
        num_validation_samples = round(n * validation_split)
        num_training_samples = (n - num_validation_samples)
        training_data, validation_data = torch.utils.data.random_split(data,
                                                                       [num_training_samples, num_validation_samples],
                                                                       generator=torch.Generator().manual_seed(seed))
    else:
        training_data = data

    if dataset in "MNIST":
        num_classes = max(training_data.dataset.targets).item() + 1
    elif dataset == "SVHN":
        num_classes = max(training_data.dataset.labels).item() + 1
    elif dataset == "CIFAR10":
        num_classes = max(training_data.dataset.targets) + 1

    assert num_labelled_samples % num_classes == 0, "The number of training samples ("  + str(num_training_samples) \
                                                    +") must be divisible by the number of classes (" + str(num_classes)\
                                                    +")"

    num_samples_per_class = num_labelled_samples//num_classes

    # separate the data per class
    data_divided_per_class = [[] for _ in range(num_classes)]
    for (img,label) in training_data:
        data_divided_per_class[label].append((img,label))

    # shuffle all lists and select a subset
    selected_training_labelled_data = []
    selected_training_unlabelled_data = []
    if pre_saved_shuffling == None:
        pre_saved_shuffling_to_save.append([])
    for j in range(num_classes):
        if pre_saved_shuffling != None:
            indices = pre_saved_shuffling[0][j]
        else:
            indices = list(range(len(data_divided_per_class[j])))
            random.shuffle(indices)
            pre_saved_shuffling_to_save[0].append(indices)
        data_divided_per_class[j] = [data_divided_per_class[j][i] for i in indices]
        selected_training_labelled_data.extend(data_divided_per_class[j][:num_samples_per_class])
        selected_training_unlabelled_data.extend(data_divided_per_class[j][num_samples_per_class:])

    # remove the label from the unlabelled data
    processed_training_unlabelled_data = []
    for (img,label) in selected_training_unlabelled_data:
        processed_training_unlabelled_data.append((img,-1))

    if pre_saved_shuffling != None:
        indices = pre_saved_shuffling[1]
    else:
        indices = list(range(len(processed_training_unlabelled_data)))
        random.shuffle(indices)
        pre_saved_shuffling_to_save.append(indices)
    processed_training_unlabelled_data = [processed_training_unlabelled_data[i] for i in indices]
    num_unlabelled = round(percentage_unlabelled_set * len(processed_training_unlabelled_data))
    processed_training_unlabelled_data = processed_training_unlabelled_data[:num_unlabelled]

    # combine both labelled and unlabelled data and shuffle
    processed_training_data =  processed_training_unlabelled_data + selected_training_labelled_data
    indices = list(range(len(processed_training_data)))
    random.shuffle(indices)
    processed_training_data = [processed_training_data[i] for i in indices]

    if pre_saved_shuffling == None:

        # create the directory if necessary
        dir_to_create = os.path.dirname("".join(file_path))
        os.makedirs(dir_to_create, exist_ok=True)

        dic_to_save = {"pre_saved_shuffling":pre_saved_shuffling_to_save,
                       "seed":seed}
        # save the object
        with open(file_path, 'wb') as f:
            pickle.dump(dic_to_save, f)

    # wrap an iterable over the datasets
    train_dataloader = DataLoader(processed_training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    if validation_split:
        validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    else:
        validation_dataloader = None

    # get img_width and height
    num_channels, img_height, img_width = train_dataloader.dataset[0][0].shape[0], train_dataloader.dataset[0][0].shape[1], train_dataloader.dataset[0][0].shape[2]
    return train_dataloader, validation_dataloader, test_dataloader, num_classes, num_unlabelled, img_height, img_width, num_channels

def load_supervised_data_as_matrix(num_training_samples=100,cheat_validation=False,dataset="MNIST"):

    # load the data
    if dataset == "MNIST":
        training_data = datasets.MNIST(
                root="data",
                train=True,
                download=True,
                transform=ToTensor(),
            )

        test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
            )

    elif dataset == "SVHN":
        training_data = datasets.SVHN(
            root="data",
            split="train",
            download=True,
            transform=ToTensor(),
        )

        # Download test data from open datasets.
        test_data = datasets.SVHN(
            root="data",
            split="test",
            download=True,
            transform=ToTensor(),
        )

    elif dataset == "CIFAR10":
        training_data = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        # Download test data from open datasets.
        test_data = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )

    # convert the data to numpy matrices
    X = training_data.data
    total_nbr_samples, img_height,img_width = X.shape[0], X.shape[2], X.shape[3]
    if len(X.shape) == 4:
        num_channels = X.shape[1]

    X = X.reshape(total_nbr_samples, -1)
    X_test = test_data.data
    total_nbr_test_samples, img_height_test, img_width_test = X_test.shape[0], X_test.shape[2], X_test.shape[3]
    X_test = X_test.reshape(total_nbr_test_samples, -1)
    if dataset == "MNIST" :
        y = training_data.targets
        y_test = test_data.targets
    elif dataset == "SVHN":
        y = training_data.labels
        y_test = test_data.labels
    elif dataset == "CIFAR10":
        y = training_data.targets
        y_test = test_data.targets

    # shuffle the data ordering (only shuffles rows in the matrix)
    shuffler = np.random.permutation(X.shape[0])
    X = X[shuffler]
    y = y[shuffler]

    # get the number of training and validation samples per class
    if dataset in "MNIST":
        num_classes = max(training_data.targets).item() + 1
    elif dataset == "SVHN":
        num_classes = max(training_data.labels).item() + 1
    elif dataset == "CIFAR10":
        num_classes = max(training_data.targets) + 1

    assert num_training_samples % num_classes == 0, "The number of training samples (" + str(num_training_samples) \
                                                    + ") must be divisible by the number of classes (" + str(
        num_classes) \
                                                    + ")"

    num_samples_per_class = num_training_samples // num_classes

    num_validation_samples = num_training_samples // 10

    # if we want many validation samples 
    if cheat_validation:
        num_validation_samples = max(num_validation_samples,1000)

    num_validation_samples_per_class = max(num_validation_samples // num_classes, 1)

    # separate the data per class (by indices)
    indices_per_class = [[] for _ in range(num_classes)]
    for i,label in enumerate(y):
        indices_per_class[label].append(i)

    # create the validation and test matrices with the right amount of training and validation samples
    X_train = np.zeros((num_training_samples,img_height * img_width * num_channels))
    X_validation = np.zeros((num_validation_samples,img_height * img_width * num_channels))
    y_train = np.zeros((num_training_samples))
    y_validation = np.zeros((num_validation_samples))

    idx = -1
    for label in range(num_classes):
        idx += 1
        y_train[idx*num_samples_per_class:(idx+1)*num_samples_per_class] = y[indices_per_class[label][:num_samples_per_class]]
        y_validation[idx*num_validation_samples_per_class:(idx+1)*num_validation_samples_per_class] = y[indices_per_class[label][num_samples_per_class:num_samples_per_class+num_validation_samples_per_class]]
        X_train[idx*num_samples_per_class:(idx+1)*num_samples_per_class] = X[indices_per_class[label][:num_samples_per_class]]
        X_validation[idx*num_validation_samples_per_class:(idx+1)*num_validation_samples_per_class] = X[indices_per_class[label][num_samples_per_class:num_samples_per_class+num_validation_samples_per_class]]

    # shuffle one last time to avoid the class ordering in the data
    shuffler2 = np.random.permutation(X_train.shape[0])
    X_train = X_train[shuffler2]
    y_train = y_train[shuffler2]
    shuffler3 = np.random.permutation(X_validation.shape[0])
    X_validation = X_validation[shuffler3]
    y_validation = y_validation[shuffler3]

    return X_train,y_train,X_validation,y_validation,X_test,y_test