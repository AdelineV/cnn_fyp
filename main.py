import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from models import DenseNet121
from sklearn.metrics import confusion_matrix
import seaborn as sn
import random


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


##########################
### SETTINGS
##########################

# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 16
NUM_EPOCHS = 20

# Architecture
NUM_CLASSES = 3

# Other
DEVICE = "cuda:0"
GRAYSCALE = True

# Dataset Folder
# dataroot = "datasets_clean/tb_clean"
# datatrain = "data/augment/tuberculosis-50/train"
# datatest = "data/augment/tuberculosis-50/test"
# datatrain = "data/augment/tuberculosis-5/train"
# datatest = "data/augment/tuberculosis-5/test"

# datatrain = "datasets_clean/aug-GAN/tuberculosis-100/train"
# datatest = "datasets_clean/aug-GAN/tuberculosis-100/test"
# datatrain = "datasets_clean/aug-classic/tuberculosis-100/train"
# datatest = "datasets_clean/aug-classic/tuberculosis-25/test"
# datatrain = "datasets_clean/no-aug/tuberculosis-100/train"
# datatest = "datasets_clean/no-aug/tuberculosis-100/test"

datatrain = "datasets_clean/testing/covid4-100-test/train"
datatest = "datasets_clean/testing/covid4-100-test/test"

# datatrain = "datasets_clean/aug-classic/covid4-100/train"
# datatest = "datasets_clean/aug-classic/covid4-10/test"
# datatrain = "datasets_clean/no-aug/covid4-100/train"
# datatest = "datasets_clean/no-aug/covid4-100/test"
# datatrain = "datasets_clean/no-aug/covid4-100/train"
# datatest = "datasets_clean/no-aug/covid4-100/test"

# covidtrain = "data/covid_three/train"
# covidfour = "data/covid_four"


def main():
    train_and_valid = datasets.ImageFolder(root=datatrain,
                                           transform=transforms.Compose([
                                               transforms.Resize(size=(224, 224)),
                                               transforms.Grayscale(1),
                                               transforms.ToTensor()
                                           ]))

    test_dataset = datasets.ImageFolder(root=datatest,
                                        transform=transforms.Compose([
                                            transforms.Resize(size=(224, 224)),
                                            transforms.Grayscale(1),
                                            transforms.ToTensor()
                                        ]))

    train_size = int(0.9 * len(train_and_valid))
    valid_size = len(train_and_valid) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(train_and_valid, [train_size, valid_size])
    print(len(train_and_valid), train_size, valid_size, len(test_dataset))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers=4,
                              shuffle=True)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers=4,
                              shuffle=False)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             num_workers=4,
                             shuffle=True)

    device = torch.device(DEVICE)
    torch.manual_seed(0)

    for epoch in range(2):

        for batch_idx, (x, y) in enumerate(train_loader):
            print('Epoch:', epoch + 1, end='')
            print(' | Batch index:', batch_idx, end='')
            print(' | Batch size:', y.size()[0])

            x = x.to(device)
            y = y.to(device)
            break

    # Check that shuffling works properly
    # i.e., label indices should be in random order.
    # Also, the label order should be different in the second
    # epoch.
    print(len(train_loader))

    for images, labels in train_loader:
        pass
    print(labels[-10:])

    for images, labels in train_loader:
        pass
    print(labels[-10:])

    # Check that validation set and test sets are diverse
    # i.e., that they contain all classes

    for images, labels in valid_loader:
        pass
    print(labels[:10])

    for images, labels in test_loader:
        pass
    print(labels[:10])

    torch.manual_seed(RANDOM_SEED)

    model = DenseNet121(num_classes=NUM_CLASSES, grayscale=GRAYSCALE, drop_rate=0.1)
    model.to(DEVICE)
    # print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    def compute_acc(model, data_loader, device):
        correct_pred, num_examples = 0, 0
        model.eval()
        for i, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.to(device)

            logits, probas = model(features)
            _, predicted_labels = torch.max(probas, 1)
            num_examples += targets.size(0)
            assert predicted_labels.size() == targets.size()
            correct_pred += (predicted_labels == targets).sum()

        return correct_pred.float() / num_examples * 100

    y_pred = []
    y_true = []

    def compute_acc_test(model, data_loader, device):
        correct_pred, num_examples = 0, 0
        model.eval()
        for i, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.to(device)

            logits, probas = model(features)
            _, predicted_labels = torch.max(probas, 1)
            num_examples += targets.size(0)
            assert predicted_labels.size() == targets.size()
            correct_pred += (predicted_labels == targets).sum()

            y_pred.extend(predicted_labels.cpu().numpy())
            y_true.extend(targets.cpu().numpy())

        return correct_pred.float() / num_examples * 100

    start_time = time.time()

    cost_list = []
    train_acc_list, valid_acc_list = [], []

    for epoch in range(NUM_EPOCHS):

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(DEVICE)
            targets = targets.to(DEVICE)

            ### FORWARD AND BACK PROP
            logits, probas = model(features)
            cost = F.cross_entropy(logits, targets)
            optimizer.zero_grad()

            cost.backward()

            ### UPDATE MODEL PARAMETERS
            optimizer.step()

            #################################################
            ### CODE ONLY FOR LOGGING BEYOND THIS POINT
            ################################################
            cost_list.append(cost.item())
            if not batch_idx % 150:
                print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d} | '
                      f'Batch {batch_idx:03d}/{len(train_loader):03d} |'
                      f' Cost: {cost:.4f}')

        model.eval()
        with torch.set_grad_enabled(False):  # save memory during inference

            train_acc = compute_acc(model, train_loader, device=DEVICE)
            valid_acc = compute_acc(model, valid_loader, device=DEVICE)

            # print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d}\n'
            #       f'Train ACC: {train_acc:.2f}')

            print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d}\n'
                  f'Train ACC: {train_acc:.2f} | Validation ACC: {valid_acc:.2f}')

            train_acc_list.append(train_acc.item())
            valid_acc_list.append(valid_acc.item())

        print(scheduler.get_last_lr())
        scheduler.step()

        elapsed = (time.time() - start_time) / 60
        print(f'Time elapsed: {elapsed:.2f} min')

    elapsed = (time.time() - start_time) / 60
    print(f'Total Training Time: {elapsed:.2f} min')

    plt.plot(cost_list, label='Minibatch cost')
    plt.plot(np.convolve(cost_list,np.ones(200, ) / 200, mode='valid'),label='Running average')

    plt.ylabel('Cross Entropy')
    plt.xlabel('Iteration')
    plt.legend()
    plt.show()

    plt.plot(np.arange(1, NUM_EPOCHS + 1, 1), train_acc_list, label='Training')
    plt.plot(np.arange(1, NUM_EPOCHS + 1, 1), valid_acc_list, label='Validation')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    with torch.set_grad_enabled(False):
        test_acc = compute_acc_test(model=model,
                                    data_loader=test_loader,
                                    device=DEVICE)

        valid_acc = compute_acc(model=model,
                                data_loader=valid_loader,
                                device=DEVICE)

    print(f'Validation ACC: {valid_acc:.2f}%')
    print(f'Test ACC: {test_acc:.2f}%')

    path = os.getcwd()
    # result_dir = "TB-CNN"
    # os.makedirs(result_dir)

    torch.save(model.state_dict(), os.path.join(path, "CNN-models/CNN-CV3-2.pth"))

    classes = ('normal', 'tuberculosis')
    # classes = ('covid', 'normal', 'viral pneumonia')
    # classes = ('covid', 'lung opacity', 'normal', 'viral pneumonia')

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix, classes, classes)
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True, cbar=False, fmt='g')
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.savefig('output_TB_augment-100-2_11-6-22.png')
    plt.show()


if __name__ == "__main__":
    main()