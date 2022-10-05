import torch
import torchvision.utils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import random

from models import DenseNet121

NUM_CLASSES = 4
GRAYSCALE = True
BATCH_SIZE = 16

DEVICE = "cuda:0"
# PATH = "CNN-models/CNN-TB-3.pth"
PATH = "CNN-models/CNN-CV4-1.pth"

# classes = ('normal', 'tuberculosis')
# classes = ('covid', 'normal', 'viral pneumonia')
classes = ('covid', 'lung opacity', 'normal', 'viral pneumonia')

# datatest = "datasets_clean/demo/TB-test"
# datatest = "datasets_clean/demo/CV3-test"
datatest = "datasets_clean/demo/CV4-test"


def demo():
    test_dataset = datasets.ImageFolder(root=datatest,
                                        transform=transforms.Compose([
                                            transforms.Resize(size=(224, 224)),
                                            transforms.Grayscale(1),
                                            transforms.ToTensor()
                                            ]))

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             num_workers=0,
                             shuffle=True)

    model = DenseNet121(num_classes=NUM_CLASSES, grayscale=GRAYSCALE, drop_rate=0.1)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    # image_transforms = transforms.Compose([
    #     transforms.Resize(size=(224, 224)),
    #     transforms.Grayscale(1),
    #     transforms.ToTensor()
    # ])

    def compute_acc(models, data_loader):
        models = models.eval()

        features, labels = next(iter(data_loader))

        plt.figure(figsize=(15, 15))

        for i in range(9):
            plt.subplot(3, 3, i+1)
            plt.imshow(np.squeeze(features[i]))

            logits, probas = models(features)
            _, predicted_labels = torch.max(probas, 1)

            actual_class = classes[labels[i]]

            # print(classes[predicted_labels.item()])

            plt.title(f"Actual : {actual_class},\n Predicted:{classes[predicted_labels[i].item()]}.")

            plt.axis("off")
        plt.show()

    compute_acc(model, test_loader)


if __name__ == "__main__":
    demo()