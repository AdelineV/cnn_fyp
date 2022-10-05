import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from models import DenseNet121

NUM_CLASSES = 3
GRAYSCALE = True
BATCH_SIZE = 16

DEVICE = "cuda:0"
# PATH = "CNN-models/CNN-TB-3.pth"
# PATH = "CNN-models/CNN-CV4-1.pth"
PATH = "CNN-models/CNN-CV3-1.pth"

# classes = ('normal', 'tuberculosis')
# classes = ('covid', 'lung opacity', 'normal', 'viral pneumonia')
# classes = ('covid', 'normal', 'viral pneumonia')

# datatest = "datasets_clean/demo/TB-test"
# datatest = "datasets_clean/demo/CV4-test"
datatest = "datasets_clean/demo/CV3-test"

y_pred = []
y_true = []


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

    def compute_acc(model, data_loader, device):
        correct_pred, num_examples = 0, 0
        model.eval()
        for i, (features, targets) in enumerate(data_loader):
            logits, probas = model(features)
            _, predicted_labels = torch.max(probas, 1)
            num_examples += targets.size(0)
            assert predicted_labels.size() == targets.size()
            correct_pred += (predicted_labels == targets).sum()

            y_pred.extend(predicted_labels.cpu().numpy())
            y_true.extend(targets.cpu().numpy())

            accuracy = correct_pred.float() / num_examples * 100
            accuracy = accuracy.item()

        return accuracy

    print(compute_acc(model, test_loader, device=DEVICE))


if __name__ == "__main__":
    demo()