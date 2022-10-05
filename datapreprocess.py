import os
from torchvision import transforms
from PIL import Image
import random

images = []
# dataroot = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\datasets_clean_old\tb_clean"
# savetest = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\data\augment\tuberculosis-5\test"
# savetrain = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\data\augment\tuberculosis-5\train"
# savetest = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\data\augment\tuberculosis-25\test"
# savetrain = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\data\augment\tuberculosis-25\train"
# savetest = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\data\augment\tuberculosis-50\test"
# savetrain = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\data\augment\tuberculosis-50\train"
# dataroot = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\datasets_clean\no-aug\tuberculosis-100\train"
# savetest = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\datasets_clean\no-aug\tuberculosis-100\test"
savetrain = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\datasets_clean\no-aug\tuberculosis-100\train"
saveaug = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\datasets_clean\aug-classic\tuberculosis-10\train"

# dataroot = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\datasets_clean_old\covid4_clean"
# savetest = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\datasets_clean\no-aug\covid4-100\test"
# savetrain = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\datasets_clean\no-aug\covid4-100\train"
# saveaug = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\datasets_clean\aug-classic\covid4-10\train"

# if not os.path.isdir(savetest):
#     os.mkdir(savetest)
#
# if not os.path.isdir(savetrain):
#     os.mkdir(savetrain)

if not os.path.isdir(saveaug):
    os.mkdir(saveaug)

for f in os.listdir(savetrain):
    sub_folder_path = os.path.join(savetrain, f)
    for file in os.listdir(sub_folder_path):
        if file.lower().endswith(('.png', '.jpg')):
            file_path = os.path.join(sub_folder_path, file)
            img = Image.open(file_path)
            images.append(img)

print(len(images))

# for train dataset resize
# train = []
# for f in os.listdir(savetrain):
#     sub_folder_path = os.path.join(savetrain, f)
#     for file in os.listdir(sub_folder_path):
#         if file.lower().endswith(('.png', '.jpg')):
#             file_path = os.path.join(sub_folder_path, file)
#             img = Image.open(file_path)
#             train.append(img)
#
# print(len(train))

# split into train-test
test = []

# split into train-test for images
# train = random.sample(images, int(len(images) * 0.80))
# for i in images:
#     if i not in train:
#         test.append(i)

print(len(test))
# split into train-test for sample_5
# sample_5 = random.sample(images, int(len(images) * 0.05))

# train = random.sample(images, int(len(images) * 0.05))

train = random.sample(images, int(len(images) * 0.1))
for i in train:
    label = os.path.split(os.path.split(i.filename)[0])[1]
    class_path = os.path.join(saveaug, label)

    if not os.path.isdir(class_path):
        os.mkdir(class_path)

    filename = os.path.split(i.filename)[1]
    file_path = os.path.join(class_path, filename)

    i.save(file_path)

# data augmentation methods
# tresize = transforms.Resize(size=(224,224))
# trhf = transforms.RandomHorizontalFlip(p=0.5)
# trvf = transforms.RandomVerticalFlip(p=0.5)
# trr90 = transforms.RandomRotation(degrees=90)
# trr180 = transforms.RandomRotation(degrees=180)
# trr270 = transforms.RandomRotation(degrees=270)

trr5 = transforms.RandomRotation(degrees=(-5, 5))
trah = transforms.RandomAffine(degrees=0, translate=(0.15, 0))
trav = transforms.RandomAffine(degrees=0, translate=(0.0, 0.15))
tras = transforms.RandomAffine(degrees=0, scale=(0.8, 1.2))

def rotate5(imgs):
    for i in imgs:
        label = os.path.split(os.path.split(i.filename)[0])[1]
        class_path = os.path.join(saveaug, label)

        if not os.path.isdir(class_path):
            os.mkdir(class_path)

        filename = os.path.split(i.filename)[1]
        file_path = os.path.join(class_path, "trr5-" + filename)
        i = trr5(i)

        i.save(file_path)


def translateHorizontal(imgs):
    for i in imgs:
        label = os.path.split(os.path.split(i.filename)[0])[1]
        class_path = os.path.join(saveaug, label)

        if not os.path.isdir(class_path):
            os.mkdir(class_path)

        filename = os.path.split(i.filename)[1]
        file_path = os.path.join(class_path, "trah-" + filename)
        i = trah(i)

        i.save(file_path)


def translateVertical(imgs):
    for i in imgs:
        label = os.path.split(os.path.split(i.filename)[0])[1]
        class_path = os.path.join(saveaug, label)

        if not os.path.isdir(class_path):
            os.mkdir(class_path)

        filename = os.path.split(i.filename)[1]
        file_path = os.path.join(class_path, "trav-" + filename)
        i = trav(i)

        i.save(file_path)


def translateScale(imgs):
    for i in imgs:
        label = os.path.split(os.path.split(i.filename)[0])[1]
        class_path = os.path.join(saveaug, label)

        if not os.path.isdir(class_path):
            os.mkdir(class_path)

        filename = os.path.split(i.filename)[1]
        file_path = os.path.join(class_path, "tras-" + filename)
        i = tras(i)

        i.save(file_path)


# random_trr5 = random.sample(train, int(len(train) * 0.5))
rotate5(train)

random_trah = random.sample(train, int(len(train) * 0.5))
translateHorizontal(random_trah)

random_trav = random.sample(train, int(len(train) * 0.5))
translateVertical(random_trav)

random_tras = random.sample(train, int(len(train) * 0.5))
translateScale(random_tras)



# train = []
# for f in os.listdir(savetrain):
#     sub_folder_path = os.path.join(savetrain, f)
#     for file in os.listdir(sub_folder_path):
#         if file.lower().endswith(('.png', '.jpg')):
#             file_path = os.path.join(sub_folder_path, file)
#             img = Image.open(file_path)
#             train.append(img)
#
# # random horizontal flip
# random_trhf = random.sample(train, int(len(train) * 0.5))
# horizontalFlip(random_trhf)
# # random vertical flip
# random_trvf = random.sample(train, int(len(train) * 0.5))
# verticalFlip(random_trvf)
# # random rotate 90
# random_trr90 = random.sample(train, int(len(train) * 0.5))
# random90(random_trr90)
# # random rotate 180
# random_trr180 = random.sample(train, int(len(train) * 0.5))
# random180(random_trr180)
# # random rotate 270
# random_trr270 = random.sample(train, int(len(train) * 0.5))
# random270(random_trr270)


# construct test set
# for i in test:
#     label = os.path.split(os.path.split(i.filename)[0])[1]
#     class_path = os.path.join(savetest, label)
#
#     if not os.path.isdir(class_path):
#         os.mkdir(class_path)
#
#     filename = os.path.split(i.filename)[1]
#     file_path = os.path.join(class_path, filename)
#     i = tresize(i)
#
#     i.save(file_path)
#
# print(len(train))
# print(len(test))