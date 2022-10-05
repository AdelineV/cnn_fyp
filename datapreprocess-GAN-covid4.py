import os
from torchvision import transforms
from PIL import Image
import random

augGANC= r"C:\Users\Adeline\PycharmProjects\cGAN\results\covid4-gan-v7\COVID"
augGANO= r"C:\Users\Adeline\PycharmProjects\cGAN\results\covid4-gan-v7\Lung_Opacity"
augGANN= r"C:\Users\Adeline\PycharmProjects\cGAN\results\covid4-gan-v7\Normal"
augGANV= r"C:\Users\Adeline\PycharmProjects\cGAN\results\covid4-gan-v7\Viral_Pneumonia"

saveaugC = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\datasets_clean\aug-GAN\covid4-10\train\COVID"
saveaugO = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\datasets_clean\aug-GAN\covid4-10\train\Lung_Opacity"
saveaugN = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\datasets_clean\aug-GAN\covid4-10\train\Normal"
saveaugV = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\datasets_clean\aug-GAN\covid4-10\train\Viral Pneumonia"

if not os.path.isdir(saveaugC):
    os.mkdir(saveaugC)

if not os.path.isdir(saveaugO):
    os.mkdir(saveaugO)

if not os.path.isdir(saveaugN):
    os.mkdir(saveaugN)

if not os.path.isdir(saveaugV):
    os.mkdir(saveaugV)

covid = []
for f in os.listdir(augGANC):
    if f.lower().endswith(('.png', '.jpg')):
        file_path = os.path.join(augGANC, f)
        img = Image.open(file_path)
        covid.append(img)

covid_img = random.sample(covid, 204)

for i in covid_img:
    filename = os.path.split(i.filename)[1]
    file_path = os.path.join(saveaugC, filename)

    i.save(file_path)

# lung = []
# for f in os.listdir(augGANO):
#     if f.lower().endswith(('.png', '.jpg')):
#         file_path = os.path.join(augGANO, f)
#         img = Image.open(file_path)
#         lung.append(img)
#
# lung_img = random.sample(lung, 200)
#
# for i in lung_img:
#     filename = os.path.split(i.filename)[1]
#     file_path = os.path.join(saveaugO, filename)
#
#     i.save(file_path)

# normal = []
# for f in os.listdir(augGANN):
#     if f.lower().endswith(('.png', '.jpg')):
#         file_path = os.path.join(augGANN, f)
#         img = Image.open(file_path)
#         normal.append(img)
#
# normal_img = random.sample(normal, 212)
#
# for i in normal_img:
#     filename = os.path.split(i.filename)[1]
#     file_path = os.path.join(saveaugN, filename)
#
#     i.save(file_path)
#
# viral = []
# for f in os.listdir(augGANV):
#     if f.lower().endswith(('.png', '.jpg')):
#         file_path = os.path.join(augGANV, f)
#         img = Image.open(file_path)
#         viral.append(img)
#
# viral_img = random.sample(viral, 184)
#
# for i in viral_img:
#     filename = os.path.split(i.filename)[1]
#     file_path = os.path.join(saveaugV, filename)
#
#     i.save(file_path)