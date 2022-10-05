import os
from torchvision import transforms
from PIL import Image
import random

augGANnormal = r"C:\Users\Adeline\PycharmProjects\cGAN\results\TB-gan-v4\augment\normalv3"
augGANtb = r"C:\Users\Adeline\PycharmProjects\cGAN\results\TB-gan-v4\augment\tuberculosisv2"

saveaugN = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\datasets_clean\aug-GAN\tuberculosis-10\train\normal"
saveaugT = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\datasets_clean\aug-GAN\tuberculosis-10\train\tuberculosis"

if not os.path.isdir(saveaugN):
    os.mkdir(saveaugN)

if not os.path.isdir(saveaugT):
    os.mkdir(saveaugT)

normal = []
for f in os.listdir(augGANnormal):
    if f.lower().endswith(('.png', '.jpg')):
        file_path = os.path.join(augGANnormal, f)
        img = Image.open(file_path)
        normal.append(img)

normal_img = random.sample(normal, 138)

for i in normal_img:
    filename = os.path.split(i.filename)[1]
    file_path = os.path.join(saveaugN, filename)

    i.save(file_path)

tb = []
for f in os.listdir(augGANtb):
    if f.lower().endswith(('.png', '.jpg')):
        file_path = os.path.join(augGANtb, f)
        img = Image.open(file_path)
        tb.append(img)

tb_img = random.sample(tb, 142)

for i in tb_img:
    filename = os.path.split(i.filename)[1]
    file_path = os.path.join(saveaugT, filename)

    i.save(file_path)