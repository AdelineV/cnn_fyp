import os
from PIL import Image
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import random


dataroot = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\data\covid_four"
savecovid = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\datasets_clean\covid_four_clean"
covid_f = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\data\covid_four\COVID"
lung_opacity = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\data\covid_four\Lung_Opacity"
normal_f = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\data\covid_four\Normal"
viral = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\data\covid_four\Viral Pneumonia"

savetb = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\datasets_clean\tb_clean"
tuberculosis = r"C:\Users\Adeline\PycharmProjects\cnn_fyp\data\tuberculosis\Normal"

if not os.path.isdir(savecovid):
    os.mkdir(savecovid)

if not os.path.isdir(savetb):
    os.mkdir(savetb)

count = 0

covid = []
lung_opa = []
normal = []
viral_pneu = []
tb_normal = []

# for f in os.listdir(covid_f):
#     if f.lower().endswith(('.png', '.jpg')):
#         file_path = os.path.join(covid_f, f)
#         img = Image.open(file_path)
#         covid.append(img)
#
# # choose 1000 random sample
# sample = random.sample(covid, 1000)
#
# for i in sample:
#     label = os.path.split(os.path.split(i.filename)[0])[1]
#     class_path = os.path.join(savecovid, label)
#
#     if not os.path.isdir(class_path):
#         os.mkdir(class_path)
#
#     filename = os.path.split(i.filename)[1]
#     file_path = os.path.join(class_path, filename)
#
#     i.save(file_path)

# for f in os.listdir(lung_opacity):
#     if f.lower().endswith(('.png', '.jpg')):
#         file_path = os.path.join(lung_opacity, f)
#         img = Image.open(file_path)
#         lung_opa.append(img)
#
# # choose 1000 random sample
# sample = random.sample(lung_opa, 1000)
#
# for i in sample:
#     label = os.path.split(os.path.split(i.filename)[0])[1]
#     class_path = os.path.join(savecovid, label)
#
#     if not os.path.isdir(class_path):
#         os.mkdir(class_path)
#
#     filename = os.path.split(i.filename)[1]
#     file_path = os.path.join(class_path, filename)
#
#     i.save(file_path)

# for f in os.listdir(normal_f):
#     if f.lower().endswith(('.png', '.jpg')):
#         file_path = os.path.join(normal_f, f)
#         img = Image.open(file_path)
#         normal.append(img)
#
# # choose 1000 random sample
# sample = random.sample(normal, 1000)
#
# for i in sample:
#     label = os.path.split(os.path.split(i.filename)[0])[1]
#     class_path = os.path.join(savecovid, label)
#
#     if not os.path.isdir(class_path):
#         os.mkdir(class_path)
#
#     filename = os.path.split(i.filename)[1]
#     file_path = os.path.join(class_path, filename)
#
#     i.save(file_path)

# for f in os.listdir(viral):
#     if f.lower().endswith(('.png', '.jpg')):
#         file_path = os.path.join(viral, f)
#         img = Image.open(file_path)
#         viral_pneu.append(img)
#
# # choose 1000 random sample
# sample = random.sample(viral_pneu, 1000)
#
# for i in sample:
#     label = os.path.split(os.path.split(i.filename)[0])[1]
#     class_path = os.path.join(savecovid, label)
#
#     if not os.path.isdir(class_path):
#         os.mkdir(class_path)
#
#     filename = os.path.split(i.filename)[1]
#     file_path = os.path.join(class_path, filename)
#
#     i.save(file_path)

for f in os.listdir(tuberculosis):
    if f.lower().endswith(('.png', '.jpg')):
        file_path = os.path.join(tuberculosis, f)
        img = Image.open(file_path)
        tb_normal.append(img)

# choose 1000 random sample
sample = random.sample(tb_normal, 700)

for i in sample:
    label = os.path.split(os.path.split(i.filename)[0])[1]
    class_path = os.path.join(savetb, label)

    if not os.path.isdir(class_path):
        os.mkdir(class_path)

    filename = os.path.split(i.filename)[1]
    file_path = os.path.join(class_path, filename)

    i.save(file_path)

# for f in os.listdir(dataroot):
#     sub_folder_path = os.path.join(dataroot, f)
#     for file in os.listdir(sub_folder_path):
#         if file.lower().endswith(('.png', '.jpg')):
#             if count == 0:
#                 file_path = os.path.join(sub_folder_path, file)
#                 img = Image.open(file_path)
#                 covid.append(img)
#             elif count == 1:
#                 file_path = os.path.join(sub_folder_path, file)
#                 img = Image.open(file_path)
#                 lung_opa.append(img)
#             elif count == 2:
#                 file_path = os.path.join(sub_folder_path, file)
#                 img = Image.open(file_path)
#                 normal.append(img)
#             elif count == 3:
#                 file_path = os.path.join(sub_folder_path, file)
#                 img = Image.open(file_path)
#                 viral_pneu.append(img)
#     count += 1

print(len(covid), len(lung_opa), len(normal), len(viral_pneu), len(tb_normal))