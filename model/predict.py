import os
import json
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import csv
from densenet import *
# from model import resnet50


activation = {}

def get_activation(name):
    def hook(model, input, output):
        # 如果你想feature的梯度能反向传播，那么去掉 detach（）
        activation[name] = output.detach()

    return hook

def main():
    img_root = './data/T4-jipangji-predict'
    images = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose(
        [  # transforms.CenterCrop(size=(400,1000)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    # load image
    for root, dirs, files in os.walk(img_root):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':  # 判断，只记录jpg
                images.append(os.path.join(root, file))

    # model = resnet50(num_classes=2, include_top=True).to(device)
    model = densenet121().to(device)
    weights_path = "./denseNet-zhuanyi-BSE-T11.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    with torch.no_grad():
        # predict class
        for img_path in images:
            list = []
            img = cv2.imread(img_path)
            img = np.array(img)
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            img = data_transform(img)
            img = torch.unsqueeze(img, dim=0)
            features = torch.FloatTensor(1, 17)
            model.fc.register_forward_hook(get_activation('fc'))
            #            output = torch.squeeze(model(img.to(device),features.to(device)))
            output = torch.squeeze(model(img.to(device)))
            threshold = 0
            output = torch.where(output >= threshold, torch.ones_like(output), output)
            output = torch.where(output < threshold, torch.zeros_like(output), output)
            # predict_y = torch.max(output, dim=0)[1].int().item()
            i = img_path[27:]  # tests set 27 train set 23
            i = int(i[:-4])  #
            list.append(i)
            tensor = activation['fc'].cpu()  # .tolist()
            tensor = torch.squeeze(tensor)
            t = tensor[0].float().item()
            for x in range(0, len(tensor)):
                list.append(tensor[x].float().item())
            list.append(output.int().item())
            with open("./Result/tb_densenet121/T11-pre.csv", "a", newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(list)


if __name__ == '__main__':
    main()
