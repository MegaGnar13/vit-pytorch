import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import one_hot_vector
import matplotlib.pyplot as plt
import sys


class my_dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # self.labels = os.listdir(self.root)
        self.img_list = []
        self.label_list = []

        for label in os.listdir(self.root):
            files = os.listdir(os.path.join(self.root, label))
            for f in files:
                self.img_list.append(f)
                self.label_list.append(label)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.label_list[idx], self.img_list[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)

        #img는 transform된 이미지
        #label은 폴더 name

        label = self.label_list[idx]

        #vectors 는 label을 one-hot vector
        vectors = one_hot_vector.vector

        #index는 몇번 째 폴더에 이미지가 있는가
        label_index = one_hot_vector.items
        index = label_index.index(label)

        new_label = vectors[index]
        return img, index


if __name__ == '__main__':
    transforms = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transforms = transforms.Compose([transforms.Resize((224, 224)),
    #                                  transforms.ToTensor()])
    trainset = my_dataset(root='/data/dongkyu/ILSVRC2012/train', transforms=transforms)
    valset = my_dataset(root='/data/dongkyu/ILSVRC2012/val', transforms=transforms)

    traindl = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
    valdl = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True, num_workers=4)

    for idx, (img, label, index) in enumerate(traindl):
        print(f"img: {img}")
        print(f"label: {label}")
        print(f"index: {index}")

        # input = img
        # target = label
        # print(input.shape, target.shape, label_index)
        #
        # input = input.squeeze().permute(1, 2, 0).cpu().numpy()
        # plt.figure()
        # plt.imshow(input)
        # plt.title(f'One-hot: {label_index[0]} | Name: {name[0]}')
        # plt.savefig('./test.png')
        # sys.exit()