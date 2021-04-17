import torch
import cv2
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, csv, train, validation, test):

        self.csv = csv
        self.train = train
        self.validation = validation
        self.test = test
        self.all_image_names = self.csv[:]['Id']
        self.all_labels = np.array(self.csv.drop(['Id', 'Genre'], axis=1))
        self.train_len = int(0.85*len(self.csv))
        self.valid_len = len(self.csv) - self.train_len

        # set the training data images and labels
        if self.train == True:
            print(f"Number of training images: {self.train_len}")
            self.image_names = list(self.all_image_names[:self.train_len])
            self.labels = list(self.all_labels[:self.train_len])

            # define the training transform
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((400, 400)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ToTensor(),
            ])

        elif self.validation == True:
            print(f"Number of Validation images: {self.valid_len}")
            self.image_names = list(self.all_image_names[-self.valid_len:-10])
            self.labels = list(self.all_labels[-self.valid_len:])

            # define the validatio transform
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((400, 400)),
                transforms.ToTensor(),
            ])

        elif self.test == True:
            print("Number of Test images: 10")
            self.image_names = list(self.all_image_names[-10:])
            self.labels = list(self.all_labels[-10:])

            # define the test transform
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):

        image = cv2.imread(
            f"../input/movie-classifier/Multi_Label_dataset/Images/{self.image_names[index]}.jpg")

        # convert it from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # apply transforms
        image = self.transform(image)
        targets = self.labels[index]

        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'label': torch.tensor(targets, dtype=torch.float32)
        }
