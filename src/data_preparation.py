# xai_project/data_preparation.py
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class TumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.classes = ['neg', 'pos'] # as per the pancreas dataset, Binary classification
        self.image_paths = []
        self.labels = []
        
        # from the specified path access images & labels
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, 'image', class_name)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# to transform
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# initializing dataset & data loader
your_dataset_path = 'data_pancreas/'
#data preparation
dataset = TumorDataset(root_dir=your_dataset_path, transform=train_transform)
#preparing to train the model , as data loader
train_loader = DataLoader(dataset, batch_size=4, shuffle=True) #16