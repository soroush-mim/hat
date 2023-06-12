import torch
from PIL import Image
import torchvision
import torchvision.transforms as transforms


DATA_DESC = {
    'data': 'cifar10',
    'classes': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    'num_classes': 10,
    'mean': [0.4914, 0.4822, 0.4465], 
    'std': [0.2023, 0.1994, 0.2010],
}


def load_cifar10(data_dir, use_augmentation=False, prime_data = False):
    """
    Returns CIFAR10 train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (bool): whether to use augmentations for training set.
    Returns:
        train dataset, test dataset. 
    """

    print('loading cifar10...')
    test_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5), 
                                              transforms.ToTensor()])
    else: 
        train_transform = test_transform

    if prime_data:
        train_dataset = CIFAR10_prime(root=data_dir, train=True, download=True, transform=train_transform)
    else:
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    print('cifar10 loaded')    
    return train_dataset, test_dataset


class CIFAR10_prime(torchvision.datasets.CIFAR10):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.data_prime = None

    def add_data_prime(self,data_prime):
        self.data_prime = data_prime

        
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.data_prime is not None:
            return img, target, index, self.data_prime[str(index)]
        else:
            return img, target, index