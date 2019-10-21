import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np

from get_link_idol.crawler.control import RAW_DATA_PATH, PROCESSED_DATA_PATH


def get_dataloader(mtcnn,
                   batch_size: int,
                   test_ratio=0.2,
                   valid_ratio=0.2,
                   random_state=420,
                   shuffle=True,
                   num_workers=4,
                   pin_memory=False):
    crop_face(mtcnn, batch_size, num_workers, pin_memory)
    src_folder = os.path.join(PROCESSED_DATA_PATH, 'cropped')

    transforms_img = transforms.Compose([transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                                         transforms.RandomRotation(degrees=15),
                                         transforms.ColorJitter(),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.CenterCrop(size=224),  # Image net standards
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])
                                         ])
    dataset = datasets.ImageFolder(src_folder, transform=transforms_img)

    train_set, test_set = train_test_split(np.arange(len(dataset)),
                                           test_size=test_ratio,
                                           random_state=random_state,
                                           shuffle=shuffle,
                                           stratify=dataset.targets)
    train_set, valid_set = train_test_split(np.arange(len(train_set)),
                                            test_size=valid_ratio,
                                            random_state=random_state,
                                            shuffle=shuffle,
                                            stratify=train_set.targets)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                              collate_fn=collate_pil)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                              collate_fn=collate_pil)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                             collate_fn=collate_pil)

    return train_loader, valid_loader, test_loader


def crop_face(mtcnn,
              batch_size: int,
              num_workers=4,
              pin_memory=False):
    dest_folder = os.path.join(PROCESSED_DATA_PATH, 'cropped')
    src_folder = RAW_DATA_PATH

    dataset = datasets.ImageFolder(src_folder, transform=transforms.Resize((1024, 1024)))
    dataset.samples = [(p, p.replace(src_folder, dest_folder)) for p, _ in dataset.samples]

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        collate_fn=collate_pil)

    for i, (x, y) in enumerate(loader):
        print('\rImages processed: {:8d} of {:8d}'.format(i + 1, len(loader)), end='')
        mtcnn(x, save_path=y)
        print((x, y))


def collate_pil(x):
    out_x, out_y = [], []
    for xx, yy in x:
        out_x.append(xx)
        out_y.append(yy)
    return out_x, out_y
