# Dataset: https://www.kaggle.com/datasets/polomarco/chest-ct-segmentation

from multiprocessing.dummy import freeze_support
import os
from random import random
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import segmentation_models_pytorch as smp
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset as BaseDataset

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if not torch.cuda.is_available():
  raise Exception("GPU not availalbe. CPU training will be too slow.")


DATA_DIR = './chest-ct-segmentation/'
csv_path = DATA_DIR + 'train.csv'
PRETRAINED_MODEL_PATH = DATA_DIR + 'best_model_ct.pth'
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['background',
            'lung', 'heart', 'trachea'
            ]

ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'
EPOCHS = 0

# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
loss = smp.utils.losses.DiceLoss() #smp.utils.losses.CrossEntropyLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5)#, ignore_channels=[0])
]
test_metrics = [
    smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[1, 2, 3]),
    smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0, 2, 3]),
    smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0, 1, 3]),
    smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0, 1, 2]),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.00001)#, betas=(0.9, 0.999)),
    #dict(params=model.parameters(), lr=0.001, betas=(0.9, 0.999)),
])

# load repo with data if it is not exists
if not os.path.exists(DATA_DIR):
    print('Data not exists')

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

class Dataset(BaseDataset):
    """Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        file_path (str): path to csv file
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """
    def __init__(
            self,
            file_path,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        df = pd.read_csv(file_path)
        self.x = df.iloc[1:, 0].values
        self.y = df.iloc[1:, 1].values
        self.CLASSES = CLASSES
        self.length = len(self.x)

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(DATA_DIR + 'images/' + self.x[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(DATA_DIR + 'masks/' + self.y[i])

        mask[mask < 240] = 0    # remove artifacts
        mask[mask > 0] = 1
        masks = [mask[:,:,v] >= 1 for v in range(3)]
        # extract certain classes from mask
        #masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # set background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((background, mask), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return self.length

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

dataset = Dataset(csv_path, CLASSES,     #augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),)

dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)    # 80%
val_size = int(dataset_size*0.1)        # 10%
test_size = dataset_size - train_size - val_size    # 10%

train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=1)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)

# create epoch runners
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

if __name__ == '__main__':
    freeze_support()
    # train model
    max_score = 0
    x_epoch_data = []
    train_loss = []
    train_iou_score = []
    valid_loss = []
    valid_iou_score = []
    try:
        model = torch.load(PRETRAINED_MODEL_PATH)
        print('Model loaded!')
    except:
        print('Train new model')
    for i in range(1, EPOCHS+1):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        x_epoch_data.append(i)
        train_loss.append(train_logs['dice_loss'])
        #train_loss.append(train_logs['cross_entropy_loss'])
        train_iou_score.append(train_logs['iou_score'])
        valid_loss.append(valid_logs['dice_loss'])
        #valid_loss.append(train_logs['cross_entropy_loss'])
        valid_iou_score.append(valid_logs['iou_score'])

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, PRETRAINED_MODEL_PATH)
            print('Model saved!')
        if i == 5:
            optimizer.param_groups[0]['lr'] = 0.000005
            print('Decrease decoder learning rate to 0.0001!')
        if i == 10:
            optimizer.param_groups[0]['lr'] = 0.000001
            print('Decrease decoder learning rate to 0.00005!')
        '''
        if i == 15:
            optimizer.param_groups[0]['lr'] = 0.00005
            print('Decrease decoder learning rate to 0.00001!')
        '''
    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    line1, = ax1.plot(x_epoch_data,train_loss,label='train')
    line2, = ax1.plot(x_epoch_data,valid_loss,label='validation')
    ax1.set_title("loss")
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.legend(loc='upper right')

    ax2 = fig.add_subplot(1, 2, 2)
    line1, = ax2.plot(x_epoch_data,train_iou_score,label='train')
    line2, = ax2.plot(x_epoch_data,valid_iou_score,label='validation')
    ax2.set_title("iou score")
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('iou_score')
    ax2.legend(loc='upper left')

    plt.show()

    # load best saved checkpoint
    best_model = torch.load(PRETRAINED_MODEL_PATH)

    # evaluate model on test set
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=test_metrics,
        device=DEVICE,
    )

    for i in range(len(CLASSES)):
        test_epoch.metrics[i].__name__ = 'iou_class_'+str(i)
    logs = test_epoch.run(test_dataloader)
    print("Per-Class IoU")
    iou_sum = 0
    for i in range(len(CLASSES)):
        iou_sum += logs['iou_class_'+str(i)]
        print(CLASSES[i]+':', logs['iou_class_'+str(i)])
    print("mIoU:", iou_sum / len(CLASSES))


    for i in range(10):
        n = np.random.choice(len(test_dataset))

        image_vis = test_dataset[n][0].astype('uint8')
        image_vis = np.transpose(image_vis, (1, 2, 0))
        image, gt_mask = test_dataset[n]

        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        gt_mask = np.transpose(gt_mask, (1, 2, 0))
        pr_mask = np.transpose(pr_mask, (1, 2, 0))

        gt_mask_color = np.zeros((gt_mask.shape[0],gt_mask.shape[1],3))
        for ii in range(1, gt_mask.shape[2]):
            gt_mask_color[:,:,ii-1] = gt_mask[:,:,ii]

        pr_mask_color = np.zeros((pr_mask.shape[0],pr_mask.shape[1],3))
        print('\nimage', i)
        for ii in range(1,pr_mask.shape[2]):
            if np.any(pr_mask[:,:,ii] >= 1):
                print(ii, CLASSES[ii], end = ' ')
            pr_mask_color[:,:,ii-1] = pr_mask[:,:,ii]

        cv2.imwrite(DATA_DIR + 'output/' + str(i) + '.jpg', pr_mask_color*255)
        visualize(
            image=image_vis,
            ground_truth_mask=gt_mask_color,
            predicted_mask=pr_mask_color
        )