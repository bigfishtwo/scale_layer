import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import os
import copy
import scipy
from skimage import io, transform
import math
from PIL import Image
from sklearn import metrics
from sklearn.model_selection import train_test_split
import albumentations as A

class DogCatDataset(Dataset):
    '''Dogs and Cats Dataset'''

    def __init__(self, root_dir, train, test, isdog, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.train = train
        self.test = test
        self.transform = transform
        self.labels = 'dogs' if isdog else 'cats'

    def __len__(self):
        dir = self.root_dir + '/'+self.labels
        # dir_cats = self.root_dir + '/cats'
        # TODO: change number of images
        return len(os.listdir(dir))
        # return 100
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        start = 0
        if not self.train and not self.test:
            start += 6000
        if self.test:
            start += 8000
        # i = np.random.randint(0,2)
        if self.labels=='dogs':
            img_name = self.root_dir+'/dogs/dog.'+str(index+start)+'.jpg'
            label = 0
        else:
            img_name = self.root_dir+ '/cats/cat.'+ str(index+start)+'.jpg'
            label = 1
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image=np.array(image))['image']
            image = torch.from_numpy(
                np.moveaxis(image / (255.0 if image.dtype == np.uint8 else 1), -1, 0).astype(np.float32))

        return image, label

class ISICDataset(Dataset):
    '''ISIC Dataset'''
    def __init__(self, root_dir, labels, transform):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Directory of label file
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # TODO: change number of images
        # return self.labels.shape[0]
        return 520

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = os.path.join(self.root_dir,
                                self.labels.iloc[index, 0])
        image = Image.open(img_name + '.jpg')
        label = self.labels.iloc[index, 1:].astype(int).to_numpy()
        label = np.argmax(label)
        if self.transform:
            image = self.transform(image=np.array(image))['image']
            image = torch.from_numpy(
                np.moveaxis(image / (255.0 if image.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
        return image, label

class ISICTest(Dataset):
    '''ISIC Dataset'''
    def __init__(self, root_dir, transform = None):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Directory of label file
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.file_names = os.listdir(self.root_dir)
        self.transform = transform


    def __len__(self):
        # TODO: change number of images
        return len(self.file_names)
        # return 100

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img_name = os.path.join(self.root_dir,
                                self.file_names[index])
        image = Image.open(img_name)
        # image = np.asarray(image).transpose((2,0,1))
        img_name = self.file_names[index].split('.')[0]

        if self.transform:
            image = self.transform(image)

        return img_name,image

class AlbumentationDataset(Dataset):
    def __init__(self, root_dir, csv_file, train,test, transform):

        self.root_dir = root_dir
        self.train = train
        self.test = test
        self.csv_file = pd.read_csv(csv_file)
        y_train, y_validation = train_test_split(self.csv_file, test_size=0.2, shuffle=False)
        y_val, y_test = train_test_split(y_validation, test_size=0.5, shuffle=False)
        if self.train:
            self.labels = y_train
        elif self.test:
            self.labels = y_val
        else:
            self.labels = y_test
        # TODO
        # self.labels = self.csv_file
        self.transform = transform

    def __len__(self):
        return self.labels.shape[0]
        # return 100

    def __getitem__(self, index):
        # if torch.is_tensor(index):
        #     index = index.tolist()

        img_name = os.path.join(self.root_dir,
                                self.labels.iloc[index, 0])
        image = Image.open(img_name+'.jpg')
        label = self.labels.iloc[index,1:].astype(int).to_numpy()
        label = np.argmax(label)
        if self.transform:
            # img_tensor = self.transform(image)
            image = self.transform(image=np.array(image))['image']
            image = torch.from_numpy(np.moveaxis(image / (255.0 if image.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
        return image, label

def run():
    # for multiprocessing
    torch.multiprocessing.freeze_support()

def get_device():
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def load_mnist(data_dir, batch_size):
    # Use standard FashionMNIST dataset
    train_set = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=False, #transforms.Resize(32),
        transform=transforms.Compose([
                                      transforms.ToTensor()
                                      ])
    )
    valid_set = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=False, #transforms.Resize(32),
        transform=transforms.Compose([
                                      transforms.ToTensor()])
    )
    test_set = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=False, #transforms.Resize(32),
        transform=transforms.Compose([
                                      transforms.ToTensor()])
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size,
                                              shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False)
    # Create training and validation dataloaders
    dataloaders_dict = {'train': train_loader,
                        'val': valid_loader,
                        'test': test_loader}

    return dataloaders_dict

def load_dogs(batch_size, resolution,phase):
    if phase == 'train':
        is_train = True
        is_test = False
        prefix = r'\train'
    elif phase == 'test':
        is_train = False
        is_test = True
        prefix = r'\test'
    else:
        is_train = False
        is_test = False
        prefix = r'\validation'

    data_transform = A.Compose([
            A.Resize(resolution[0], resolution[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=(-90,90)),
            A.RandomBrightnessContrast(),
        ])

    dogset = DogCatDataset(
        root_dir='D:\Subjects\PycharmProjects\Onmyoji\data\dogs_small'+prefix,
        # root_dir = './data/dogs_small'+prefix,
        train=is_train,
        test=is_test,
        isdog=True,
        transform= data_transform
    )
    catset = DogCatDataset(
        root_dir='D:\Subjects\PycharmProjects\Onmyoji\data\dogs_small' + prefix,
        train=is_train,
        test=is_test,
        isdog=False,
        transform=data_transform
    )

    dogcatset = torch.utils.data.ConcatDataset([dogset,catset])
    data_loader = torch.utils.data.DataLoader(dogcatset, batch_size=batch_size,
                                               shuffle=False, num_workers=1)
    return data_loader

def my_collate(batch):
    # load image with original size
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

def load_isic(batch_size, sampled = False):
    # load isic data set
    csv_file = './data/ISIC/ISIC_2019_Training_GroundTruth.csv'
    if sampled:
        csv_file = './data/ISIC/ISIC_2019_Training_Input/sampled_files.csv'
    csv_file = pd.read_csv(csv_file)
    y_train, y_validation = train_test_split(csv_file, test_size=0.2, shuffle=True)
    y_val, y_test = train_test_split(y_validation, test_size=0.5, shuffle=True)
    data_transform = A.Compose([
            # A.Resize(resolution,resolution),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=(-180,180)),
            A.RandomBrightnessContrast(),
        ])
    trainset = ISICDataset(
        root_dir='./data/ISIC/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
        labels=y_train,
        transform= data_transform
    )
    valset = ISICDataset(
        root_dir='./data/ISIC/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
        labels=y_val,
        transform= data_transform
    )
    testset = ISICDataset(
        root_dir='./data/ISIC/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
        labels=y_test,
        transform=data_transform
    )
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, collate_fn=my_collate,num_workers=1)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                              shuffle=True, collate_fn=my_collate,num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, collate_fn=my_collate,num_workers=1)
    dataloaders_dict = {'train': train_loader,
                        'resize': val_loader,
                        'val': val_loader,
                        'test': test_loader}
    return dataloaders_dict

def kfold_load_isic(labels,batch_size, resolution):
    # data loader for k-fold validation
    dataset = ISICDataset(
        root_dir ='./data/ISIC/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
        labels=labels,
        transform= A.Compose([
            A.Resize(resolution, resolution),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=(-90, 90)),
            A.RandomBrightnessContrast(),
        ])
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=1)
    return data_loader

def set_dataloader(dataset_name, batch_size, resolution):
    if dataset_name == 'mnist':
        data_dir = "./data/FashionMNIST"
        dataloaders_dict = load_mnist(data_dir, batch_size)
        # Number of classes in the dataset
        num_classes = 10

    elif dataset_name == 'dogs':
        dataloaders_dict = {'train': load_dogs(batch_size, resolution, 'train'),
                            'val': load_dogs(batch_size, resolution, 'val'),
                            'test': load_dogs(batch_size, resolution, 'test')}
        # dataloaders_dict, datalens_dict = load_dogs_separat(batch_size)
        # Number of classes in the dataset
        num_classes = 2

    elif dataset_name == 'isic':
        dataloaders_dict = {'train': load_isic(batch_size, resolution, 'train'),
                            'resize': load_isic(batch_size, resolution, 'resize'),
                            'val': load_isic(batch_size, resolution, 'val'),
                            'test': load_isic(batch_size, resolution, 'test')}
        # dataloaders_dict = load_isic(batch_size, resolution)
        num_classes = 9
    elif dataset_name == 'isic_sampled':
        dataloaders_dict = {'train': load_isic(batch_size, resolution, 'train', sampled=True),
                            'resize': load_isic(batch_size, resolution, 'resize',sampled=True),
                            'val': load_isic(batch_size, resolution, 'val', sampled=True),
                            'test': load_isic(batch_size, resolution, 'test', sampled=True)}
        num_classes = 9
    else:
        num_classes = 0
        dataloaders_dict = None
    return num_classes, dataloaders_dict


def show_img(img):
    # functions to show an image
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    print(npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def save_logs(hist,output_directory,dataset_name):
    '''
    function to save logs
    :param hist: dictionary type
    :param output_directory: output directory
    :param dataset_name: dataset_name
    :return: none, save logs to given directory
    '''
    hist_df = pd.DataFrame.from_dict(hist,orient='index').transpose()
    hist_df.to_csv(output_directory + '/history_'+dataset_name+'.csv', index=False)


def save_paras(paras,output_directory,dataset_name):
    # function to save parameters of training
    df = pd.DataFrame.from_dict(paras,orient='index',columns=['parameters'])
    df.to_csv(output_directory+ '/paras_'+dataset_name+'.csv', index=True)

def plot_loss_res(loss, res_x, res_y,output_directory,dataset_name,loss_type):
    '''
    plot of loss and resolution
    :param loss: can be train loss, valid loss and test loss
    :param res_x: resolution of x
    :param res_y: resolution of y
    :param loss_type: str, for naming
    :return:
    '''
    plt.title(loss_type+" loss and resolution")
    x, = plt.plot(res_x, loss,'o', label='x')
    y, = plt.plot(res_y,loss,'o',label='y')
    plt.legend(handles=[x,y])
    plt.xlabel("Resolution")
    plt.ylabel(loss_type)
    # TODO: change directory
    plt.savefig(output_directory+"/Loss_resolution_"+dataset_name+".png", bbox_inches='tight')
    plt.show()

def plot_scatter(dir):
    hist = pd.read_csv(dir)
    x = np.array(hist['resolution_x'])
    y = np.array(hist['resolution_y'])
    loss = np.array(hist['loss'])
    p = plt.scatter(x,y,c=loss,cmap=mpl.cm.cool)
    plt.xlabel('x')
    plt.ylabel('y')
    h = plt.colorbar(p)
    h.set_label('loss')
    plt.title('x, y, loss--dogs')
    plt.show()

def plot_roc(fpr,tpr, roc_auc, nums_class,output_directory,dataset_name):
    # plot toc curve
    plt.figure(figsize=(14, 8))
    lw = 2
    for i in range(nums_class):
        plt.plot(fpr[i], tpr[i], lw=lw,
                 label='ROC curve of class'+str(i)+'(area = %0.2f)' % roc_auc[i])

    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=lw)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=lw)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic ' + dataset_name)
    plt.legend(loc="lower right")
    plt.savefig(output_directory + "/ROC_curve_" + dataset_name + ".png", bbox_inches='tight')
    plt.show()

def calculate_f1(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred,average='weighted')

def calculate_metrics(y_true, y_pred):
    f1 = metrics.f1_score(y_true, y_pred, average='weighted')
    jaccard = metrics.jaccard_score(y_true,y_pred, average='weighted')
    accuracy = metrics.accuracy_score(y_true, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    return f1, jaccard,accuracy,confusion_matrix

def calculate_roc(y_true, y_score, num_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    one_hot_labels = np.zeros((y_true.shape[0],num_classes), dtype=int)
    for idx, j in enumerate(y_true):
        one_hot_labels[idx][j] = 1
    prediction = y_score.to_numpy()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(one_hot_labels[:, i], prediction[:, i])

        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    #
    # # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(one_hot_labels.ravel(), prediction.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += scipy.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc

def shades_of_gray(img, mink_norm=6):
    illum = []
    for channel in range(3):
        illum.append(np.power(np.mean(np.power(img[:,:,channel], mink_norm), (0,1)), 1 / mink_norm))
        # normalize estimated illumination
    som = np.sqrt(np.sum(np.power(illum, 2)))
    # som = np.power(np.prod(img.shape), 1 / mink_norm)
    illum = np.divide(illum, som)

    correcting_illum = illum * np.sqrt(3)
    corrected_image = img / 255.

    for channel in range(3):
        corrected_image[:, :, channel] /= correcting_illum[channel]
    return corrected_image

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    # inp = np.clip(inp, 0, 1)
    return inp

def visualize_transform(model, dataloader, resolution):
    with torch.no_grad():
        # Get a batch of training data
        # self.model.load_state_dict(self.best_model)

        data = next(iter(dataloader))[0]

        input_tensor = data[0]
        transformed_input_tensor = transform.resize(input_tensor, (3,resolution,resolution))

        in_grid = convert_image_np(
               torchvision.utils.make_grid(input_tensor))

        out_grid = transformed_input_tensor.transpose((1, 2, 0))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

        plt.ioff()
        plt.show()

def visualization_cnn(model, best_model):
    model.load_state_dict(best_model)

    model_weights = []  # we will save the conv layer weights in this list
    conv_layers = []  # we will save the conv layers in this list

    # get all the model children as list
    model_children = list(model.children())

    # counter to keep count of the conv layers
    counter = 0

    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                child = model_children[i][j]
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
    print(f"Total convolutional layers: {counter}")

    # visualize the first conv layer filters
    plt.figure(figsize=(20, 17))
    for i, filter in enumerate(model_weights[0]):
        plt.subplot(int(len(model_weights[0]) / 8), 8,
                    i + 1)  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        plt.imshow(filter[0, :, :].cpu().detach(), cmap='gray')
        plt.axis('off')
        # plt.savefig('../outputs/filter.png')
    plt.show()

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

def transform_fix(self, x):
    s = self.scale.item()
    theta = torch.tensor([[[1, 0, 0], [0, 1, 0]]],
                             dtype=torch.float, requires_grad=True)
    theta = torch.mul(theta, self.scale)
    scaled = int(self.resolution / abs(s))
    output = None
    grid = F.affine_grid(theta, [1, 3, self.resolution, self.resolution]).to('cuda')
    for img in x:
        img = F.grid_sample(img.unsqueeze(0), grid)
        output = torch.cat((output, img), 0) if output is not None else img
    x = output
    return x, scaled
