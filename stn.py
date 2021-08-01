from utils import *
import math
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from torch.utils.tensorboard import SummaryWriter

torch.cuda.empty_cache()

class Net(nn.Module):
    def __init__(self, model_name, resolution,s,device, use_pretrained=False):
        super(Net, self).__init__()
        # TODO: pretrained
        if model_name.startswith("efficientnet"):
            if use_pretrained:
                self.features = EfficientNet.from_pretrained("efficientnet-b0", num_classes=9)
            else:
                self.features = EfficientNet.from_name("efficientnet-b0")
                nums_feature = self.features._fc.in_features
                self.features._fc = nn.Linear(in_features=nums_feature, out_features=9)
        elif model_name == "alexnet":
            self.features = models.alexnet(pretrained=use_pretrained)
            self.features.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.features.classifier = nn.Linear(256, 9)
        elif model_name == "resnet18":
            self.features = models.resnet18(pretrained=use_pretrained)
            self.features.fc = nn.Linear(512, 9, bias=True)
        elif model_name == "resnet":
            self.features = models.resnet50(pretrained=use_pretrained)
            self.features.fc = nn.Linear(2048, 9, bias=True)

        self.scale = torch.tensor([s],requires_grad=True, dtype=torch.float)
        self.resolution = resolution
        self.device = device

    # transformer network forward function
    def transform(self, x):
        s = 1/(self.scale.item())
        theta = torch.tensor([[[s,0,0],[0,s,0]]],
                             dtype=torch.float, requires_grad=True)
        theta = torch.mul(theta, self.scale)

        scaled = int(abs(self.scale.item())* self.resolution)
        scaled = max(scaled, 32)
        scaled = min(scaled, 650)

        output = None
        grid1 = F.affine_grid(theta, [1, 3, scaled, scaled]).to(self.device)
        for img in x:
            img = F.grid_sample(img.unsqueeze(0), grid1)
            output = torch.cat((output, img),0) if output is not None else img
        x = output
        return x, scaled

    def forward(self, x):
        # transform the input
        x,scaled = self.transform(x)

        # Perform the usual forward pass
        x = self.features(x)
        return x, scaled, self.scale

class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

class Model:
    def __init__(self,model_name, dataset_name, batch_size, resolution,scale, output_directory,device, learning_rate=1e-5,pretrained=False):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.resolution = resolution
        self.learning_rate = learning_rate
        self.output_directory = output_directory
        self.num_classes=9

        self.dataloader = load_isic(batch_size)
        self.scale = scale
        self.model = Net(model_name, resolution[0],scale, device, pretrained).to(device)
        self.set_optimizer()
        self.set_loss()

        self.phase = None
        self.best_acc = 0.0
        self.best_resolution = [float('inf'), self.resolution]
        self.best_model = {'model_state_dict':copy.deepcopy(self.model.state_dict()),
                           'transform_state_dict':self.model.scale,
                           'optimizer_sgd_state_dict': copy.deepcopy(self.optimizer.optimizers[0].state_dict()),
                           'optimizer_adam_state_dict': copy.deepcopy(self.optimizer.optimizers[1].state_dict()),
                           }
        self.last_model = {'model_state_dict': copy.deepcopy(self.model.state_dict()),
                           'transform_state_dict': self.model.scale,
                           'optimizer_sgd_state_dict': copy.deepcopy(self.optimizer.optimizers[0].state_dict()),
                           'optimizer_adam_state_dict': copy.deepcopy(self.optimizer.optimizers[1].state_dict()),
                           }


    def set_optimizer(self):
        # Observe that all parameters are being optimized
        # TODO: sgd learning rate
        self.optimizer = MultipleOptimizer(optim.SGD([self.model.scale], lr=0.0001),
                                optim.Adam(self.model.parameters(), lr=self.learning_rate))
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.Adam([
        #     {'params': self.model.parameters()},
        #     {'params': self.model.scale, 'lr':1e-3}
        # ], lr=self.learning_rate)
        # TODO: disable
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer.optimizers[0], mode='min',patience=5, verbose=True)

    def set_loss(self):
        # Setup the loss fxn
        self.criterion = nn.CrossEntropyLoss(reduction='sum')


    def stop_train(self, hist, threshold):
        if len(hist) < threshold:
            return False
        sub = hist[(-1) * threshold:]
        # count = hist.iloc[(-1) * threshold:][0].duplicated().sum()
        return all([abs(sub[i+1] -sub[i]) < 3 for i in range(len(sub) - 1)])

    def resize_img(self, img):
        # functions to resize an image
        img = [transform.resize(i, (3, self.resolution[0],self.resolution[1])) for i in img]
        img = torch.tensor(img)
        img = img.to(self.device)
        return img

    def train(self, num_epochs):
        history = {'train_loss': [],
                   'train_acc': [],
                   'train_scale': [],
                   'train_s':[],
                   'val_loss':[],
                   'val_acc': [],
                   'val_scale': [],
                   'val_s':[]
                   }
        since = time.time()
        for epoch in tqdm(range(num_epochs)):
            print('Epoch:{}'.format(epoch))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                running_scales = 0
                running_s = 0

                for batch_idx, (data, target) in enumerate(self.dataloaders[phase]):
                    data = [d.to(self.device) for d in data]
                    target = target.to(self.device)

                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(True):
                        output, scale, s = self.model(data)
                        # try:
                        #     output, scale = self.model(data, previous_scale)
                        # except RuntimeError as exception:
                        #     if "out of memory" in str(exception):
                        #         print("WARNING: out of memory")
                        #         if hasattr(torch.cuda, 'empty_cache'):
                        #             torch.cuda.empty_cache()
                        #     else:
                        #         raise exception
                        loss = self.criterion(output, target)
                        output = F.softmax(output, dim=1)
                        _, pred = torch.max(output, 1)

                        if phase == 'train':
                            s.retain_grad()
                            # backward + optimize only if in training phase
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item()
                    running_corrects += torch.sum(pred == target.data).item()
                    running_scales += scale * len(data)
                    running_s += s.item()* len(data)


                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = running_corrects / len(self.dataloaders[phase].dataset)
                epoch_scale = int(running_scales / len(self.dataloaders[phase].dataset))
                epoch_s = running_s / len(self.dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                print('Resolution:{} Scale:{:.4f} grad:{:.4f}'.format(int(epoch_scale), epoch_s, s.grad.item()))

                if phase == 'train':
                    history['train_loss'].append(epoch_loss)
                    history['train_acc'].append(epoch_acc)
                    history['train_scale'].append(epoch_scale)
                    history['train_s'].append(epoch_s)
                else:
                    history['val_loss'].append(epoch_loss)
                    history['val_acc'].append(epoch_acc)
                    history['val_scale'].append(epoch_scale)
                    history['val_s'].append(epoch_s)
                    self.last_model = {'model_state_dict': copy.deepcopy(self.model.state_dict()),
                                       'transform_state_dict': self.model.scale,
                                       'optimizer_sgd_state_dict': copy.deepcopy(
                                           self.optimizer.optimizers[0].state_dict()),
                                       'optimizer_adam_state_dict': copy.deepcopy(
                                           self.optimizer.optimizers[1].state_dict()),
                                       }
                    if epoch_loss < self.best_resolution[0]:
                        self.best_resolution = [epoch_loss, [epoch_scale, epoch_scale]]
                        self.best_model = {'model_state_dict':copy.deepcopy(self.model.state_dict()),
                                            'transform_state_dict':self.model.scale,
                                            'optimizer_sgd_state_dict': copy.deepcopy(self.optimizer.optimizers[0].state_dict()),
                                            'optimizer_adam_state_dict': copy.deepcopy(self.optimizer.optimizers[1].state_dict()),
                                           }
            # if self.stop_train(history['val_scale'],5):
            #     break
            pd.DataFrame(history).to_csv(
                self.output_directory + '_' + 'history.csv')
            torch.save(self.last_model, self.output_directory + '_' + 'last_weight.pth')
            torch.save(self.best_model, self.output_directory + '_'+ 'best_weight.pth')

        time_elapsed = time.time() - since
        plt.figure()
        plt.plot(history['train_scale'])

        return time_elapsed

    def test(self):
        preds_all = None
        labels_all = None
        outputs_all = None

        with torch.no_grad():
            self.model.load_state_dict(self.best_model["model_state_dict"])
            self.model.scale = self.best_model["transform_state_dict"]
            self.optimizer.optimizers[0].load_state_dict(self.best_model["optimizer_sgd_state_dict"])
            self.optimizer.optimizers[1].load_state_dict(self.best_model["optimizer_adam_state_dict"])

            self.model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_scale = 0
            running_s = 0

            for batch_idx, (data, target) in enumerate(self.dataloaders['test']):
                if labels_all is None:
                    labels_all = target
                else:
                    labels_all = torch.cat((labels_all, target), 0)

                data = [d.to(self.device) for d in data]
                target = target.to(self.device)
                output, scale, s = self.model(data)

                # sum up batch loss
                running_loss += self.criterion(output, target).item()
                # get the index of the max log-probability
                output = F.softmax(output, dim=1)
                _, pred = torch.max(output, 1)

                running_corrects += torch.sum(pred == target.data).item()
                running_scale += scale * len(data)
                running_s += s.item() * len(data)

                if outputs_all is None:
                    outputs_all = output.cpu()
                else:
                    outputs_all = torch.cat((outputs_all, output.cpu()), 0)
                if preds_all is None:
                    preds_all = pred.cpu()
                else:
                    preds_all = torch.cat((preds_all, pred.cpu()), 0)


            epoch_loss = running_loss / len(self.dataloaders['test'].dataset)
            epoch_accuracy = running_corrects / len(self.dataloaders['test'].dataset)
            epoch_scale = int(running_scale / len(self.dataloaders['test'].dataset))
            epoch_s = running_s / len(self.dataloaders['test'].dataset)

            f1_score, jaccard, accuracy, confusion_matrix = calculate_metrics(labels_all, preds_all)
            rescaled = int(epoch_scale)

            print('\nTest set: Average loss: {:.4f}'.format(epoch_loss))
            print('Scale: {:.4f}'.format(rescaled))
            print('test F1 score: {:4f} Jaccard:{:4f} Accuracy:{:4f}'.format(f1_score, jaccard, accuracy))
            print('Confusion Matrix:\n', confusion_matrix)

            pd.DataFrame(confusion_matrix).to_csv(
                self.output_directory + '_'+'confus.csv')
            pd.DataFrame({'labels': labels_all,
                          'predictions': preds_all}).to_csv(
                self.output_directory + '_' +'pred.csv')
            pd.DataFrame(outputs_all.cpu().numpy()).to_csv(
                self.output_directory + '_'+'outputs.csv')

            return epoch_loss,accuracy,jaccard,f1_score,epoch_scale,epoch_s

    def convert_image_np(self,inp):
        """Convert a Tensor to numpy image."""
        inp = inp.numpy().transpose((1, 2, 0))
        return inp

    def visualize_stn(self):
        with torch.no_grad():
            # Get a batch of training data
            self.model.load_state_dict(self.best_model["model_state_dict"])
            # self.model.scale = self.best_model["transform_state_dict"]

            data = next(iter(self.dataloader['test']))[0]
            # data = self.resize_img(data)
            input_tensor = [d.to(self.device) for d in data]
            transformed_input_tensor, scaled = self.model.transform(input_tensor)

            transformed_input_tensor = transformed_input_tensor.cpu()

            in_grid = self.convert_image_np(
                torchvision.utils.make_grid(data[0]))

            out_grid = self.convert_image_np(
                torchvision.utils.make_grid(transformed_input_tensor[0]))

            # Plot the results side-by-side
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(in_grid)
            axarr[0].set_title('Dataset Images')

            axarr[1].imshow(out_grid)
            axarr[1].set_title('Transformed Images')

            print("transformed size:{}".format(scaled))
            print("s:{}".format(self.model.scale))





if __name__ == '__main__':
    run()   # necessary for dataloader

    root_directory = './output'

    # Models to choose from [alexnet, vgg, resnet, efficientnet-bx]
    model_name = "alexnet"

    pretrained = False

    # Choose dataset
    dataset_name = 'isic'

    batch_size = 4

    learning_rate = 1e-5

    num_epochs = 1

    resolution = [224,224]

    scale = 1

    repeat = 0

    output_directory = root_directory + '/' + str(dataset_name)

    is_continue = False

    file_name = output_directory + '/' + 'stn_'+ dataset_name + '_'+\
                model_name + str(resolution)+'_'+str(scale)+'_'+str(repeat)

    if is_continue:
        prev_model = torch.load(file_name+'_last_weight.pth')
        scale = prev_model["transform_state_dict"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(model_name, dataset_name,batch_size, resolution,scale, file_name,device, learning_rate, pretrained)
    param = count_param(model.model)
    print('totoal parameters: %.2fM (%d)' % (param / 1e6, param))
    print("model name:{} \ndataset name:{} \ninitial resolution:{} "
          "\ninitial scale:{} \nrepeat:{} \npretrained:{} \nlearning rate:{}".format(
        model_name, dataset_name, resolution, scale, repeat, pretrained, learning_rate
    ))

    if is_continue:
        model.model.load_state_dict(prev_model["model_state_dict"])
        # model.model.scale = torch.tensor([prev_model["transform_state_dict"]],requires_grad=True, dtype=torch.float)
        model.optimizer.optimizers[0].load_state_dict(prev_model["optimizer_sgd_state_dict"])
        model.optimizer.optimizers[1].load_state_dict(prev_model["optimizer_adam_state_dict"])
        model.best_model = torch.load(file_name+'_best_weight.pth')

    duration = model.train(num_epochs)

    test_loss, test_accuracy,test_jaccard, test_f1, test_scale,test_s = model.test()

    paras = {'model_name': model_name,
             'optimizer': 'Adam',
             'dataset_name': dataset_name,
             'batch_size': batch_size,
             'num_epochs': num_epochs,
             'learning_rate': learning_rate,
             'duration': duration,
             'test_loss': test_loss,
             'test_acc': test_accuracy,
             'test_resolution': test_scale,
             'test_s':test_s,
             'test_jaccard':test_jaccard,
             'test_f1': test_f1,
             }

    pd.DataFrame.from_dict(paras,orient='index',columns=['parameters']).to_csv(
        file_name +'_paras.csv',index=True)

    torch.save(model.best_model, output_directory + '/' + 'stn_effi' + '_'
               + dataset_name + str(resolution) + '.pth')

    # Visualize the STN transformation on some input batch
    model.visualize_stn()
    #
    plt.savefig('./images/stn1.svg')
    plt.ioff()
    plt.show()

