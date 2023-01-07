import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.models as models
from pytorch_lightning.metrics import functional as FM
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score, confusion_matrix
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms.functional import hflip, vflip
from sklearn.metrics import confusion_matrix
import pickle
from mydataset import myDataSet

# PATH = './chest_xray_kaggle/'
BATCH_SIZE = 128


IF_TUNE_HYPER_PARAM = True

FLAG_TO_NUM_CLASSES = {
    "pathmnist": 9,
    "dermamnist": 7,
    "pneumoniamnist": 2,
    "chestmnist": 2,
    "chestkaggle": 2

}


FLAG_TO_MEAN = {
    "pathmnist": 0.5,
    "dermamnist": 0.5,
    "pneumoniamnist": 0.5,
    "chestmnist": (0.5, 0.5, 0.5)

}

FLAG_TO_STD = {
    "pathmnist": 0.5,
    "dermamnist": 0.5,
    "pneumoniamnist": 0.5,
    "chestmnist": 0.5,
    "chestkaggle": 0.5,
}

FLAG_TO_INPUT_CHANNELS = {
    "pathmnist": 3,
    "dermamnist": 3,
    "pneumoniamnist": 1,
    "chestmnist": 1,
    "chestkaggle":3
}

FLAG = "chestkaggle"
class MedClassifier(pl.LightningModule):
    def __init__(self, data_path, biases=None, num_annotators=3, dropout_rate=0.2, kldiv=True, alpha=1.0):
        super().__init__()
        self.num_classes = FLAG_TO_NUM_CLASSES[FLAG]
        self.model = models.resnet18(num_classes=self.num_classes)
        self.validation_losses = []
        self.epoch = -1
        self.training_losses = []
        self.kldiv = kldiv
        self.training_acc = []
        self.validation_acc = []
        self.alpha = alpha
        self.data_path = data_path
        if biases is None:
            self.biases = {}
        else:
            self.biases = biases
        self.num_annotators = num_annotators
        self.batch_size = 32
        self.dropout_rate = dropout_rate
        # self.model = models.resnet18(pretrained=True)
        # linear_size = list(self.model.children())[-1].in_features
        # replace final layer for fine tuning
        # self.model.fc = nn.Linear(linear_size, num_classes)
        # self.model.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=self.dropout_rate, training=m.training))

    def set_batch_size(self):
        if self.epoch >= 100:
            self.batch_size = 64
        if self.epoch >= 200:
            self.batch_size = 32

    def setup(self, stage):

        preprocess = transforms.Compose([

            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2365, 0.2365, 0.2365))
        ])

        self.load_and_combine_reader_data(preprocess)


    def load_and_combine_reader_data(self, transform):
        # self.mnist_test = datasets.ImageFolder(os.path.join(self.data_path, 'test'), transform=transform)
        self.mnist_test = myDataSet(os.path.join(self.data_path, 'test'),
                                   os.path.join('./all_images', 'test'), transform=transform)
        if self.num_annotators == 1:
            # self.mnist_train = datasets.ImageFolder(os.path.join(self.data_path, 'train'), transform=transform)
            # self.mnist_val = datasets.ImageFolder(os.path.join(self.data_path, 'val'), transform=transform)
            self.mnist_train = myDataSet(os.path.join(self.data_path, 'train'),
                                         os.path.join('./all_images', 'train'), transform=transform)
            self.mnist_val = myDataSet(os.path.join(self.data_path, 'val'),
                                         os.path.join('./all_images', 'val'), transform=transform)
            num_total = len(self.mnist_train.targets)
            num_pneu = sum(self.mnist_train.targets)
            num_norm = num_total - num_pneu
        else:
            train_dataset_lst = []
            val_dataset_lst = []
            num_total = 0
            num_pneu = 0
            for i in range(self.num_annotators):
                train_ds = datasets.ImageFolder(os.path.join(self.data_path, 'train', f'Reader{i}'),
                                                              transform=transform)
                num_total += len(train_ds.targets)
                num_pneu += sum(train_ds.targets)
                train_dataset_lst.append(train_ds)
                val_dataset_lst.append(datasets.ImageFolder(os.path.join(self.data_path, 'val', f'Reader{i}'),
                                                              transform=transform))
            self.mnist_train = torch.utils.data.ConcatDataset(train_dataset_lst)
            self.mnist_val = torch.utils.data.ConcatDataset(val_dataset_lst)

            num_norm = num_total - num_pneu

        self.pos_weight = torch.tensor([num_pneu / num_norm, num_pneu / num_pneu], dtype=torch.float, device='cuda:0')

    def train_dataloader(self):
        self.set_batch_size()
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        self.set_batch_size()
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        self.set_batch_size()
        self.batch_size = 700
        # change it back for testing!
        # only for hyper-parameter tuning purpose
        if IF_TUNE_HYPER_PARAM:
            return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=0)
        else:
            return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=0)

    def forward(self, X):
        X = self.model(X)
        # X = torch.log_softmax(X, dim=1)
        return X

    def cross_entropy_loss(self, logits, labels):
        # return F.nll_loss(logits, labels)
        if self.num_classes == 2:
            criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            # criterion = nn.BCEWithLogitsLoss(torch.tensor([1, 1214 / 3439], device='cuda'))
        else:
            criterion = nn.CrossEntropyLoss()
        return criterion(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        y = torch.reshape(y, (-1,))
        y = torch.as_tensor(data=y, dtype=torch.long)
        logits = self.forward(x)

        if FLAG_TO_NUM_CLASSES[FLAG] == 2:
            target = torch.zeros(y.shape[0], 2)

            target[range(target.shape[0]), y] = 1
            target = target.to(device='cuda')
        else:
            target = y

        ce_loss = self.cross_entropy_loss(logits, target)

        # geometric augmentation
        x_aug1 = hflip(x)
        x_aug2 = vflip(x)

        y_aug1 = self.forward(x_aug1)
        y_aug1 = torch.log_softmax(y_aug1, dim=1)
        y_aug2 = self.forward(x_aug2)
        y_aug2 = torch.log_softmax(y_aug2, dim=1)

        # 2 ways of getting kldiv loss
        kl_loss = F.kl_div(y_aug1, y_aug2)

        criterion = nn.KLDivLoss(reduction='sum', log_target=True)
        kl_loss2 = criterion(y_aug1, y_aug2)

        if self.kldiv:
            final_loss = ce_loss + self.alpha * kl_loss2
        else:
            final_loss = ce_loss
        # final_loss = ce_loss
        # print(final_loss)
        self.log('train_loss', final_loss)

        labels_hat = torch.argmax(logits, dim=1)
        # val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        train_acc = FM.accuracy(labels_hat, y)
        return {'loss': final_loss, 'acc': train_acc}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        y = torch.reshape(y, (-1,))
        y = torch.as_tensor(data=y, dtype=torch.long)

        if FLAG_TO_NUM_CLASSES[FLAG] == 2:
            target = torch.zeros(y.shape[0], 2)
            target[range(target.shape[0]), y] = 1
            target = target.to(device='cuda')
        else:
            target = y

        loss = self.cross_entropy_loss(logits, target)


        labels_hat = torch.argmax(logits, dim=1)
        # val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = FM.accuracy(labels_hat, y)
        # self.log('val_loss', loss, enable_graph=True)
        # self.log('val_acc', val_acc, enable_graph=True)
        self.log_dict({'val_loss': loss, 'val_acc': val_acc})
        # print({'val_loss': loss, 'val_acc': val_acc})
        return {'val_loss': loss, 'val_acc': val_acc}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_loss = avg_loss.item()
        self.training_losses.append((self.epoch, avg_loss))

        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        avg_acc = avg_acc.item()
        self.training_acc.append((self.epoch, avg_acc))

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_loss = avg_loss.item()
        self.validation_losses.append((self.epoch, avg_loss))

        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_acc = avg_acc.item()
        self.validation_acc.append((self.epoch, avg_acc))
        self.epoch += 1


    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y = torch.reshape(y, (-1,))
        y = torch.as_tensor(data=y, dtype=torch.long)
        logits = self.forward(x)

        if FLAG_TO_NUM_CLASSES[FLAG] == 2:
            target = torch.zeros(y.shape[0], 2)

            target[range(target.shape[0]), y] = 1
            target = target.to(device='cuda')
        else:
            target = y
        # logits = torch.log_softmax(logits, dim=1)
        loss = self.cross_entropy_loss(logits, target)
        logits = torch.log_softmax(logits, dim=1)
        labels_hat = torch.argmax(logits, dim=1)
        # labels_hat = torch.argmax(torch.exp(logits), dim=1)
        # test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        test_acc = FM.accuracy(labels_hat, y)
        test_recall = recall_score(y_true=y.to(device='cpu'), y_pred=labels_hat.to(device='cpu'), pos_label=1)
        # test_precision = 1
        test_precision = precision_score(y_true=y.to(device='cpu'), y_pred=labels_hat.to(device='cpu'), pos_label=1)
        y_true = y.to(device='cpu')
        y_pred = labels_hat.to(device='cpu')
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        try:
            test_auc = roc_auc_score(y_true=y.to(device='cpu'), y_score=labels_hat.to(device='cpu'))

        except ValueError as e:
            print(e)
            test_auc = torch.tensor(0)
        # log the outputs!
        self.log_dict({'test_loss': loss, 'test_acc': test_acc, 'test_auc_roc': test_auc, "test_recall": test_recall,
                       "test_precision": test_precision, 'test_specificity': specificity, 'test_sensitivity': sensitivity,
                       'tn': tn, 'fp': fp, 'tp': tp, 'fn': fn})
        return {'test_loss': loss, 'test_acc': test_acc, 'test_auc_roc': test_auc, "test_recall": test_recall,
                       "test_precision": test_precision, 'test_specificity': specificity, 'test_sensitivity': sensitivity,
                       'tn': tn, 'fp': fp, 'tp': tp, 'fn': fn}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def save_plot(train, valid, name, number=0):
    """ Displays training curve.
    :param train: Training statistics
    :param valid: Validation statistics
    :param y_label: Y-axis label of the plot
    :param number: The number of the plot
    :return: None
    """
    plt.figure(number)
    plt.clf()
    train = np.array(train)
    valid = np.array(valid)
    plt.plot(train[:, 0], train[:, 1], "b", label="Train")
    plt.plot(valid[:, 0], valid[:, 1], "g", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(name)
    plt.legend()
    plt.draw()
    # plt.pause(1)
    # plt.show()
    # plt.pause(0.01)
    # modify to save figure
    plt.savefig(f'{name}.png')


def train_model(name, bias, num_annotators, epochs, dropout_rate, path, alpha=1.0, kldiv=True):
    print('='*20)
    # data_module = MultiAnnoMedMNISTDataModule(bias, num_annotators)
    # print('bias:', bias)
    print('epochs:', epochs)
    if kldiv:
        print('alpha: ', alpha)
    print('=' * 20)
    # train
    # model = MedClassifier()
    # pl.seed_everything(0, workers=True)
    model = MedClassifier(data_path=path, biases=bias, num_annotators=num_annotators, dropout_rate=dropout_rate, kldiv=kldiv, alpha=alpha)
    # trainer = pl.Trainer(gpus=1, max_epochs=epochs, progress_bar_refresh_rate=0)
    trainer = pl.Trainer(gpus=1, max_epochs=epochs, checkpoint_callback=False, logger=False, deterministic=True)
    # trainer.fit(model, data_module)
    trainer.fit(model)
    save_plot(model.training_losses, model.validation_losses[1:], f'./output/{name}_loss')
    save_plot(model.training_acc, model.validation_acc[1:], f'./output/{name}_acc')
    result = trainer.test(model)
    result = result[0]
    result['train_acc_lst'] = model.training_acc
    result['train_loss_lst'] = model.training_losses
    result['val_acc_lst'] = model.validation_acc[1:]
    result['val_loss_lst'] = model.validation_losses[1:]
    trainer.test(model)
    with open(f'./output/{name}.pickle', 'wb') as data:
        pickle.dump(result, data)
    # trainer.save_checkpoint(name)
    torch.save(model.model.state_dict(), os.path.join('output', name))
    print(f'save as {name}.pickle')

    # data_module = MultiAnnoMedMNISTDataModule(bias, num_annotators)
    # loaded weight
    loaded_model = MedClassifier(data_path=path, biases=bias, num_annotators=num_annotators, dropout_rate=dropout_rate,
                          kldiv=kldiv, alpha=alpha)
    loaded_model.model.load_state_dict(torch.load(os.path.join('output', name)), strict=True)
    # loaded_model = LightningMNISTClassifier.load_from_checkpoint(name)
    # loaded_model = MedClassifier.load_from_checkpoint(name)
    trainer_loaded = pl.Trainer(gpus=1)
    trainer_loaded.test(loaded_model)
    # trainer_loaded.test(loaded_model, datamodule=data_module)

'''
Load model checkpoint and test using **CURRENT** test data
'''
def get_acc(model_path, data_path):
    print(model_path)
    #loaded_model = MedClassifier.load_from_checkpoint(model_path, data_path=data_path)
    loaded_model = MedClassifier(data_path=data_path, num_annotators=1)
    loaded_model.model.load_state_dict(torch.load(model_path))
    loaded_model.model.eval()
    trainer_loaded = pl.Trainer(gpus=1)
    d = trainer_loaded.test(loaded_model)[0]
    result = {'acc': d['test_acc'], 'auc_roc': d['test_auc_roc'],
              'tp': d['tp'], 'tn': d['tn'], 'fp': d['fp'], 'fn': d['fn']}
    return result
    # trainer_loaded.test(loaded_model, datamodule=data_module)
'''
Load model checkpoint in the pytorch lightning way and test using **CURRENT** test data
Not used anymore since save in lightning format have a higher storage cost
'''
def get_acc_lightning(model_path, data_path):
    print(model_path)
    loaded_model = MedClassifier.load_from_checkpoint(model_path, data_path=data_path, num_annotators=1)
    #loaded_model = MedClassifier(data_path=data_path, num_annotators=1)
    #loaded_model.model.load_state_dict(torch.load(model_path))
    #loaded_model.model.eval()
    trainer_loaded = pl.Trainer(gpus=1)
    d = trainer_loaded.test(loaded_model)[0]
    result = {'acc': d['test_acc'], 'auc_roc': d['test_auc_roc'],
              'tp': d['tp'], 'tn': d['tn'], 'fp': d['fp'], 'fn': d['fn']}
    return result

'''
Directly load test result from pickle
'''
def get_acc_from_pickle(pickle_path):
    print('-' * 20)
    print(pickle_path)
    result = {}
    with open(pickle_path, 'rb') as f:
        d = pickle.load(f)
        print('acc', d['test_acc'])
        print('auc_roc', d['test_auc_roc'])
        print('tp', d['tp'])
        print('tn', d['tn'])
        print('fp', d['fp'])
        print('fn', d['fn'])
        result = {'acc': d['test_acc'], 'auc_roc': d['test_auc_roc'],
                  'tp': d['tp'], 'tn': d['tn'], 'fp': d['fp'], 'fn':d['fn']}
    return result

if __name__ == '__main__':
    # preprocess = transforms.Compose([
    #
    #     transforms.Resize([224, 224]),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])
    # folder_name = 'chest_xray_kaggle-bias1'
    # trial_data1 = datasets.ImageFolder('./CXR_example/train/Reader1', transform=preprocess)
    # trial_data2 = datasets.ImageFolder('./CXR_example/train/Reader2', transform=preprocess)
    # l = [trial_data1, trial_data2]
    # image_datasets = torch.utils.data.ConcatDataset(l)
    # image_datasets
    # trial_data_loader = torch.utils.data.DataLoader(dataset=image_datasets, batch_size=624, shuffle=True, num_workers=4)
    # # print(test_data)
    # images = trial_data_loader.dataset.imgs
    # for i in range(len(trial_data_loader.dataset.imgs)):
    #     images[i] = images[i][0]
    # bias_name = '5_to_each'
    # file_path = f'./chest_xray_kaggle-5_to_each/'
    if not os.path.exists('output'): os.mkdir('output')

    bias0 = {'bias_name': '5_5', 'bias_type': {'0to1': 0.05, '1to0': 0.05}}
    bias1 = {'bias_name': '10_10', 'bias_type': {'0to1': 0.1, '1to0': 0.1}}
    bias2 = {'bias_name': '15_15', 'bias_type': {'0to1': 0.15, '1to0': 0.15}}
    bias3 = {'bias_name': '20_20', 'bias_type': {'0to1': 0.2, '1to0': 0.2}}
    bias4 = {'bias_name': '25_25', 'bias_type': {'0to1': 0.25, '1to0': 0.25}}

    bias0_0 = {'bias_name': '10_0', 'bias_type': {'0to1': 0.1, '1to0': 0}}
    bias1_0 = {'bias_name': '20_0', 'bias_type': {'0to1': 0.2, '1to0': 0}}
    bias2_0 = {'bias_name': '30_0', 'bias_type': {'0to1': 0.3, '1to0': 0}}
    bias3_0 = {'bias_name': '40_0', 'bias_type': {'0to1': 0.4, '1to0': 0}}
    bias4_0 = {'bias_name': '50_0', 'bias_type': {'0to1': 0.5, '1to0': 0}}

    bias0_1 = {'bias_name': '0_10', 'bias_type': {'0to1': 0, '1to0': 0.1}}
    bias1_1 = {'bias_name': '0_20', 'bias_type': {'0to1': 0, '1to0': 0.2}}
    bias2_1 = {'bias_name': '0_30', 'bias_type': {'0to1': 0, '1to0': 0.3}}
    bias3_1 = {'bias_name': '0_40', 'bias_type': {'0to1': 0, '1to0': 0.4}}
    bias4_1 = {'bias_name': '0_50', 'bias_type': {'0to1': 0, '1to0': 0.5}}

    # biases_lst = [bias0, bias1, bias2, bias3, bias4, bias0_0, bias1_0,
    #           bias2_0,
    #           bias3_0,
    #           bias4_0,
    #           bias0_1,
    #           bias1_1,
    #           bias2_1,
    #           bias3_1,
    #           bias4_1]
    biases_lst = [bias0_0, bias1_0,
                  bias2_0,
                  bias3_0,
                  bias4_0,
                  bias0_1,
                  bias1_1,
                  bias2_1,
                  bias3_1,
                  bias4_1]

    trial_date = 'Jan6'
    # bias_name_lst = ['bias{x}' for x in n]
    # file_name_lst = [f'./chest_xray_kaggle-bias{x}/' for x in n]
    bias_name_lst_only = [b['bias_name'] for b in biases_lst]
    bias_name_lst = []
    file_name_lst = []

    trial_num_lst = [1,2] # the same one as in introduce_bias.py
    base_folder = 'bias_csv' # the same one as in introduce_bias.py
    for name in bias_name_lst_only:
        for t in trial_num_lst:
            file_name_lst.append(f"./{base_folder}/{name}_trial{t}_csv")
            bias_name_lst.append(f"{name}_trial{t}")

    alpha_list = [1/4, 1/2, 1, 2, 4]

    seed = 2139
    pl.seed_everything(seed, workers=True)

    no_bias = {} # deprecated param, no actual use, please ignore
    epoch_num = 5
    for bias_name, file_name in zip(bias_name_lst, file_name_lst):
        file_path = os.path.join(os.getcwd(), file_name)
        for alpha in alpha_list:
            train_model(f'epoch{epoch_num}-kaggle-{bias_name}-KLDIV-alpha{alpha}-trial{trial_date}', no_bias,
                        1, epoch_num, 0, path=file_path, kldiv=True, alpha=alpha)

            # result = get_acc_from_pickle(f'epoch{epoch_num}-kaggle-{bias_name}-KLDIV-alpha{alpha}-trial{trial_date}.pickle')
            result = get_acc(f'./output/epoch{epoch_num}-kaggle-{bias_name}-KLDIV-alpha{alpha}-trial{trial_date}', file_path)

    WRITE_TEST_RESULT_TO_CSV = False
    if WRITE_TEST_RESULT_TO_CSV:
        output_csv_file_name = 'portion_test_result_more_seed-10-1.csv'
        with open(output_csv_file_name, 'w', newline='') as csvfile:
            fieldname = ['name','trial','model', 'acc', 'auc_roc', 'tp', 'tn', 'fp', 'fn']
            writer = csv.DictWriter(csvfile, fieldnames=fieldname)
            writer.writeheader()

            for bias_name, file_name in zip(bias_name_lst, file_name_lst):
                file_path = os.path.join(os.getcwd(), file_name)

                # result = get_acc_from_pickle(f'epoch50-kaggle-{bias_name}-ce-trial{trial_num}.pickle')
                # # train_model(f'epoch250-kaggle-{bias_name}-DROPOUT-trial{trial_num}', no_bias,
                # #             3, 250, 0.2, path=file_path, kldiv=False)
                # result['name'] = bias_name[:bias_name.rindex('_')]
                # result['model'] = 'ce'
                # result['trial'] = bias_name[-1]
                # writer.writerow(result)

                for alpha in alpha_list:
                    result = get_acc(f'./output/epoch{epoch_num}-kaggle-{bias_name}-KLDIV-alpha{alpha}-trial{trial_date}', file_path)
                    result['name'] = bias_name[:bias_name.rindex('_')]
                    result['model'] = 'kldiv'
                    result['trial'] = bias_name[-1]
                    writer.writerow(result)

    # bias_name_lst = ['bias0']
    # file_name_lst = ['./chest_xray_kaggle-bias0/']
    # with open('perfect.csv', 'w', newline='') as csvfile:
    #     fieldname = ['name', 'trial', 'model', 'acc', 'auc_roc', 'tp', 'tn', 'fp', 'fn']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldname)
    #     writer.writeheader()
    #     for trial_num in [5,6,7]:
    #         for bias_name, file_name in zip(bias_name_lst, file_name_lst):
    #             file_path = os.path.join(os.getcwd(), file_name)
    #
    #             result = get_acc_lightning(f'epoch80-kaggle-{bias_name}-ce-trial{trial_num}', file_name)
    #             result['name'] = 'nobias'
    #             result['model'] = 'ce'
    #             result['trial'] = trial_num
    #             # train_model(f'epoch250-kaggle-{bias_name}-DROPOUT-trial{trial_num}', no_bias,
    #             #             3, 250, 0.2, path=file_path, kldiv=False)
    #             writer.writerow(result)
    #
    #             for alpha in [1]:
    #                 result = get_acc_lightning(f'epoch80-kaggle-{bias_name}-KLDIV-alpha{alpha}-trial{trial_num}', file_name)
    #                 result['name'] = 'nobias'
    #                 result['model'] = 'kldiv'
    #                 result['trial'] = trial_num
    #                 writer.writerow(result)

    # recall_lst = []
    # acc_dict = {}
    # # 'test_acc': test_acc, 'test_auc_roc': test_auc, "test_recall": test_recall,
    # # 'test_specificity': specificity

    # for bias_name, file_name in zip(bias_name_lst, file_name_lst):
    #     file_path = os.path.join(os.getcwd(), file_name)
    #     # acc_dict[bias_name] = np.zeros(8)
    #     # if bias_name == "bias6": continue
    #     for trial_num in [6, 5, 8]:
    #         baseline_result = get_acc(f'epoch150-kaggle-{bias_name}-ce-trial{trial_num}', file_path)
    #         # result = get_acc(f'epoch250-kaggle-{bias_name}-DROPOUT', file_path)[0]
    #         # ce_recall = result['test_recall']
    #         for alpha in [1]:
    #             kldiv_result = get_acc(f'epoch150-kaggle-{bias_name}-KLDIV-alpha{alpha}-trial{trial_num}', file_path)
    #             # print(result)



    #         acc_dict[bias_name][0] += baseline_result[0]['test_acc']
    #         acc_dict[bias_name][1] += baseline_result[0]['test_recall']
    #         acc_dict[bias_name][2] += baseline_result[0]['test_auc_roc']
    #         acc_dict[bias_name][3] += baseline_result[0]['test_specificity']
    #
    #         acc_dict[bias_name][4] += kldiv_result[0]['test_acc']
    #         acc_dict[bias_name][5] += kldiv_result[0]['test_recall']
    #         acc_dict[bias_name][6] += kldiv_result[0]['test_auc_roc']
    #         acc_dict[bias_name][7] += kldiv_result[0]['test_specificity']
    #     acc_dict[bias_name] /= 3
    # print(acc_dict)

