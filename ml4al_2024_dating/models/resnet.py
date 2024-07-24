import json
import glob
import torch.nn.functional as F
from sklearn.metrics import f1_score
import wandb
from torchvision.datasets import ImageFolder
import argparse
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import time
import os
import copy
from PIL import Image, ImageFile
import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from collections import defaultdict
import random
import csv
import pickle

ImageFile.LOAD_TRUNCATED_IMAGES = True

num_class = 14


def get_args():
    parser = argparse.ArgumentParser(
        description="Train a model on cuneiform data")
    parser.add_argument('--data_path', type=str,
                        default="/graft3/code/tracy/data/collection/segmented/", help='Root path to the dataset')
    parser.add_argument('--cid_file', type=str,
                        default="/graft3/code/tracy/data/idx2collection.json")
    parser.add_argument('--reg', type=int, default=0,
                        help='Regularization flag (0 or 1)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float,
                        default=0.0001, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-1, help='Weight decay for regularization')
    parser.add_argument('--num_adv', type=int, default=1,
                        help='Number of adversarial examples, default is 1')
    return parser.parse_args()


class CuneiformDataset(Dataset):
    def __init__(self, root, cid_file, transform=None, most=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform
        self.cidfile = json.load(open(cid_file, 'r'))
        self.images = glob.glob(os.path.join(root, "*/*.jpg"))
        print("finish loading images")
        self.pid2cid = {}
        self.name2cid = {"unknown": 0}
        freq_cid_cnt = Counter()
        for x in self.images:
            # {pid: cid, ...}
            # {"1": "Vorderasiatisches Museum, Berlin, Germany", ...}
            pid = x.split("/")[-1].strip(".jpg").strip("P")
            pid = str(int(pid))
            if pid not in self.cidfile:
                self.pid2cid[pid] = 0
            else:
                cid = self.cidfile[pid]
                if cid not in self.name2cid:
                    self.name2cid[cid] = len(self.name2cid)
                freq_cid_cnt[cid] += 1
                self.pid2cid[pid] = self.name2cid[cid]

        # if set, the dataset initialization will only consider the top N most common categories as specified by self.most.
        self.most = most
        self.most_common_cid = {
            x: idx + 1 for idx, (x, _) in enumerate(freq_cid_cnt.most_common(most))}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.images[idx]
        pid = image_path.split("/")[-1].strip(".jpg").strip("P")
        image = Image.open(image_path)
        label = int(image_path.split("/")[-2])
        pid = str(int(pid))
        if self.transform:
            image = self.transform(image)

        if self.most:
            if self.pid2cid[pid] in self.most_common_cid:
                cid = self.most_common_cid[self.pid2cid[pid]]
            else:
                cid = 0
        else:
            cid = self.pid2cid[pid]

        sample = {'image': image,
                  'label': label,
                  'cid': cid,
                  'pid': pid
                  }
        return sample


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TwoBranchPred(nn.Module):
    def __init__(self, output_size=17, n2=212, n_in=2048):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=output_size),
        )

    def forward(self, x):
        return self.branch1(x), x


args = get_args()
base_path = args.data_path  # set default base_path =
num_adv = args.num_adv  # set default = 1
batch_size = args.batch_size
epochs = args.epochs
lr = args.learning_rate  # set default = 0.0001
weight_decay = args.weight_decay  # set default = 1e-1
reg_on = bool(args.reg)  # set default = false
cid_file = args.cid_file  # "/graft3/code/tracy/data/idx2collection.json"

disc_freq = 10


resnet50_weights = torchvision.models.ResNet50_Weights.DEFAULT
resnet50_model = torchvision.models.resnet50(
    weights=resnet50_weights).to(device)
resnet50_model.fc = TwoBranchPred(output_size=num_class)
# printModelSummary(resnet50_model)


train_dataset = CuneiformDataset(
    root=base_path + "/train/", cid_file=cid_file, transform=data_transforms['train'], most=num_adv-1)
val_dataset = CuneiformDataset(
    root=base_path + "/valid/", cid_file=cid_file, transform=data_transforms['val'], most=num_adv-1)
test_dataset = CuneiformDataset(
    root=base_path + "/test/", cid_file=cid_file, transform=data_transforms['val'], most=num_adv-1)

train_dataset_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
val_dataset_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, num_workers=12)
test_dataset_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, num_workers=12)


dnet = nn.Sequential(
    nn.BatchNorm1d(2048),
    nn.Dropout(p=0.25),
    nn.Linear(in_features=2048, out_features=1024),
    nn.ReLU(),
    nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1),
    nn.Dropout(p=0.5),
    nn.Linear(in_features=1024, out_features=num_adv),
).to(device)

# train a disc
ce = nn.CrossEntropyLoss()
maxe = nn.KLDivLoss()
optimizer = torch.optim.AdamW(
    resnet50_model.parameters(), lr=lr, weight_decay=weight_decay)
opt2 = torch.optim.AdamW(dnet.parameters(), lr=lr, weight_decay=weight_decay)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-4)

wandb.login()

run = wandb.init(
    # Set the project where this run will be logged
    project="cuneiform_may15",
    # summary="resnet50-reg",
    settings=wandb.Settings(start_method="fork"),
    name='sample_split',
    # Track hyperparameters and run metadata
    # tags = [base_path.split("/")[-1], 'resnet50-reg'],
    # tags = [],
    # config={
    #     "learning_rate": 0.0001,
    #     "epochs": 7,
    #     "batch_size": batch_size,
    #     "num_adv": num_adv,
    # }
)


def train_model(model, train_data_loader, val_data_loader, test_data_loader, optimizer, optimizer2, num_epochs, ce, maxe):
    model.to(device)
    pseudo_label = torch.zeros(
        (batch_size, num_adv)).fill_(1/num_adv).to(device)
    loss_d, loss_maxe = 0, 0
    max_val_f1, max_test_f1, max_test_f1_micro = 0, 0, 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 100)

        model.train()

        running_size, running_loss_ce, running_loss_dis, running_loss_maxe = 0, 0.0, 0.0, 0.0
        train_true_labels = []
        train_pred_labels = []

        # batch processing
        for idx, sample in tqdm.tqdm(enumerate(train_data_loader)):
            inputs, labels = sample['image'], sample['label']
            dlabels = sample['cid']
            train_true_labels = train_true_labels + labels.tolist()
            inputs = inputs.to(device)
            labels = labels.to(device)
            dlabels = dlabels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs, output_beforefc = model(inputs)
                doutputs = dnet(output_beforefc)
                logits = outputs.detach().cpu().numpy()

                train_pred_labels = train_pred_labels + \
                    np.argmax(logits, axis=1).flatten().tolist()
                loss_ce = ce(outputs, labels)
                doutputs = F.log_softmax(doutputs, dim=1)

                if reg_on:
                    loss_maxe = maxe(
                        doutputs, pseudo_label[:doutputs.size(0), :])
                    loss = loss_ce + loss_maxe
                else:
                    loss = loss_ce

                loss.backward()
                optimizer.step()
                # set it to 1, 5, 10, etc..
                if reg_on and epoch > 0 and epoch < 8 and idx % disc_freq == 0 and loss_ce.item() > loss_maxe.item() * 0.1:
                    # train descriminator
                    _, output_beforefc = model(inputs)
                    doutputs = dnet(output_beforefc)
                    output_beforefc.detach()
                    loss_d = ce(doutputs, dlabels)
                    optimizer2.zero_grad()
                    loss_d.backward()
                    optimizer2.step()

            running_loss_ce += loss_ce.item() * inputs.size(0)
            if loss_d:
                running_loss_dis += loss_d.item() * inputs.size(0)
            if loss_maxe:
                running_loss_maxe += loss_maxe.item() * inputs.size(0)
            running_size += inputs.size(0)

        epoch_train_loss = running_loss_ce / running_size
        epoch_train_accuracy = accuracy_score(
            train_true_labels, train_pred_labels)
        epoch_train_f1 = f1_score(
            train_true_labels, train_pred_labels, average='macro')
        epoch_train_f1_micro = f1_score(
            train_true_labels, train_pred_labels, average='micro')

        print('Train Loss: {:.2f}'.format(epoch_train_loss * 2), 'Train Accuracy: ',
              epoch_train_accuracy, f"F1 {epoch_train_f1*100:.2f} {epoch_train_f1_micro*100:2f}")
        wandb.log({"train_accuracy": epoch_train_accuracy, "train_loss_ce": epoch_train_loss,
                  "train_avg_f1": epoch_train_f1, "train_micro_f1": epoch_train_f1_micro})

        running_size, running_loss = 0, 0.0
        val_true_labels = []
        val_pred_labels = []
        for idx, sample in tqdm.tqdm(enumerate(val_data_loader)):
            inputs, labels = sample['image'], sample['label']
            val_true_labels = val_true_labels + labels.tolist()
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(True):
                outputs, _ = model(inputs)
                logits = outputs.detach().cpu().numpy()
                val_pred_labels = val_pred_labels + \
                    np.argmax(logits, axis=1).flatten().tolist()
                loss = ce(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_size += inputs.size(0)

        epoch_val_loss = running_loss / running_size
        epoch_val_accuracy = accuracy_score(val_true_labels, val_pred_labels)
        epoch_val_f1 = f1_score(
            val_true_labels, val_pred_labels, average='macro')
        epoch_val_f1_micro = f1_score(
            val_true_labels, val_pred_labels, average='micro')

        print('Val Loss: {:.4f}'.format(epoch_val_loss), 'Val Accuracy: ',
              epoch_val_accuracy,  'F1', epoch_val_f1, epoch_val_f1_micro)
        wandb.log({"val_accuracy": epoch_val_accuracy, "val_loss": epoch_val_loss,
                  "val_avg_f1": epoch_val_f1, "val_micro_f1": epoch_val_f1_micro})

        model.eval()

        running_size, running_loss = 0, 0.0
        test_true_labels = []
        test_pred_labels = []
        test_logits = []
        for idx, sample in tqdm.tqdm(enumerate(test_data_loader)):
            inputs, labels = sample['image'], sample['label']
            test_true_labels = test_true_labels + labels.tolist()
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(True):
                outputs, _ = model(inputs)
                logits = outputs.detach().cpu().numpy()
                test_pred_labels = test_pred_labels + \
                    np.argmax(logits, axis=1).flatten().tolist()
                test_logits = test_logits + logits.tolist()
                loss = ce(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_size += inputs.size(0)

        epoch_test_loss = running_loss / running_size
        epoch_test_accuracy = accuracy_score(
            test_true_labels, test_pred_labels)
        epoch_test_f1 = f1_score(
            test_true_labels, test_pred_labels, average='macro')
        epoch_test_f1_micro = f1_score(
            test_true_labels, test_pred_labels, average='micro')

        print('Test Loss: {:.4f}'.format(epoch_test_loss), 'Test Accuracy: ',
              epoch_test_accuracy,  'F1', epoch_test_f1, epoch_test_f1_micro)
        wandb.log({"test_accuracy": epoch_test_accuracy, "test_loss": epoch_test_loss,
                  "test_avg_f1": epoch_test_f1, "test_micro_f1": epoch_test_f1_micro})

        # best epoch which have the largest f1 for validating
        if epoch_val_f1 > max_val_f1:
            max_val_f1 = epoch_val_f1
            max_test_f1, max_test_f1_micro = epoch_test_f1, epoch_test_f1_micro

        # from sklearn.metrics import confusion_matrix
        # import seaborn as sns
        # cf_matrix = confusion_matrix(test_true_labels, test_pred_labels)
        # sns.heatmap(cf_matrix, annot=True)

        # df = pd.DataFrame({
        #     'True Labels': test_true_labels,
        #     'Predictions': test_pred_labels,
        # })
        # logits_columns = {f'Logit_{i}': [logits[i] for logits in test_logits] for i in range(len(test_logits[0]))}
        # df = pd.concat([df, pd.DataFrame(logits_columns)], axis = 1)
        # csv_file = "resnet50_predictions.csv"
        # df.to_csv(csv_file, index=False)
        # print(f"Saved model predictions to {csv_file}")

    print('max_f1: {:.4f}'.format(max_test_f1),
          'max_f1_micro: {:.4f}'.format(max_test_f1_micro))
    wandb.log({"max_f1": max_test_f1, "max_f1_micro": max_test_f1_micro})

    return model


model = train_model(resnet50_model, train_dataset_loader, val_dataset_loader,
                    test_dataset_loader, optimizer, opt2, epochs, ce, maxe)

torch.cuda.empty_cache()
