from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import random
import argparse
import re
import numpy as np
import tqdm
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
import wandb
import json
import torch
from torch.utils.data import Dataset
import torchtext
torchtext.disable_torchtext_deprecation_warning()

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

best_test_details = {
    'test1': {'true_labels': [], 'pred_labels': [], 'max_f1': 0, 'pred_labels_vote': [], 'max_f1_vote': 0},
    'test2': {'true_labels': [], 'pred_labels': [], 'max_f1': 0, 'pred_labels_vote': [], 'max_f1_vote': 0},
    'test3': {'true_labels': [], 'pred_labels': [], 'max_f1': 0, 'pred_labels_vote': [], 'max_f1_vote': 0}
}


def get_args():
    parser = argparse.ArgumentParser(description="Train an LSTM on text data")
    parser.add_argument('--data_path', type=str,
                        default='/graft3/code/tracy/data/final_may24/')
    parser.add_argument('--text_field', type=str, default='raw',
                        choices=['sign', 'raw'], help='Text field to use for training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--embedding_dim', type=int,
                        default=256, help='Dimension of the embedding layer')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Dimension of the LSTM hidden layer')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--maxl', type=int, default=1024,
                        help='Max length for training')
    parser.add_argument('--dropout', type=float,
                        default=0.5, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument("--line", type=int, default=1)
    parser.add_argument("--wd", type=float, default=1e-3)
    # parser.add_argument("--metrics", type=str, default='default')

    return parser.parse_args()


args = get_args()


class CustomTextDataset(Dataset):
    def __init__(self, json_file, split, vocab=None, max_length=1024):
        self.split = split
        self.label_encoder = LabelEncoder()
        self.data = self.load_data(json_file)
        self.texts, self.labels, self.seg_ids = self.extract_text_and_labels(
            self.data)
        self.max_length = max_length
        if vocab is None:
            self.vocab = self.create_vocab(self.texts)
        else:
            self.vocab = vocab
        self.text_tensors = self.text_to_tensor(self.texts, self.vocab)
        self.labels = torch.tensor(
            self.label_encoder.fit_transform(self.labels), dtype=torch.long)

    @staticmethod
    def text_to_tensor(texts, vocab):
        # import pdb; pdb.set_trace()
        text_tensor = [torch.tensor(
            vocab(list(text)), dtype=torch.long) for text in texts]
        # text_tensor = pad_sequence(text_tensor, batch_first=True, padding_value=vocab['<pad>'])
        return text_tensor

    def print_label_mapping(self):
        for index, label in enumerate(self.label_encoder.classes_):
            print(f"{label} -> {index}")

    @staticmethod
    def load_data(json_file):
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    @staticmethod
    def extract_text_and_labels(data):
        texts = []
        labels = []
        seg_ids = []
        # import pdb; pdb.set_trace()
        for idx, (key, entry) in enumerate(data.items()):
            entry_time = entry['time']
            tmp_text = []
            acc_line = 0
            for key in entry['text']:
                for text_item in entry['text'][key]:
                    # transliteration - 'raw'
                    # sign token - 'sign'
                    if args.text_field in text_item:
                        tmp_text += text_item[args.text_field]
                        acc_line += 1

                        if args.line > 0:
                            if acc_line == args.line:
                                texts.append(tmp_text)
                                labels.append(entry_time)
                                seg_ids.append(idx)
                                tmp_text = []
                                acc_line = 0
                            else:
                                tmp_text.append("<B>")
                        else:
                            tmp_text.append("<B>")

            if acc_line > 0:
                texts.append(tmp_text[:-1])  # not appending the last <B>
                labels.append(entry_time)
                seg_ids.append(idx)

        return texts, labels, seg_ids

    # s_tokens = ['<B>', # broken
    #         '<M>', # missing one or more token?
    #         "<S>", # blank space
    #         "<D>", # divine
    #         "<munus>", # young woman, or woman
    #         "<ansze>",
    #         "<ki>",
    #         "<disz>",
    #         "x", # uknown signs
    #         ]

    @staticmethod
    def yield_tokens(texts):
        for text in texts:
            # print(text)
            # text = text.strip()
            # text = re.sub(r"<S>", "", text)
            # text = re.sub(r"<B>|<M>|<S>", "", text)
            # if False:args.text_field == 'raw':
            #     text = re.sub(r"<D>|<munus>|<ansze>|<ki>|<disz>|x", "", text)
            yield list(text)

    def create_vocab(self, texts):
        vocab = build_vocab_from_iterator(self.yield_tokens(
            texts), specials=['<unk>', '<pad>', "<B>"])
        vocab.set_default_index(vocab['<unk>'])
        return vocab

    def __getitem__(self, idx):
        text_tensor = self.text_tensors[idx]
        # import pdb; pdb.set_trace()
        import copy
        text = copy.copy(text_tensor)
        N = len(text)
        # if self.split == 'train' :
        if args.line < 0:
            # sample a segment from the full docs
            if N > self.max_length:
                start_idx = random.randint(0, N - self.max_length)
                # print("rand", start_idx)
                text = text[start_idx:start_idx + self.max_length]

        label = self.labels[idx]
        seg_id = self.seg_ids[idx]
        return text, label, seg_id

    def __len__(self):
        return len(self.texts)


def collate(batch):
    # [(t1, l1, s1), (t2, l2, s2), ...]
    texts = [x for x, y, z in batch]
    labels = [y for x, y, z in batch]
    seg_ids = [z for x, y, z in batch]
    texts = pad_sequence(texts, batch_first=True,
                         padding_value=train_dataset.vocab['<pad>'])
    labels = torch.LongTensor(labels)
    return texts, labels, seg_ids


train_json = args.data_path + '/train_data.json'
valid_json = args.data_path + '/valid_data.json'

test_json = args.data_path + '/test_data.json'
test2_json = args.data_path + '/test_data_2.json'
test3_json = args.data_path + '/test_data_3.json'

train_dataset = CustomTextDataset(
    train_json, split='train', max_length=args.maxl)  # It will create its vocab
valid_dataset = CustomTextDataset(
    valid_json, vocab=train_dataset.vocab, split='valid', max_length=args.maxl)  # Reuse vocab

test_dataset = CustomTextDataset(
    test_json, vocab=train_dataset.vocab, split='test', max_length=args.maxl)  # Reuse vocab
test2_dataset = CustomTextDataset(
    test2_json, vocab=train_dataset.vocab, split='test', max_length=args.maxl)  # Reuse vocab
test3_dataset = CustomTextDataset(
    test3_json, vocab=train_dataset.vocab, split='test', max_length=args.maxl)  # Reuse vocab

train_dataset.print_label_mapping()

train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                          shuffle=True, num_workers=4, collate_fn=collate)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=4, collate_fn=collate)

test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=4, collate_fn=collate)
test2_loader = DataLoader(test2_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=4, collate_fn=collate)
test3_loader = DataLoader(test3_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=4, collate_fn=collate)

print("vocab size:", len(train_dataset.vocab))
print("trainset size:", len(train_dataset))
print("testset size:", len(test_dataset))
print("valid size:", len(valid_dataset))


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, dropout=0.5):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        out = self.fc(hidden)
        return out


model = CharLSTM(vocab_size=len(train_dataset.vocab), embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
                 output_dim=len(train_dataset.label_encoder.classes_), num_layers=args.num_layers, dropout=args.dropout)

wandb.login()

suffix = args.data_path.split("/")[-1][:4]
run = wandb.init(
    project="cuneiform_may23",
    # summary="resnet101-reg",
    settings=wandb.Settings(start_method="fork"),
    name=f'P_{suffix}_{args.text_field}_bsz{args.batch_size}_lr{args.lr}_drop{args.dropout}_max{args.maxl}_1e-3wd_line{args.line}_wd{args.wd}',
    # Track hyperparameters and run metadata
    # tags = [base_path.split("/")[-1], 'resnet101-reg'],
    # tags = [],
    config={
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "weight_decay": args.wd,
    }
)


def calc_scores(labels, preds, average, seg_ids):
    # if args.metrics == 'default':
    #     return f1_score(labels, preds, average=average)
    # elif args.metrics == 'vote': ### majority vote
    # import pdb; pdb.set_trace()
    cur_seg_id = None
    final_labels = []
    final_preds = []
    tmp_preds = []
    zip_data = list(zip(labels, preds, seg_ids))
    zip_data = sorted(zip_data, key=lambda x: x[-1])
    for l, p, s in zip_data:
        if cur_seg_id is None:
            cur_seg_id = s
            cur_label = l
            tmp_preds = [p]

        elif cur_seg_id != s:
            c = Counter(tmp_preds)
            # majority vote
            final_preds.append(c.most_common(1)[0][0])
            final_labels.append(cur_label)

            tmp_preds = [p]
            cur_label = l
            cur_seg_id = s
        else:
            tmp_preds.append(p)

    # handle last one
    if len(tmp_preds) > 0:
        c = Counter(tmp_preds)
        final_preds.append(c.most_common(1)[0][0])
        final_labels.append(cur_label)

    return f1_score(labels, preds, average=average), f1_score(final_labels, final_preds, average=average)


def train_model(model, train_loader, valid_loader, test_loaders, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    model.to(device)
    max_val_f1, max_val_f1_vote = 0, 0
    max_test_metrics = {name: {'max_f1': 0, 'max_f1_micro': 0, 'max_f1_vote': 0,
                               'max_f1_micro_vote': 0} for name in ['test1', 'test2', 'test3']}

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 100)

        model.train()
        train_loss, train_size = 0.0, 0
        train_true_labels, train_pred_labels = [], []
        cnt = 0
        seg_ids = []
        for texts, labels, seg_id in tqdm.tqdm(train_loader, desc="Training"):
            texts, labels = texts.to(device), labels.to(device)
            # wandb.log({})
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * texts.size(0)
            train_size += texts.size(0)
            # if (cnt + 1) % 100 == 0:
            wandb.log(
                {"lr": optimizer.param_groups[0]['lr'], "train_iter_loss": train_loss / train_size})
            cnt += 1
            _, predicted = torch.max(outputs.data, 1)
            train_true_labels.extend(labels.cpu().numpy())
            train_pred_labels.extend(predicted.cpu().numpy())
            seg_ids.extend(seg_id)

        # f1_score(train_true_labels, train_pred_labels, average='macro')
        train_f1_macro, train_f1_macro_vote = calc_scores(
            train_true_labels, train_pred_labels, average='macro', seg_ids=seg_ids)
        # f1_score(train_true_labels, train_pred_labels, average='micro')
        train_f1_micro, train_f1_micro_vote = calc_scores(
            train_true_labels, train_pred_labels, average='micro', seg_ids=seg_ids)
        print(
            f"train loss:{train_loss / train_size}, train_f1_macro: {train_f1_macro}, train_f1_micro: {train_f1_micro}")
        print(
            f"vote_train_f1_macro: {train_f1_macro_vote}, train_f1_micro: {train_f1_micro_vote}")

        wandb.log({"train loss": train_loss / train_size, "train_f1_macro": train_f1_macro, "train_f1_micro": train_f1_micro,
                   "vote_train_f1_macro": train_f1_macro_vote, "vote_train_f1_micro": train_f1_micro_vote})

        model.eval()
        val_loss, val_size = 0.0, 0
        val_true_labels, val_pred_labels = [], []
        seg_ids = []
        with torch.no_grad():
            for texts, labels, seg_id in tqdm.tqdm(valid_loader, desc="Validating"):
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * texts.size(0)
                val_size += texts.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_true_labels.extend(labels.cpu().numpy())
                val_pred_labels.extend(predicted.cpu().numpy())
                seg_ids.extend(seg_id)

        # f1_score(val_true_labels, val_pred_labels, average='macro')
        val_f1_macro, val_f1_macro_vote = calc_scores(
            val_true_labels, val_pred_labels, average='macro', seg_ids=seg_ids)
        # f1_score(val_true_labels, val_pred_labels, average='micro')
        val_f1_micro, val_f1_micro_vote = calc_scores(
            val_true_labels, val_pred_labels, average='micro', seg_ids=seg_ids)

        print(f"val_f1_macro: {val_f1_macro}, val_f1_micro: {val_f1_micro}")
        print(
            f"vote_val_f1_macro: {val_f1_macro_vote}, val_f1_micro: {val_f1_micro_vote}")

        wandb.log({"val_f1_macro": val_f1_macro, "val_f1_micro": val_f1_micro,
                  "vote_val_f1_macro": val_f1_macro_vote, "vote_val_f1_micro": val_f1_micro_vote})

        model.eval()
        for loader, name in zip(test_loaders, ['test1', 'test2', 'test3']):
            test_loss, test_size = 0.0, 0
            test_true_labels = []
            test_pred_labels = []
            seg_ids = []
            test_logits = []
            with torch.no_grad():
                for texts, labels, seg_id in loader:
                    texts, labels = texts.to(device), labels.to(device)
                    outputs = model(texts)
                    logits = outputs.detach().cpu().numpy()
                    test_pred_labels += np.argmax(logits,
                                                  axis=1).flatten().tolist()
                    test_true_labels += labels.cpu().tolist()
                    test_logits += logits.tolist()
                    loss = criterion(outputs, labels)
                    test_loss += loss.item() * texts.size(0)
                    test_size += texts.size(0)
                    seg_ids.extend(seg_id)

            # f1_score(test_true_labels, test_pred_labels, average='macro')
            test_f1_macro, test_f1_macro_vote = calc_scores(
                test_true_labels, test_pred_labels, average='macro', seg_ids=seg_ids)
            # f1_score(test_true_labels, test_pred_labels, average='micro')
            test_f1_micro, test_f1_micro_vote = calc_scores(
                test_true_labels, test_pred_labels, average='micro', seg_ids=seg_ids)

            print(
                f"{name} - test_f1_macro: {test_f1_macro}, test_f1_micro: {test_f1_micro}")
            print(
                f"{name} - vote test_f1_macro: {test_f1_macro_vote}, test_f1_micro: {test_f1_micro_vote}")

            wandb.log({f"{name}_test_f1_macro": test_f1_macro, f"{name}_test_f1_micro": test_f1_micro,
                       f"{name}_vote_test_f1_macro": test_f1_macro_vote, f"{name}_vote_test_f1_micro": test_f1_micro_vote})

            if val_f1_macro >= max_val_f1:
                max_val_f1 = val_f1_macro
                max_test_metrics[name]['max_f1'], max_test_metrics[name]['max_f1_micro'] = test_f1_macro, test_f1_micro
                # store three best model
                best_test_details[name]['max_f1'] = test_f1_macro
                best_test_details[name]['pred_labels'] = test_pred_labels.copy()
            if val_f1_macro_vote >= max_val_f1_vote:
                max_val_f1_vote = val_f1_macro_vote
                max_test_metrics[name]['max_f1_vote'], max_test_metrics[name][
                    'max_f1_micro_vote'] = test_f1_macro_vote, test_f1_micro_vote
                # store three best model
                best_test_details[name]['max_f1_vote'] = test_f1_macro_vote
                best_test_details[name]['pred_labels_vote'] = test_pred_labels.copy(
                )

            best_test_details[name]['true_labels'] = test_true_labels.copy()

            print(
                f"{name} - max_f1: {max_test_metrics[name]['max_f1']:.4f}, max_f1_micro: {max_test_metrics[name]['max_f1_micro']:.4f}")
            print(
                f"{name} - vote max_f1: {max_test_metrics[name]['max_f1_vote']:.4f}, max_f1_micro: {max_test_metrics[name]['max_f1_micro_vote']:.4f}")
            wandb.log({f"{name}_max_f1": max_test_metrics[name]['max_f1'], f"{name}_max_f1_micro": max_test_metrics[name]['max_f1_micro'],
                       f"{name}_vote_max_f1": max_test_metrics[name]['max_f1_vote'], f"{name}_vote_max_f1_micro": max_test_metrics[name]['max_f1_micro_vote']})

    for name in ['test1', 'test2', 'test3']:
        np.save(
            f"/graft3/code/tracy/data/predictions/{args.text_field}/{name}_best_true_labels.npy", best_test_details[name]['true_labels'])
        np.save(
            f"/graft3/code/tracy/data/predictions/{args.text_field}/{name}_best_pred_labels.npy", best_test_details[name]['pred_labels'])
        np.save(
            f"/graft3/code/tracy/data/predictions/{args.text_field}/{name}_best_pred_labels_vote.npy", best_test_details[name]['pred_labels_vote'])

    torch.cuda.empty_cache()
    return model


if __name__ == "__main__":
    train_model(model, train_loader, valid_loader, [
                test_loader, test2_loader, test3_loader], device, epochs=args.epochs)
