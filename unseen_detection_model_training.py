from torch import nn
from torch.utils.data import DataLoader, RandomSampler

import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import json

import random
from tqdm import tqdm

from efficientnet_pytorch import EfficientNet
from bengali_classifier_model import BengalClassifier
from data_preprocess import *

BATCH_SIZE = 24
EPOCH = 20
TQDM_DISABLE = False

print('Start Out of Distribution Detection training.')
torch.cuda.empty_cache()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

# Load Data
images = load_images_df([
    'data/given_data/train_image_data_0.parquet',
    'data/given_data/train_image_data_1.parquet',
    'data/given_data/train_image_data_2.parquet',
    'data/given_data/train_image_data_3.parquet'])

info = pd.read_csv('data/given_data/train.csv')
multi_diacritics_info = pd.read_csv('data/given_data/train_multi_diacritics.csv')
info = info.set_index('image_id')
multi_diacritics_train_info = multi_diacritics_info.set_index('image_id')
info.update(multi_diacritics_info)

print('Done loading data.')

# Split Train / Valid Data
# Split Seen / Unseen Data
info['class_label'] = (info['grapheme_root'] * NUM_VOWEL_DIACRITIC + info['vowel_diacritic']) * NUM_CONSONANT_DIACRITIC + info['consonant_diacritic']
class_label_set = set(info['class_label'].tolist())
random.seed(1)
unseen_class_label_set = random.sample(class_label_set, int(len(class_label_set) * 0.1))
num_class = len(class_label_set)  # 1292
num_seen_class = len(class_label_set) - len(unseen_class_label_set)
print('Total number of different graphemes presented in given data is:{}.'.format(num_class))

# One Hot Encoder
ohe = OneHotEncoder(sparse = False)
ohe.fit(info['class_label'].to_numpy().reshape(-1,1))

seen_ohl = np.ones((1,len(class_label_set)))
for unseen_class_label in unseen_class_label_set:
    unseen_ohl = ohe.transform(np.array(unseen_class_label).reshape(-1, 1))
    seen_ohl -= unseen_ohl
seen_ohl = seen_ohl[0]

is_seen = [1 for _ in range(info.shape[0])]
for i in range(info.shape[0]):
    if info['class_label'][i] in unseen_class_label_set:
        is_seen[i] = 0

info['is_seen'] = is_seen
images['is_seen'] = is_seen
unseen_valid_info = info[info['is_seen'] == 0]
seen_info = info[info['is_seen'] == 1]
unseen_valid_images = images[images['is_seen'] == 0]
seen_images = images[images['is_seen'] == 1]

seen_train_images, seen_valid_images, seen_train_info, seen_valid_info = train_test_split(seen_images, seen_info, test_size = 0.1, random_state = 42)
valid_images = pd.concat([seen_valid_images, unseen_valid_images])
valid_info = pd.concat([seen_valid_info, unseen_valid_info])

images_np = image_df2np(images)
valid_images_np = image_df2np(valid_images)

print('Done splitting train/valid data.')

# Create Dataset

class OODD_Dataset(torch.utils.data.Dataset):

    def __init__(self, info, images, transform=None):
        self.info = info
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        np_image = self.images[idx].copy()
        out_image = self.transform(np_image)
        class_label = self.info['class_label'][idx]
        is_seen = self.info['is_seen'][idx]
        return out_image, class_label, is_seen

train_dataset = OODD_Dataset(info, images_np, SVHN_transform)
valid_dataset = OODD_Dataset(valid_info, valid_images_np, no_transform)

train_sampler = RandomSampler(train_dataset, True, int(len(train_dataset))*(EPOCH))
valid_sampler = RandomSampler(valid_dataset, True, int(len(valid_dataset))*(EPOCH))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=True, sampler=train_sampler)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=True, sampler=valid_sampler)

train_loader_iter = iter(train_loader)
valid_loader_iter = iter(valid_loader)

print('Done creating dataset & dataloader.')

# Create Model
backbone = EfficientNet.from_pretrained('efficientnet-b7')
model = BengalClassifier(backbone, hidden_size = 2560, class_num = num_class).to(device)

print('Done creating model.')

# Training


def train_step(model, train_loader_iter, criterion, optimizer, scheduler, device):
    image, class_label, seen_label = next(train_loader_iter)
    image = image.to(device)
    class_label = class_label.to('cpu')
    one_hot_label = torch.from_numpy(ohe.transform (class_label.reshape(-1,1)))
    one_hot_label = one_hot_label.to(device)

    optimizer.zero_grad()

    out = model(image)
    sig_out = out.sigmoid()
    loss = criterion(out, one_hot_label)
    train_loss_position = (1 - one_hot_label) * (sig_out.detach() > 0.1) + one_hot_label
    loss = ((loss * train_loss_position).sum(dim=1) / train_loss_position.sum(dim=1)).mean()

    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss


def valid_step(model, valid_loader_iter, device):
    image, class_label, seen_label = next(valid_loader_iter)
    image = image.to(device)
    with torch.no_grad():
        out = model(image)
        out = out.sigmoid()
        out = out.to('cpu').numpy()
        out = out * seen_ohl
        seen_pred = out.max(axis=1)

    seen_label = seen_label.to('cpu').numpy()
    return seen_pred, seen_label


optimizer = torch.optim.AdamW(model.parameters())
criterion = nn.BCEWithLogitsLoss(reduction = 'none')

num_train_step_per_epoch = len(train_loader)//EPOCH
num_valid_step_per_epoch = len(valid_loader)//EPOCH
train_steps = num_train_step_per_epoch*EPOCH
WARM_UP_STEP = train_steps*0.5


def warmup_linear_decay(step):
    if step < WARM_UP_STEP:
        return step/WARM_UP_STEP
    else:
        return (train_steps-step)/(train_steps-WARM_UP_STEP)


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_linear_decay)


log = []
best_score = 0.
for epoch in range(EPOCH):
    metric = {}
    metric['epoch'] = epoch

    model.train()
    tot_loss = []
    for i in tqdm(range(num_train_step_per_epoch), disable=TQDM_DISABLE):
        loss = train_step(model,
                          train_loader_iter,
                          criterion,
                          optimizer,
                          scheduler,
                          device)
        tot_loss.append(loss.item())
    metric['train_loss'] = sum(tot_loss) / len(tot_loss)

    model.eval()
    seen_preds = []
    seen_labels = []
    for i in tqdm(range(num_valid_step_per_epoch), disable=TQDM_DISABLE):
        seen_pred, seen_label = valid_step(model, valid_loader_iter, device)

        seen_preds.append(seen_pred)
        seen_labels.append(seen_label)

    seen_preds = np.concatenate(seen_preds)
    seen_labels = np.concatenate(seen_labels)
    auroc = sklearn.metrics.roc_auc_score(seen_labels, seen_preds)
    metric['valid_auroc'] = auroc

    print(metric)
    log.append(metric)

    if auroc > best_score:
        best_score = auroc
        torch.save(model.state_dict(), 'unseen_detection_result/best.pth')
    torch.save(model.state_dict(), 'unseen_detection_result/model.pth')
    with open('unseen_detection_result/log.json', 'w') as fout:
        json.dump(log, fout, indent=4)