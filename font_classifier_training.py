import math
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from efficientnet_pytorch import EfficientNet

from tqdm import tqdm
import sklearn.metrics
import json

from data_preprocess import *
from bengali_classifier_model import *
torch.cuda.empty_cache()

BATCH_SIZE = 72
EPOCH = 60
TQDM_DISABLE = False

print('Start font-classifier training.')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

font_num = 0
Font_DIR = 'data/font_data/font_{}'

# Load Data
font_info = load_info([
    Font_DIR.format(font_num) + '/font_{}_info_0.csv'.format(font_num),
    Font_DIR.format(font_num) + '/font_{}_info_1.csv'.format(font_num),
    Font_DIR.format(font_num) + '/font_{}_info_2.csv'.format(font_num),
    Font_DIR.format(font_num) + '/font_{}_info_3.csv'.format(font_num)
])

font_images = load_images_np([
    Font_DIR.format(font_num) + '/font_{}_image_0.parquet'.format(font_num),
    Font_DIR.format(font_num) + '/font_{}_image_1.parquet'.format(font_num),
    Font_DIR.format(font_num) + '/font_{}_image_2.parquet'.format(font_num),
    Font_DIR.format(font_num) + '/font_{}_image_3.parquet'.format(font_num)
])
print('Done loading font data.')

# Create Dataset
train_dataset = GraphemeDataset(font_info, font_images, train_transform)
valid_dataset = GraphemeDataset(font_info, font_images, valid_transform)

#Create Data Loader
train_sampler = RandomSampler(train_dataset, True, int(len(train_dataset)) * (EPOCH))
valid_sampler = RandomSampler(valid_dataset, True, int(len(valid_dataset)) * (EPOCH))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    drop_last=True,
    sampler=train_sampler)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    drop_last=True,
    sampler=valid_sampler)

train_loader_iter = iter(train_loader)
valid_loader_iter = iter(valid_loader)

# Create Model
backbone = EfficientNet.from_name('efficientnet-b0')
classifier = BengalClassifier(backbone, hidden_size = 1280, class_num = NUM_GRAPHEME_ROOT * NUM_VOWEL_DIACRITIC * NUM_CONSONANT_DIACRITIC).to(device)

classifier_loss = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(classifier.parameters())

num_step_per_epoch = len(train_loader) // EPOCH
num_valid_step_per_epoch = len(valid_loader) // EPOCH
train_steps = num_step_per_epoch * EPOCH
WARM_UP_STEP = train_steps * 0.2

def warmup_exp_decay(step):
    if step < WARM_UP_STEP:
        return 1.0
    else:
        return math.exp(- 0.04 *(step - WARM_UP_STEP))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_exp_decay)

#Training
def train_step(model, train_loader_iter, classifier_loss, optimizer, scheduler, device):
    image, label = next(train_loader_iter)
    image = image.to(device)
    label = label.to(device)

    optimizer.zero_grad()

    out = model(image)
    loss = classifier_loss(out, label)

    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss

def valid_step(model, valid_loader_iter, device):
    image, label = next(valid_loader_iter)
    image = image.to(device)

    with torch.no_grad():
        out = model(image)
        pred = out.argmax(dim=1).cpu().numpy()
        # returns the indices of the maximum values of a tensor across a dimension
        # returns a copy of this object in CPU memory

    label = label.numpy()
    return pred, label


log = []
best_score = 0.

for epoch in range(EPOCH):
    metric = {}
    metric['epoch'] = epoch

    classifier.train()
    tot_loss = []
    for i in tqdm(range(num_step_per_epoch), disable=TQDM_DISABLE):
        loss = train_step(classifier,
                          train_loader_iter,
                          classifier_loss,
                          optimizer,
                          scheduler,
                          device)
        tot_loss.append(loss.item())
    metric['train_loss'] = sum(tot_loss) / len(tot_loss)

    classifier.eval()
    preds = []
    labels = []
    for i in tqdm(range(num_valid_step_per_epoch), disable=TQDM_DISABLE):
        pred, label = valid_step(classifier, valid_loader_iter, device)

        preds.append(pred)
        labels.append(label)

    preds = np.concatenate(preds)
    # join a sequence of arrays along an existing axis
    labels = np.concatenate(labels)
    accuracy = sklearn.metrics.accuracy_score(labels, preds)
    # in multilabel classification, this function computes subset accuracy
    metric['valid_accuracy'] = accuracy

    print(metric)
    log.append(metric)

    if accuracy > best_score:
        best_score = accuracy
        torch.save(classifier.state_dict(), 'font_classifier.pth')
    with open('font_classifier_log.json', 'w') as fout:
        json.dump(log, fout, indent=4)