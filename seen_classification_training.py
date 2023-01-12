from torch.utils.data import DataLoader, RandomSampler

from tqdm import tqdm
import json
import sklearn.metrics
from sklearn.model_selection import train_test_split

from data_preprocess import *
from bengali_classifier_model import *
from efficientnet_pytorch import EfficientNet


BATCH_SIZE = 24
EPOCH = 200
TQDM_DISABLE = False

print('Start seen-classifier training.')
torch.cuda.empty_cache()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

# Load Data
info = pd.read_csv('data/given_data/train.csv')
multi_diacritics_info = pd.read_csv('data/given_data/train_multi_diacritics.csv')
info = info.set_index('image_id')
multi_diacritics_train_info = multi_diacritics_info.set_index('image_id')
info.update(multi_diacritics_info)

images = load_images_np([
    'data/given_data/train_image_data_0.parquet',
    'data/given_data/train_image_data_1.parquet',
    'data/given_data/train_image_data_2.parquet',
    'data/given_data/train_image_data_3.parquet'
])

print('Done loading font data.')

# Split Train/Test Data
train_images, valid_images, train_info, valid_info = train_test_split(images, info, test_size = 0.1, random_state = 42)

# Create Dataset
train_dataset = GraphemeDataset(train_info, train_images, SVHN_transform)
valid_dataset = GraphemeDataset(valid_info, valid_images, no_transform)

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

backbone = EfficientNet.from_pretrained('efficientnet-b7')
classifier = BengalClassifier(backbone, hidden_size = 2560, class_num = NUM_GRAPHEME_ROOT * NUM_VOWEL_DIACRITIC * NUM_CONSONANT_DIACRITIC).to(device)

classifier_loss = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(classifier.parameters())

num_step_per_epoch = len(train_loader) // EPOCH
num_valid_step_per_epoch = len(valid_loader) // EPOCH
train_steps = num_step_per_epoch * EPOCH
WARM_UP_STEP = train_steps * 0.5

def warmup_linear_decay(step):
    if step < WARM_UP_STEP:
        return step/WARM_UP_STEP
    else:
        return (train_steps-step)/(train_steps-WARM_UP_STEP)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_linear_decay)

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
        torch.save(classifier.state_dict(), 'result/seen_classification_result/seen_classifier.pth')
    with open('result/seen_classification_result/seen_classifier_log.json', 'w') as fout:
        json.dump(log, fout, indent=4)