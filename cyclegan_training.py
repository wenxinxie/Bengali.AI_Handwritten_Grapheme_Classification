import json
import itertools
from tqdm import tqdm

from data_preprocess import  *
from font_classifier_training import BengalClassifier
from efficientnet_pytorch import EfficientNet
from cyclegan_model import *

BATCH_SIZE = 6
EPOCH = 40
TQDM_DISABLE = False # progeress bar

print('Start cycleGAN training.')
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} for inference.')

font_num = 0
Font_DIR = 'data/font_data/font_{}'
Result_DIR = 'result/font_generator_result/font{}'
# Load Data

font_info = load_info([
    Font_DIR.format(font_num) + '/font_{}_info_0.csv'.format(font_num),
    Font_DIR.format(font_num) + '/font_{}_info_1.csv'.format(font_num),
    Font_DIR.format(font_num) + '/font_{}_info_2.csv'.format(font_num),
    Font_DIR.format(font_num) + '/font_{}_info_3.csv'.format(font_num)
])

font_images = load_images([
    Font_DIR.format(font_num) + '/font_{}_image_0.parquet'.format(font_num),
    Font_DIR.format(font_num) + '/font_{}_image_1.parquet'.format(font_num),
    Font_DIR.format(font_num) + '/font_{}_image_2.parquet'.format(font_num),
    Font_DIR.format(font_num) + '/font_{}_image_3.parquet'.format(font_num)
])

print('Done loading font data.')

train_info = pd.read_csv('data/given_data/train.csv')
multi_diacritics_info = pd.read_csv('data/given_data/train_multi_diacritics.csv')
train_info = train_info.set_index('image_id')
multi_diacritics_train_info = multi_diacritics_info.set_index('image_id')
train_info.update(multi_diacritics_info)

train_images = load_images_np([
    'data/given_data/train_image_data_0.parquet',
    'data/given_data/train_image_data_1.parquet',
    'data/given_data/train_image_data_2.parquet',
    'data/given_data/train_image_data_3.parquet',
])

print('Done loading train data.')

# Create Dataset
hand_dataset = GraphemeDataset(train_info, train_images, valid_transform)
font_dataset = GraphemeDataset(font_info, font_images, train_transform)

hand_sampler = torch.utils.data.RandomSampler(hand_dataset, True, int(max(len(hand_dataset), len(font_dataset)))*(EPOCH))
font_sampler = torch.utils.data.RandomSampler(font_dataset, True, int(max(len(hand_dataset), len(font_dataset)))*(EPOCH))

hand_loader = torch.utils.data.DataLoader(
    hand_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    pin_memory=True, 
    drop_last=True, 
    sampler=hand_sampler)
font_loader = torch.utils.data.DataLoader(
    font_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    pin_memory=True, 
    drop_last=True, 
    sampler=font_sampler)

hand_loader_iter = iter(hand_loader)
font_loader_iter = iter(font_loader)

# Build CycleGAN
norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
generator_a = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, norm_layer=norm_layer, use_dropout=False, n_blocks=9)
generator_b = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, norm_layer=norm_layer, use_dropout=False, n_blocks=9)
discriminator_a = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3, norm_layer=norm_layer)
discriminator_b = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3, norm_layer=norm_layer)
init_weight(generator_a, 0.02)
init_weight(generator_b, 0.02)
init_weight(discriminator_a, 0.02)
init_weight(discriminator_b, 0.02)
discriminator_loss = GANLoss('lsgan', target_real_label=1.0, target_fake_label=0.0)

backbone = EfficientNet.from_name('efficientnet-b0')
classifier = BengalClassifier(backbone, hidden_siz = 1280, class_num = NUM_GRAPHEME_ROOT * NUM_VOWEL_DIACRITIC * NUM_CONSONANT_DIACRITIC)
classifier.load_state_dict(torch.load('font_classification_result/font0/warm-up&expodecay0.04/font_classifier.pth'))
classifier_loss = nn.CrossEntropyLoss()

class CycleGan(nn.Module):

    def __init__(self,
                 generator_a, generator_b, discriminator_a, discriminator_b, classifier,
                 discriminator_loss, classifier_loss,
                 lambda_a, lambda_b, lambda_cls,
                 device):
        super(CycleGan, self).__init__()
        self.generator_a = generator_a
        self.generator_b = generator_b
        self.discriminator_a = discriminator_a
        self.discriminator_b = discriminator_b
        self.classifier = classifier.eval()
        CycleGan.set_requires_grad(self.classifier, requires_grad=False)
        self.discriminator_loss = discriminator_loss
        self.classifier_loss = classifier_loss
        self.reconstruct_loss = nn.L1Loss()
        self.device = device

        self.image_pool_a = ImagePool(50)
        self.image_pool_b = ImagePool(50)

        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        self.lambda_cls = lambda_cls

        self.real_images_a = None
        self.real_images_b = None
        self.labels_a = None
        self.labels_b = None
        self.fake_images_a = None
        self.fake_images_b = None
        self.rec_images_a = None
        self.rec_images_b = None
        self.generator_a = torch.nn.DataParallel(self.generator_a)
        self.generator_b = torch.nn.DataParallel(self.generator_b)
        self.discriminator_a = torch.nn.DataParallel(self.discriminator_a)
        self.discriminator_b = torch.nn.DataParallel(self.discriminator_b)
        self.to(device)

    def forward(self):
        self.fake_images_a = self.generator_a(self.real_images_b)
        self.fake_images_b = self.generator_b(self.real_images_a)
        self.rec_images_a = self.generator_a(self.fake_images_b)
        self.rec_images_b = self.generator_b(self.fake_images_a)

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def generator_step(self):
        CycleGan.set_requires_grad([self.discriminator_a, self.discriminator_b], False)

        loss_gen_a = self.discriminator_loss(self.discriminator_a(self.fake_images_a), True)
        loss_gen_b = self.discriminator_loss(self.discriminator_b(self.fake_images_b), True)
        loss_cyc_a = self.reconstruct_loss(self.rec_images_a, self.real_images_a) * self.lambda_a
        loss_cyc_b = self.reconstruct_loss(self.rec_images_b, self.real_images_b) * self.lambda_b
        loss_cls = self.classifier_loss(self.classifier(self.fake_images_b), self.labels_a) * self.lambda_cls

        loss_gen = loss_gen_a + loss_gen_b + loss_cyc_a + loss_cyc_b + loss_cls
        loss_gen.backward()

        return loss_gen, loss_gen_a, loss_gen_b, loss_cyc_a, loss_cyc_b, loss_cls

    def discriminator_step(self):
        CycleGan.set_requires_grad([self.discriminator_a, self.discriminator_b], True)

        pred_real_a = self.discriminator_a(self.real_images_a)
        loss_real_a = self.discriminator_loss(pred_real_a, True)
        fake_images_a = self.image_pool_a.query(self.fake_images_a).detach()
        pred_fake_a = self.discriminator_a(fake_images_a)
        loss_fake_a = self.discriminator_loss(pred_fake_a, False)
        loss_dis_a = (loss_real_a + loss_fake_a) / 2

        pred_real_b = self.discriminator_b(self.real_images_b)
        loss_real_b = self.discriminator_loss(pred_real_b, True)
        fake_images_b = self.image_pool_b.query(self.fake_images_b).detach()
        pred_fake_b = self.discriminator_b(fake_images_b)
        loss_fake_b = self.discriminator_loss(pred_fake_b, False)
        loss_dis_b = (loss_real_b + loss_fake_b) / 2

        loss_dis =  loss_dis_a + loss_dis_b
        loss_dis.backward()
        return loss_dis, loss_real_a, loss_fake_a, loss_dis_a, loss_real_b, loss_fake_b, loss_dis_b

    def set_input(self, images_a, images_b, labels_a, labels_b):
        self.real_images_a = images_a.to(self.device)
        self.real_images_b = images_b.to(self.device)
        self.labels_a = labels_a
        self.labels_b = labels_b

model = CycleGan(generator_a=generator_a,
                 generator_b=generator_b,
                 discriminator_a=discriminator_a,
                 discriminator_b=discriminator_b,
                 classifier=classifier,
                 discriminator_loss=discriminator_loss,
                 classifier_loss=classifier_loss,
                 lambda_a=10.0,
                 lambda_b=10.0,
                 lambda_cls=4.0,
                 device=device)

def train_step(model, a_iter, b_iter, generator_optimizer, discriminator_optimizer, generator_scheduler,
               discriminator_scheduler, device):
    a_image, a_label = next(a_iter)
    b_image, b_label = next(b_iter)
    a_image = a_image.to(device)
    b_image = b_image.to(device)
    a_label = a_label.to(device)
    b_label = b_label.to(device)
    model.set_input(a_image, b_image, a_label, b_label)
    model.forward()

    generator_optimizer.zero_grad()
    loss_gen, loss_gen_a, loss_gen_b, loss_cyc_a, loss_cyc_b, loss_cls = model.generator_step()
    generator_optimizer.step()
    generator_scheduler.step()

    discriminator_optimizer.zero_grad()
    loss_dis, loss_real_a, loss_fake_a, loss_dis_a, loss_real_b, loss_fake_b, loss_dis_b = model.discriminator_step()
    discriminator_optimizer.step()
    discriminator_scheduler.step()

    return loss_gen, loss_gen_a, loss_gen_b, loss_cyc_a, loss_cyc_b, loss_cls, loss_dis, loss_real_a, loss_fake_a, loss_dis_a, loss_real_b, loss_fake_b, loss_dis_b


# optimizer
generator_optimizer = torch.optim.Adam(itertools.chain(generator_a.parameters(), generator_b.parameters()), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = torch.optim.Adam(itertools.chain(discriminator_a.parameters(), discriminator_b.parameters()), lr=0.0002, betas=(0.5, 0.999))

num_step_per_epoch = len(hand_loader) // EPOCH
train_steps = num_step_per_epoch * EPOCH
WARM_UP_STEP = train_steps * 0.5

def warmup_linear_decay(step):
    if step < WARM_UP_STEP:
        return 1.0
    else:
        return (train_steps - step) / (train_steps - WARM_UP_STEP)

generator_scheduler = torch.optim.lr_scheduler.LambdaLR(generator_optimizer, warmup_linear_decay)
discriminator_scheduler = torch.optim.lr_scheduler.LambdaLR(discriminator_optimizer, warmup_linear_decay)


class LossAverager:
    def __init__(self, prefix):
        self.prefix = prefix
        self.loss_gen = []
        self.loss_gen_a = []
        self.loss_gen_b = []
        self.loss_cyc_a = []
        self.loss_cyc_b = []
        self.loss_cls = []
        self.loss_dis = []
        self.loss_real_a = []
        self.loss_fake_a = []
        self.loss_dis_a = []
        self.loss_real_b = []
        self.loss_fake_b = []
        self.loss_dis_b = []

    def append(self, loss_gen, loss_gen_a, loss_gen_b, loss_cyc_a, loss_cyc_b, loss_cls, loss_dis, loss_real_a, loss_fake_a, loss_dis_a, loss_real_b, loss_fake_b, loss_dis_b):
        self.loss_gen.append(loss_gen.item()) # torch.tensor.item() Returns the value of this tensor as a standard Python number
        self.loss_gen_a.append(loss_gen_a.item())
        self.loss_gen_b.append(loss_gen_b.item())
        self.loss_cyc_a.append(loss_cyc_a.item())
        self.loss_cyc_b.append(loss_cyc_b.item())
        self.loss_cls.append(loss_cls.item())
        self.loss_dis.append(loss_dis.item())
        self.loss_real_a.append(loss_real_a.item())
        self.loss_fake_a.append(loss_fake_a.item())
        self.loss_dis_a.append(loss_dis_a.item())
        self.loss_real_b.append(loss_real_b.item())
        self.loss_fake_b.append(loss_fake_b.item())
        self.loss_dis_b.append(loss_dis_b.item())

    def average(self):
        metric = {}
        for key, value in self.__dict__.items():
            if isinstance(value, list):
                metric[self.prefix + '/' + key] = sum(value) / len(value)
        return metric

log = []
for epoch in range(EPOCH):
    metric = {}
    metric['epoch'] = epoch

    model.train()
    model.classifier.eval()
    loss_averager = LossAverager('train')
    for i in tqdm(range(num_step_per_epoch)):
        losses = train_step(model, hand_loader_iter, font_loader_iter, generator_optimizer, discriminator_optimizer,
                            generator_scheduler, discriminator_scheduler, device)
        loss_averager.append(*losses)
    metric['loss'] = loss_averager.average()

    model.eval()

    print(metric)
    log.append(metric)
    torch.save(generator_b.state_dict(), Result_DIR.format(font_num) + '/font{}_generator.pth'.format(font_num))
    with open(Result_DIR.format(font_num) + '/cyclegan_font{}_log.json'.format(font_num), 'w') as fout:
        json.dump(log, fout, indent=4)