import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from tqdm.notebook import tqdm
import random

WIDTH = 236
HEIGHT = 137
CENTER_X = WIDTH//2
CENTER_Y = HEIGHT//2

PREFIX = 'Font_'
DATASET_ROOT = 'data/font_data'

class_map = pd.read_csv('data/given_data/class_map_corrected.csv')
grapheme_root = class_map[class_map['component_type'] == 'grapheme_root']
vowel_diacritic = class_map[class_map['component_type'] == 'vowel_diacritic']
consonant_diacritic = class_map[class_map['component_type'] == 'consonant_diacritic']
grapheme_root_list = grapheme_root['component'].tolist()
vowel_diacritic_list = vowel_diacritic['component'].tolist()
consonant_diacritic_list = consonant_diacritic['component'].tolist()

def label_to_grapheme(grapheme_root, vowel_diacritic, consonant_diacritic):
    if consonant_diacritic == 0:
        if vowel_diacritic == 0:
            return grapheme_root_list[grapheme_root]
        else:
            return grapheme_root_list[grapheme_root] + vowel_diacritic_list[vowel_diacritic]
    elif consonant_diacritic == 1:
        if vowel_diacritic == 0:
            return grapheme_root_list[grapheme_root] + consonant_diacritic_list[consonant_diacritic]
        else:
            return grapheme_root_list[grapheme_root] + vowel_diacritic_list[vowel_diacritic] + consonant_diacritic_list[consonant_diacritic]
    elif consonant_diacritic == 2:
        if vowel_diacritic == 0:
            return consonant_diacritic_list[consonant_diacritic] + grapheme_root_list[grapheme_root]
        else:
            return consonant_diacritic_list[consonant_diacritic] + grapheme_root_list[grapheme_root] + vowel_diacritic_list[vowel_diacritic]
    elif consonant_diacritic == 3:
        if vowel_diacritic == 0:
            return consonant_diacritic_list[consonant_diacritic][:2] + grapheme_root_list[grapheme_root] + consonant_diacritic_list[consonant_diacritic][1:]
        else:
            return consonant_diacritic_list[consonant_diacritic][:2] + grapheme_root_list[grapheme_root] + consonant_diacritic_list[consonant_diacritic][1:] + vowel_diacritic_list[vowel_diacritic]
    elif consonant_diacritic == 4:
        if vowel_diacritic == 0:
            return grapheme_root_list[grapheme_root] + consonant_diacritic_list[consonant_diacritic]
        else:
            if grapheme_root == 123 and vowel_diacritic == 1:
                return grapheme_root_list[grapheme_root] + '\u200d' + consonant_diacritic_list[consonant_diacritic] + vowel_diacritic_list[vowel_diacritic]
            return grapheme_root_list[grapheme_root]  + consonant_diacritic_list[consonant_diacritic] + vowel_diacritic_list[vowel_diacritic]
    elif consonant_diacritic == 5:
        if vowel_diacritic == 0:
            return grapheme_root_list[grapheme_root] + consonant_diacritic_list[consonant_diacritic]
        else:
            return grapheme_root_list[grapheme_root] + consonant_diacritic_list[consonant_diacritic] + vowel_diacritic_list[vowel_diacritic]
    elif consonant_diacritic == 6:
        if vowel_diacritic == 0:
            return grapheme_root_list[grapheme_root] + consonant_diacritic_list[consonant_diacritic]
        else:
            return grapheme_root_list[grapheme_root] + consonant_diacritic_list[consonant_diacritic] + vowel_diacritic_list[vowel_diacritic]
    elif consonant_diacritic == 7:
        if vowel_diacritic == 0:
            return consonant_diacritic_list[2] + grapheme_root_list[grapheme_root] + consonant_diacritic_list[2][::-1]
        else:
            return consonant_diacritic_list[2] + grapheme_root_list[grapheme_root] + consonant_diacritic_list[2][::-1] + vowel_diacritic_list[vowel_diacritic]

def grapheme_to_image(grapheme, font):
    width, height = font.getsize(grapheme)
    image = Image.new(size=(WIDTH, HEIGHT), mode='L', color=255)
    draw = ImageDraw.Draw(image)
    x, y = CENTER_X-width//2, CENTER_Y-height//2
    draw.text((x, y), grapheme, font=font)
    return image

grapheme_to_label = {}
alphabet = []
for grapheme_root in range(168):
    for vowel_diacritic in range(11):
        for consonant_diacritic in range(8):
            grapheme_to_label[label_to_grapheme(grapheme_root, vowel_diacritic, consonant_diacritic)] = (grapheme_root, vowel_diacritic, consonant_diacritic)
            alphabet.append(label_to_grapheme(grapheme_root, vowel_diacritic, consonant_diacritic))

font_list = [
    'data/font_data/kalpurush.ttf',
    'data/font_data/NikoshLightBan.ttf']

size_list = [84,96,108,120]

idx = 0
raw_font_data = []
raw_pixel_data = []

for i in range(len(size_list)):
    for j in range(len(font_list)):
        size = size_list[i]
        font_path = font_list[j]
        font = ImageFont.truetype(font_path, layout_engine=ImageFont.LAYOUT_RAQM, size=size)

        for c in tqdm(alphabet):
            grapheme_root, vowel_diacritic, consonant_diacritic = grapheme_to_label[c]

            grapheme = label_to_grapheme(grapheme_root, vowel_diacritic, consonant_diacritic)
            image = grapheme_to_image(grapheme, font)
            image_id = PREFIX + str(idx)
            image_data = {
                'image_id': image_id,
                'grapheme_root': grapheme_root,
                'vowel_diacritic': vowel_diacritic,
                'consonant_diacritic': consonant_diacritic,
                'grapheme': grapheme
            }
            pixel_data = np.array(image).reshape(-1)
            raw_font_data.append(image_data)
            raw_pixel_data.append(pixel_data)
            idx += 1

        font_data = pd.DataFrame(raw_font_data)
        font_image_data = font_data[['image_id']]
        pixel_data = pd.DataFrame(np.array(raw_pixel_data), columns=[str(i) for i in range(WIDTH * HEIGHT)])
        font_image_data = font_image_data.join(pixel_data)

        font_data.to_csv(os.path.join(DATASET_ROOT, 'font_{}/font_{}_info_{}.csv'.format(j, j, i)), index=False)
        font_image_data.to_parquet(os.path.join(DATASET_ROOT, 'font_{}/font{}_image_{}.parquet'.format(j, j, i)))