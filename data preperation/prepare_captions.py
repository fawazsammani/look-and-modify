import os
import numpy as np
import h5py
import json
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import skimage.transform
from PIL import Image

output_folder = 'caption data'
min_word_freq = 5
captions_per_image = 5
dataset = 'coco'
max_len = 50
word_freq = Counter()

# Read Karpathy JSON
with open('caption data/dataset_coco.json', 'r') as j:
    data = json.load(j)
    
# Construct Wordmap
with open('caption data/Instances_train.json', 'r') as j:
    train_ins = json.load(j)

with open('caption data/Instances_val.json', 'r') as j:
    val_ins = json.load(j)
    
attributes = {**train_ins, **val_ins}

for value in attributes.values():
    word_freq.update(value)
    
for img in data['images']:
    for c in img['sentences']:
        word_freq.update(c['tokens']) 
        
# Create word map
words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
word_map = {k: v + 1 for v, k in enumerate(words)}
word_map['<unk>'] = len(word_map) + 1
word_map['<start>'] = len(word_map) + 1
word_map['<end>'] = len(word_map) + 1
word_map['<pad>'] = 0

# Save word map to a JSON
with open(os.path.join(output_folder, 'WORDMAP_' + dataset + '.json'), 'w') as j:
    json.dump(word_map, j)

# Read image paths and captions for each image
train_image_paths = []
train_image_names = []
train_image_captions = []
val_image_paths = []
val_image_names = []
val_image_captions = []
test_image_paths = []
test_image_captions = []
test_image_names = []

for img in data['images']:
    captions = []
    for c in img['sentences']:
        if len(c['tokens']) <= max_len:
            captions.append(c['tokens'])      
            
    if len(captions) == 0:
        continue
    
    path = os.path.join('images', img['filepath'], img['filename']) 

    if img['split'] in {'train', 'restval'}:
        train_image_paths.append(path)
        train_image_captions.append(captions)
        train_image_names.append(img['filename'])
    elif img['split'] in {'val'}:
        val_image_paths.append(path)
        val_image_captions.append(captions)
        val_image_names.append(img['filename'])
    elif img['split'] in {'test'}:
        test_image_paths.append(path)
        test_image_captions.append(captions)
        test_image_names.append(img['filename'])

# Sanity check
assert len(train_image_paths) == len(train_image_captions)
assert len(val_image_paths) == len(val_image_captions)
assert len(test_image_paths) == len(test_image_captions)
assert len(train_image_names) == len(train_image_paths)
assert len(val_image_names) == len(val_image_paths)
assert len(test_image_names) == len(test_image_paths)


with open(os.path.join(output_folder, 'TRAIN' + '_names_' + dataset + '.json'), 'w') as j:
    json.dump(train_image_names, j)

with open(os.path.join(output_folder, 'VAL' + '_names_' + dataset + '.json'), 'w') as j:
    json.dump(val_image_names, j)
    
with open(os.path.join(output_folder, 'TEST' + '_names_' + dataset + '.json'), 'w') as j:
    json.dump(test_image_names, j)
    
# Load Wordmap file
with open(os.path.join(output_folder, 'WORDMAP_' + dataset + '.json'), 'r') as j:
    word_map = json.load(j)

    
for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                               (val_image_paths, val_image_captions, 'VAL'),
                               (test_image_paths, test_image_captions, 'TEST')]:

    with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + dataset + '.hdf5'), 'a') as h:
        # Make a note of the number of captions we are sampling per image
        h.attrs['captions_per_image'] = captions_per_image

        # Create dataset inside HDF5 file to store images
        images = h.create_dataset('images', (len(impaths), 3, 512, 512), dtype='uint8')

        print("\nReading %s images and captions, storing to file...\n" % split)

        enc_captions = []
        caplens = []

        for i, path in enumerate(tqdm(impaths)):

            # Sample captions
            if len(imcaps[i]) < captions_per_image:
                captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
            else:
                captions = sample(imcaps[i], k=captions_per_image)

            # Sanity check
            assert len(captions) == captions_per_image

            # Read images
            img = imread(impaths[i])
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2)
            img = imresize(img, (512, 512))
            img = img.transpose(2, 0, 1)
            assert img.shape == (3, 512, 512)
            assert np.max(img) <= 255

            # Save image to HDF5 file
            images[i] = img

            for j, c in enumerate(captions):
                # Encode captions
                enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                # Find caption lengths
                c_len = len(c) + 2

                enc_captions.append(enc_c)
                caplens.append(c_len)

        # Sanity check
        assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

        # Save encoded captions and their lengths to JSON files
        with open(os.path.join(output_folder, split + '_CAPTIONS_' + dataset + '.json'), 'w') as j:
            json.dump(enc_captions, j)

        with open(os.path.join(output_folder, split + '_CAPLENS_' + dataset + '.json'), 'w') as j:
            json.dump(caplens, j)
