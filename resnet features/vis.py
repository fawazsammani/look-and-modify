import os
import numpy as np
import h5py
import json
import torch
import torch.nn as nn
from torch.nn import Parameter
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import skimage.transform
from PIL import Image


class COCOValidationDataset(Dataset):

    def __init__(self, transform=None):

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join('caption data', 'VAL_IMAGES_coco' + '.hdf5'), 'r')
        self.imgs = self.h['images']

        with open(os.path.join('caption data', 'VAL_names_coco' + '.json'), 'r') as j:
            self.image_names = json.load(j)
            
        with open(os.path.join('caption data', 'VAL_CAPUTIL_coco' + '.json'), 'r') as j:
            self.caption_util = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.image_names)

    def __getitem__(self, i):
        """
        returns:
        img: the image convereted into a tensor of shape (batch_size,3, 256, 256)
        sentence_embed: the sentence embedding of the caption from the transfered model of shape (batch_size, 512)
        sentence_attr: the 5 top instances of the image of shape (batch_size, 5)
        image_id: the respective id for the image of shape (batch_size, 1)
        """
        img = torch.FloatTensor(self.imgs[i] / 255.)
        img_name = self.image_names[i]
        
        if self.transform is not None:
            img = self.transform(img)
            
        sentence_embed = torch.FloatTensor(self.caption_util[img_name]['embedding'])
        sentence_attr = torch.LongTensor(self.caption_util[img_name]['attributes'])
        return img, sentence_embed, sentence_attr, img_name

    def __len__(self):
        return self.dataset_size


def getLockedDropoutMask(size, dropout):
    # size of the tensor to be dropped. For example: the output of each LSTM of shape (batch_size, hidden_size)
    # Inverted Dropout. At training time, divide by the keeping probability
    mask = torch.ones(size).bernoulli_(1 - dropout) / (1 - dropout)  
    return mask

class Encoder(nn.Module):

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.avgpool = nn.AvgPool2d(encoded_image_size)

    def forward(self, images):
        enc_image = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32) --> (batch_size, 2048,8,8)
        enc_image = self.adaptive_pool(enc_image)  # (batch_size, 2048, 14, 14)
        spatial_image = enc_image.permute(0, 2, 3, 1)  # (batch_size, 14, 14, 2048)
        # (batch_size,num_pixels, 2048) = (batch_size, 196, 2048)
        spatial_image = spatial_image.view(spatial_image.shape[0], -1, spatial_image.shape[3])  
        global_image = self.avgpool(enc_image).squeeze(3)   # (batch_size, 2048, 1)
        global_image = global_image.permute(0,2,1)
        image_features = torch.cat([spatial_image,global_image], dim = 1)
        return image_features

    def fine_tune(self, fine_tune):
        for p in self.resnet.parameters():
            p.requires_grad = False
        # len(list(resnet.children())) = 8. For out of memory issues, set to [7:]
        for c in list(self.resnet.children())[7:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class TopDownSentinel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TopDownSentinel, self).__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size, bias=True)
        self.x_gate = nn.Linear(input_size, hidden_size)
        self.h_gate = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, h_old, c_old):
        ht, ct = self.lstm_cell(x, (h_old, c_old))
        sen_gate = F.sigmoid(self.x_gate(x) + self.h_gate(h_old))
        st =  sen_gate * F.tanh(ct)
        return ht, ct, st

class Attention(nn.Module):

    def __init__(self, features_dim, decoder_dim, attention_dim, dropout=0.5):

        super(Attention, self).__init__()
        self.features_att = nn.Linear(features_dim, attention_dim) 
        self.decoder_att = nn.Linear(decoder_dim, attention_dim) 
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, image_features, decoder_hidden):
        """
        Forward propagation.
        :param image_features: encoded images, a tensor of dimension (batch_size, 36, features_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.features_att(image_features)  # (batch_size, 36, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.dropout(self.relu(att1 + att2.unsqueeze(1)))).squeeze(2)  # (batch_size, 36)
        alpha = self.softmax(att)  # (batch_size, 196)
        context = (image_features * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, features_dim)
        return alpha, context

class ResidualGating(nn.Module):

    def __init__(self, sentence_embed_size, hidden_size, att_dim):
        super(ResidualGating, self).__init__()
        self.sen_att = nn.Linear(sentence_embed_size, att_dim)
        self.sentinel_att = nn.Linear(hidden_size, att_dim)
        
        
    def forward(self, st, sentence_embed):
        mod_gate = F.sigmoid(self.sen_att(sentence_embed) + self.sentinel_att(st))
        known = mod_gate * sentence_embed
        return known
    
class AttributeDecode(nn.Module):
    def __init__(self, embed_dim):
        super(AttributeDecode, self).__init__()
        self.attr_decode1 = nn.Linear(embed_dim, embed_dim)
        self.attr_decode2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, attributes):
        """
        attributes of shape: (batch_size, num_attributs, embed_dim)
        """
        attributes = attributes.mean(1)    # (batch_size, embed_dim)
        attributes = F.tanh(self.attr_decode1(attributes))   # (batch_size, embed_dim)
        attributes = F.tanh(self.attr_decode2(attributes))   # (batch_size, embed_dim)
        return attributes
        
        
class DecoderWithAttention(nn.Module):

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, sentence_embed_dim, features_dim=2048, dropout=0.5):

        super(DecoderWithAttention, self).__init__()
        self.features_dim = features_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.attention = Attention(features_dim, decoder_dim, attention_dim)  # attention network
        self.residual_gating = ResidualGating(sentence_embed_dim, decoder_dim, attention_dim)
        self.attribute_decoding = AttributeDecode(embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.top_down_attention = TopDownSentinel(embed_dim + features_dim + decoder_dim + sentence_embed_dim, decoder_dim) 
        self.language_model = nn.LSTMCell(features_dim + decoder_dim + embed_dim, decoder_dim, bias=True)  # language model LSTMCell
        self.fc1 = nn.Linear(decoder_dim, sentence_embed_dim)
        self.fc2 = nn.Linear(sentence_embed_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc2.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self,batch_size):

        h = torch.zeros(batch_size,self.decoder_dim).to(device)  # (batch_size, decoder_dim)
        c = torch.zeros(batch_size,self.decoder_dim).to(device)
        return h, c

    def forward(self, attributes, sentence_embed, image_features, encoded_captions, caption_lengths):
        """
        attributes of shape: (batch_size, num_attributes)
        sentence_embed of shape: (batch_size, hidden_size)
        image_features of shape: (batch_size, 197, 2048)
        encoded captions of shape: (batch_size, max_len)
        caption_lengths of shape: (batch_size, 1)
        """

        batch_size = image_features.size(0)
        vocab_size = self.vocab_size

        # Flatten image
        image_features_mean = image_features[:,:-1,:].mean(1).to(device)  # (batch_size, 2048)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        image_features = image_features[sort_ind]
        image_features_mean = image_features_mean[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        attributes = attributes[sort_ind]
        sentence_embed = sentence_embed[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        attr_embed = self.embedding(attributes)   # (batch_size, num_attributs, embed_dim)
        attributes_decoded = self.attribute_decoding(attr_embed)   # (batch_size, embed_dim)

        # Initialize LSTM state
        h1, c1 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        h2, c2 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()
        locked_mask = getLockedDropoutMask(size = (batch_size, self.decoder_dim), dropout = 0.5)
        locked_mask = locked_mask.to(device)
        # Create tensors to hold word predicion scores
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)

        
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            locked_mask = locked_mask[:batch_size_t]
            top_down_input = torch.cat([h2[:batch_size_t],
                                        image_features_mean[:batch_size_t],
                                        embeddings[:batch_size_t, t, :],
                                        sentence_embed[:batch_size_t]],dim=1)
            h1,c1,st = self.top_down_attention(top_down_input, h1[:batch_size_t], c1[:batch_size_t])
            h1 = h1 * locked_mask
            attention_weighted_encoding = self.attention(image_features[:batch_size_t],h1[:batch_size_t])
            known = self.residual_gating(st[:batch_size_t],sentence_embed[:batch_size_t])
            lan_input = torch.cat([attention_weighted_encoding[:batch_size_t],
                                   h1[:batch_size_t],
                                   attributes_decoded[:batch_size_t]], dim=1)
            h2,c2 = self.language_model(lan_input,(h2[:batch_size_t], c2[:batch_size_t]))
            h2 = h2 * locked_mask
            residual = F.tanh(self.fc1(h2))
            add_residual = residual + known
            preds = self.fc2(self.dropout(add_residual))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds

        return predictions,encoded_captions, decode_lengths, sort_ind

def caption_beamsearch(img, sentence_embed, sentence_attr, encoder, decoder, beam_size, vocab_size):
    
    decoder.eval()
    encoder.eval()
    k = beam_size
    infinite_pred = False

    # Move to GPU device, if available
    image_features = encoder(img.to(device))
    sentence_embed = sentence_embed.to(device) # (1,512)
    sentence_attr = sentence_attr.to(device)  # (1,5)

    num_features = image_features.shape[1]       
    encoder_dim = image_features.shape[2]  

    attr_embed = decoder.embedding(sentence_attr)           # (1, num_attributes, embed_dim)
    attributes_decoded = decoder.attribute_decoding(attr_embed)   # (1, embed_dim)

    image_features = image_features.expand(k, num_features, encoder_dim)  # (k, 36, encoder_dim)
    image_features_mean = image_features[:,:-1,:].mean(1)
    image_features_mean = image_features_mean.expand(k,2048)
    sentence_embed = sentence_embed.expand(k, sentence_embed.shape[1])   # (k, 512)
    attributes_decoded = attributes_decoded.expand(k, attributes_decoded.shape[1])   # (k, embed_dim)
    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
    seqs_alpha = torch.ones(k, 1, 14, 14).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences and scores
    complete_seqs = list()
    complete_seqs_scores = list()
    complete_seqs_alpha = list()

    # Start decoding
    step = 1
    h1, c1 = decoder.init_hidden_state(k)  # (batch_size, decoder_dim)
    h2, c2 = decoder.init_hidden_state(k)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
        top_down_input = torch.cat([h2,image_features_mean,embeddings,sentence_embed], dim = 1)
        h1,c1,st = decoder.top_down_attention(top_down_input, h1,c1)
        alpha, attention_weighted_encoding = decoder.attention(image_features,h1)
        known = decoder.residual_gating(st,sentence_embed)
        h2,c2 = decoder.language_model(torch.cat([attention_weighted_encoding,h1,attributes_decoded], dim=1),(h2,c2))
        residual = F.tanh(decoder.fc1(h2))
        add_residual = residual + known
        scores = decoder.fc2(add_residual)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)
        alpha = alpha[:,:-1]
        alpha = alpha.view(-1, 14, 14)  # (s, enc_image_size, enc_image_size)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim = 1) # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h1 = h1[prev_word_inds[incomplete_inds]]
        c1 = c1[prev_word_inds[incomplete_inds]]
        h2 = h2[prev_word_inds[incomplete_inds]]
        c2 = c2[prev_word_inds[incomplete_inds]]
        sentence_embed = sentence_embed[prev_word_inds[incomplete_inds]]
        attributes_decoded = attributes_decoded[prev_word_inds[incomplete_inds]]
        image_features = image_features[prev_word_inds[incomplete_inds]]
        image_features_mean = image_features_mean[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            infinite_pred = True
            break
        step += 1

    if infinite_pred is not True:
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        alphas = complete_seqs_alpha[i]
    else:
        seq = seqs[0][:20]
        seq = [seq[i].item() for i in range(len(seq))]
        alphas = None
    
    return seq, alphas

# Read word map
with open('caption data/WORDMAP_coco.json', 'r') as j:
    word_map = json.load(j)
    
rev_word_map = {v: k for k, v in word_map.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('BEST_checkpoint_31.pth.tar')
decoder = checkpoint['decoder']
decoder = decoder.to(device)
encoder= checkpoint['encoder']
encoder = encoder.to(device)


val_loader = torch.utils.data.DataLoader(COCOValidationDataset(),
                                         batch_size = 1,
                                         shuffle=True, 
                                         pin_memory=True)


with open('caption data/VAL_CAPUTIL_coco.json', 'r') as k:
    cap_util = json.load(k)


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)
    words = [rev_word_map[ind] for ind in seq if ind!=word_map['<end>']]
    alphas = torch.FloatTensor(alphas)
    print("Modified Caption:" + ' '.join(words[1:]))
    for t in range(len(words)):
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)
#         if t>0:
#             plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]                  
        alpha = skimage.transform.pyramid_expand(current_alpha.detach().numpy(), upscale=24, sigma=8)
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.5)
        plt.set_cmap('hot')
        plt.axis('off')
    plt.show()

    
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (12, 12)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
count = 0
for (img, sentence_embed, sentence_attr, image_name) in val_loader:
    count+=1
    print("Existing Caption:" + cap_util[image_name[0]]['caption'])
    seq, alphas = caption_beamsearch(img, sentence_embed, sentence_attr, encoder, decoder, beam_size = 3, vocab_size = len(word_map))
    image_path = os.path.join("val2014", image_name[0])
    visualize_att(image_path, seq, alphas, rev_word_map, smooth=True)
    if count == 40:
        break

