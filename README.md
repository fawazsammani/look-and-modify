## Look and Modify: Modification Networks for Image Captioning 
This is the official implementation of our BMVC2019 paper [Look and Modify: Modification Networks for Image Captioning](https://bmvc2019.org/wp-content/uploads/papers/0597-paper.pdf) | [arXiv](https://arxiv.org/abs/1909.03169)

![demo](https://user-images.githubusercontent.com/30661597/61649073-4cf21d00-ace3-11e9-8b71-0648a879c60c.png)

## Requirements
Python 3.6 and PyTorch 0.4

## Instructions for using Bottom-Up features and modifying captions from Top-Down model
Download the COCO 2014 dataset from [here](http://cocodataset.org/#download). In particualr, you'll need the 2014 Training and Validation images. <br/>
Then download Karpathy's Train/Val/Test Split. You may download it from [here](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).<br/>

If you want to do evaluation on COCO, download the COCO API from [here](https://github.com/cocodataset/cocoapi) if your on Linux or from [here](https://github.com/philferriere/cocoapi) if your on Windows. Then download the COCO caption toolkit from [here](https://github.com/tylin/coco-caption) and re-name the folder to `cococaptioncider`. Don't forget to download java as well. Simply dowload it from [here](https://www.java.com/en/download/) if you don't have it.

Next, download the bottom up image features from [here](https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip ). If you're modifiying captions from any framework that uses ResNet features (e.g Attend and Tell, Adaptive Attention), then you may skip this step. 
Use the instructions from [this](https://github.com/hengyuan-hu/bottom-up-attention-vqa) repo to extract the features and indices. 


The generated files should be placed in a folder called `bottom-up features`. 

Then download the caption data folder from [here](https://drive.google.com/open?id=1vuE0Tj1a1wH-Yh2G_i6Mh1lHiIMM9b7V) which includes the following: 

`Caption utilities`: a dictionary file with the following: `{"COCO image name": {"caption": "previous caption to modify", "embedding": [512-d DAN embedding of previous caption], "attributes": [the indiced of the 5 extracted attributes], "image_ids": the COCO image id}`

`COCO image names with IDs` in the following format: `["COCO_val2014_000000391895.jpg", 391895]`. This is basically for evaluation on COCO.

`Annotations`: The training annotations

`Caption lengths`: The training captions length

`Wordmap`: A dictionary to map the word to their corresponding indices

`Bottom up features mapping to images`: The mapping of the bottom up features to their corresponding COCO images in the corresponding order.


For more information on the preperation of this dataset, see the folder `data preperation`. Here, the previous captions are embedded first using Google's Universal Sentence Encoder available at TensorFlow Hub [here](https://tfhub.dev/google/universal-sentence-encoder/2) and are loaded to the model for faster processing. You can find how to extract the features from the sentence in the folder `data preperation`. If you would like to implement your own DAN, use the code provided in `util/dan.py` which makes use of the GLoVe word embeddings. You can download the 300-d 6B trained word vectors from [here](https://nlp.stanford.edu/projects/glove/) for use in this function. Moreover, you may ignore the `load embedding` function if you'd like to train the word vectors from scratch using `nn.Embedding`.

In our paper, we make use of variational dropout to effectively regularize our language model, which samples one mask and uses it repeatedly across all timesteps. In that case, all timesteps of the language model receive the same dropout mask. This implementation is included here as well. If you change the dimension of your LSTM hidden state, make sure to adjust accordingly in the `getLockedDropooutMask` function. 

The training code is provided in `train_eval.py`, the caption and attention map visualization in `vis.py` and testing captions evaluation in `test eval`. The DAN implementation is in `util` folder. 

## Instructions for using ResNet features and modifying captions from other models

Download the COCO 2014 dataset from [here](http://cocodataset.org/#download). In particualr, you'll need the 2014 Training and Validation images. <br/>
Then download Karpathy's Train/Val/Test Split. You may download it from [here](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).<br/>

If you want to do evaluation on COCO, download the COCO API from [here](https://github.com/cocodataset/cocoapi) if your on Linux or from [here](https://github.com/philferriere/cocoapi) if your on Windows. Then download the COCO caption toolkit from [here](https://github.com/tylin/coco-caption) and re-name the folder to `cococaptioncider`. Don't forget to download java as well. Simply dowload it from [here](https://www.java.com/en/download/) if you don't have it.

Use the repository [here](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning) to extract the image features to a `.hd5` file.

Then download the caption data folder from [here](https://drive.google.com/open?id=1QOU8wp_Mr-wT5_fcH3vaAPIw8qO7BPRp) which includes the following: 

`Caption utilities`: a dictionary file with the following: `{"COCO image name": {"caption": "previous caption to modify", "embedding": [512-d DAN embedding of previous caption], "attributes": [the indiced of the 5 extracted attributes], "image_ids": the COCO image id}`

`COCO image names` in the corresponding order 

`Annotations`: The training annotations

`Caption lengths`: The training captions length

`Wordmap`: A dictionary to map the word to their corresponding indices

Place the extracted `.hd5` files in the folder `caption data`.

For more information on the preperation of this dataset, see the folder `data preperation`. The previous captions are embedded first using Google's Universal Sentence Encoder available at TensorFlow Hub [here](https://tfhub.dev/google/universal-sentence-encoder/2) and are loaded to the model for faster processing. You can find how to extract the features from the sentence in the folder `data preperation`. If you would like to implement your own DAN, use the code provided in `util/dan.py` which makes use of the GLoVe word embeddings. You can download the 300-d 6B trained word vectors from [here](https://nlp.stanford.edu/projects/glove/) for use in this function. Moreover, you may ignore the `load embedding` function if you'd like to train the word vectors from scratch using `nn.Embedding`.

In our paper, we make use of variational dropout to effectively regularize our language model, which samples one mask and uses it repeatedly across all timesteps. In that case, all timesteps of the language model receive the same dropout mask. This implementation is included here as well. If you change the dimension of your LSTM hidden state, make sure to adjust accordingly in the `getLockedDropooutMask` function. 

The training code is provided in `train_eval.py`, the caption and attention map visualization in `vis.py` and testing captions evaluation in `test eval`. The DAN implementation is in `util` folder. 

</br>

If you use our code or find our paper useful in your research, please acknowledge the following paper:

```
@misc{Sammani2019ModificationNet,
author = {Sammani, Fawaz and Elsayed, Mahmoud},
title = {Look and Modify: Modification Networks for Image Captioning},
journal = {British Machine Vision Conference (BMVC)},
year = {2019}
}
```

### References
This code is adopted from my [Adaptive Attention](https://github.com/fawazsammani/knowing-when-to-look-adaptive-attention) repository and from [sgrvinod](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning) implementation on "Show, Attend and Tell". 



