import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
import time
import json

"""
Inputs to this file: The annotations to embed (e.g. TRAIN_CAPTIONS_coco.json) and the WORDMAP ('WORDMAP_coco.json).
"""

with open('TRAIN_CAPTIONS_coco.json', 'r') as r:
    train_captions = json.load(r)

with open('WORDMAP_coco.json', 'r') as j:
    word_map = json.load(j)

rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

sentences = []
for caption in train_captions:
    words = [rev_word_map[caption[i]] for i in range(len(caption))]
    words = [word for word in words if word!= '<start>']
    words = [word for word in words if word!= '<end>']
    sentence = ' '.join([word for word in words if word!= '<pad>'])
    sentences.append(sentence)

sentence_embeddings = []
#use placeholder to avoid building the graph several times
i=0
with tf.Graph().as_default():
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    messages = tf.placeholder(dtype=tf.string, shape=[None])
    output = embed(messages)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        start = time.time()
        for sen in sentences:
            i+=1
            message_embeddings = session.run(output, feed_dict={messages:[sen]})
            message_embeddings = np.array(message_embeddings).tolist()
            embed_round = [round(elem, 4) for elem in message_embeddings[0]]
            sentence_embeddings.append(embed_round)
            if i%1000 == 0:
                print("Finished 1000 captions, Time elapsed {}".format(time.time()-start))
        print("Done! Overall Time elapsed is {}".format(time.time()-start))
        print("{} captions have been embedded!".format(len(captions)))

