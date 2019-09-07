"""
Attribute Loss term to be added to the training function
Note: This will increase the training time. 
"""

def calculate_freq(prd_mtx, sentence_attr):
    """
    prd_mtx of shape: (batch_size, max_deode_len)
    sentence_attr of shape: (batch_size, 5)
    """
    freq = 0
    for i in range(prd_mtx.shape[0]):
        for j in range(prd_mtx.shape[1]):
            if prd_mtx[i][j] in sentence_attr[i]:
                freq+=1
    return freq / prd_mtx.shape[0]

  
prd_mtx = torch.zeros(batch_size, max(decode_lengths), dtype = torch.long).to(device)  
for t in range(max(decode_lengths)):
    preds = self.fc2(h2)  # (batch_size_t, vocab_size)
    _, word_idx = torch.max(preds, 1)
    prd_mtx[:batch_size_t,t] = word_idx
  
    return prd_mtx

att_loss_alpha = 0.5
def train(train_loader, decoder, criterion, decoder_optimizer, epoch):

    for i, (img, caption, caplen, sentence_embed, sentence_attr) in enumerate(train_loader):

        sentence_attr = sentence_attr.to(device)

        # Forward prop.
        prd_mtx = decoder(sentence_attr, sentence_embed, image_features, caps, caplens)
        freq = calculate_freq(prd_mtx, sentence_attr)
        att_loss = np.exp((-2/3) * freq)

        if att_loss > 0.069:
            loss+= (att_loss_alpha * att_loss)
 