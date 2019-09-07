class DAN(nn.Module):

    def __init__(self, embed_size, hidden_size, word_map, emb_file):
        """
        embed_size: the size of each word embedding from the MLP output of the transferred model
        hidden_size: the dimension of the sentence embedding
        word_map: the wordmap file constructed
        emb_file: the .txt file for the glove embedding weights 
        """
        super(DAN, self).__init__()
        with open(emb_file, 'r') as f:
            self.emb_dim = len(f.readline().split(' ')) - 1
        self.emb_file = emb_file
        self.word_map = word_map
        self.embedding_dan = nn.Embedding(len(word_map), self.emb_dim)  # embedding layer
        self.layer1 = nn.Linear(embed_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.load_embeddings()
        
    def forward(word_ids):
        """
        word_ids: the word indices from the MLP output of the transferred model of shape (batch_size, words)
        """
        embeddings = self.embedding_dan(word_ids)      # (batch_size, words, embed_size)
        #words_mean = embeddings.sum(dim = 1) / embeddings.shape[1]
        words_mean = torch.mean(embeddings, dim = 1)   # (batch_size, embed_size)
        out = self.tanh(self.layer1(words_mean))       # (batch_size, hidden_size)
        out = self.tanh(self.layer2(out))              # (batch_size, hidden_size)
        return out 
    
    def load_embeddings(self):

        vocab = set(self.word_map.keys())

        # Create tensor to hold embeddings, initialize
        embeddings = torch.FloatTensor(len(vocab), self.emb_dim)
        bias = np.sqrt(3.0 / embeddings.size(1))
        torch.nn.init.uniform_(embeddings, -bias, bias)

        # Read embedding file
        for line in open(self.emb_file, 'r', encoding="utf8"):
            line = line.split(' ')
            emb_word = line[0]
            embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))
            # Ignore word if not in train_vocab
            if emb_word not in vocab:
                continue
            embeddings[self.word_map[emb_word]] = torch.FloatTensor(embedding)

        self.embedding_dan.weight = nn.Parameter(embeddings)
        for p in self.embedding_dan.parameters():
            p.requires_grad = False

