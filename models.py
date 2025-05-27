import timm
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, image_feature_size=14):
        super().__init__()
        effnet = timm.create_model('efficientnet_b4', pretrained=True)
        self.effnet = nn.Sequential(*list(effnet.children())[:-2])

        self.feature_dim = effnet.num_features
        self.pool = nn.AdaptiveAvgPool2d((image_feature_size, image_feature_size))
        self.fine_tune(trainable=False)

    def forward(self, x):
        out = self.effnet(x)
        out = self.pool(out)
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self, trainable=True):
        for param in self.effnet.parameters():
            param.requires_grad = False

        last_blocks = list(self.effnet.children())[-3:]
        for block in last_blocks:
            for param in block.parameters():
                param.requires_grad = trainable

class Attention(nn.Module):
    """ Attention Network. """

    def __init__(self, encoder_dim = 1792, decoder_dim = 512, attention_dim = 512):
        super(Attention, self).__init__()
        """
        encoder_dim: feature size of encoded images
        decoder_dim: size of decoder's RNN
        attention_dim: size of the attention network
        """

        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)

        alpha = self.softmax(att)

        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return attention_weighted_encoding, alpha

class DecoderWithAttention(nn.Module):
    def __init__(self, embedding_dim, vocab_size, attention_dim, decoder_dim, encoder_dim=1792, dropout=0.5):
        super(DecoderWithAttention, self).__init__()
        """
        attention_dim: size of attention network
        embedding_dim: embedding size
        decoder_dim: size of decoder's RNN
        vocab_size: size of vocabulary
        encoder_dim: feature size of encoded images
        dropout: dropout
        """
        
        self.vocab_size=vocab_size
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim) 
        self.dropout = nn.Dropout(p=dropout)
        self.decode_step = nn.LSTMCell(embedding_dim + encoder_dim, decoder_dim, bias=True)
        self.initial_cell_state = nn.Linear(encoder_dim, decoder_dim)    
        self.initial_hidden_state = nn.Linear(encoder_dim, decoder_dim)  
        self.beta_gate = nn.Linear(decoder_dim, encoder_dim) 
        self.sigmoid = nn.Sigmoid()
        self.fcl = nn.Linear(decoder_dim, vocab_size)  
        self.initial_weights()  

    def initial_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fcl.bias.data.fill_(0)
        self.fcl.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, trainable=True):
        for param in self.embedding.parameters():
            param.requires_grad = trainable

    def initial_states(self, encoder_forward_out):
        mean_encoder_out = encoder_forward_out.mean(dim=1)
        h = torch.tanh(self.initial_hidden_state(mean_encoder_out))
        c = torch.tanh(self.initial_cell_state(mean_encoder_out))
        return h, c

    def forward(self, encoder_forward_out, encoded_captions, caption_lengths):
        """
        encoder_out:  (batch_size, H, W, encoder_dim)
        captions:     (batch_size, max_length)
        lengths:      (batch_size, 1)
        returns:
            output_scores:    (batch_size, max_dec_len, vocab_size)
            sorted_captions:  (batch_size, max_length)
            decode_lengths:   List[int]
            attn_weights:     (batch_size, max_dec_len, num_pixels)
            sort_indices:     (batch_size,)
        """

        batch_size = encoder_forward_out.size(0)
        vocab_size = self.vocab_size
        #flatten
        encoder_forward_out = encoder_forward_out.view(batch_size, -1, encoder_forward_out.size(3))

        #keeps the indeces before being sorted and also sorts the captions and encoder data
        caption_lengths, sort_indeces = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_forward_out = encoder_forward_out[sort_indeces]
        encoded_captions = encoded_captions[sort_indeces]

        embeddings = self.embedding(encoded_captions)
        h, c = self.initial_states(encoder_forward_out)
        
        #we start with <start> so we need only n-1 steps
        decode_lengths = (caption_lengths - 1).tolist()
        output_scores = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        attn_weights = torch.zeros(batch_size, max(decode_lengths), 196).to(device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, weight = self.attention(encoder_forward_out[:batch_size_t],h[:batch_size_t])
            gate = self.sigmoid(self.beta_gate(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),(h[:batch_size_t], c[:batch_size_t]))
            scores_for_t = self.fcl(self.dropout(h)) 
            output_scores[:batch_size_t, t, :] = scores_for_t
            attn_weights[:batch_size_t, t, :] = weight

        return output_scores, encoded_captions, decode_lengths, attn_weights, sort_indeces

