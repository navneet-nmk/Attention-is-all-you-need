import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
use_cuda  = torch.cuda.is_available()


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ScaledDotProductAttention(nn.Module):

    def __init__(self, queries_dim, keys_dim, values_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.q_dim = queries_dim
        self.k_dim = keys_dim
        self.v_dim = values_dim

    def forward(self, queries, keys, values):
        # Inputs are Queries and keys (dimension dk) and values of dimension (dv)

        # Mat mul queries and keys
        queries = queries.view((-1, self.q_dim))
        keys = keys.squeeze()
        queries = torch.transpose(queries, dim1=0, dim0=1)

        print(queries, ' q ')
        print(keys, ' k ')
        x = torch.matmul(queries, keys)
        # Scale the value of x
        x = x/np.sqrt(self.k_dim)
        # Apply softmax
        x = f.softmax(x, dim=-1)
        # Matrix multiply with values
        print(x, ' x ')
        values = values.squeeze()
        values = torch.transpose(values, dim1=0, dim0=1)
        print(values, ' v ')
        output = torch.matmul(x, values)
        print(output)
        return output


class MultiheadAttention(nn.Module):


    def __init__(self, model_dim, queries_dim, keys_dim, values_dim, num_heads, masking=False):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.q_dim = queries_dim
        self.k_dim = keys_dim
        self.v_dim = values_dim
        self.mask = masking
        self.soft_attention_layers = []
        self.linear_layers = []

        # Generate linear representations of the queries, keys and values (h times)
        for h in range(self.num_heads):
            lin_ = []
            q_linear = nn.Linear(self.model_dim, self.q_dim)
            k_linear = nn.Linear(self.model_dim, self.k_dim)
            v_linear = nn.Linear(self.model_dim, self.v_dim)
            if use_cuda:
                q_linear = q_linear.cuda()
                k_linear = k_linear.cuda()
                v_linear = v_linear.cuda()
            lin_.append(q_linear)
            lin_.append(k_linear)
            lin_.append(v_linear)
            self.linear_layers.append(lin_)

        self.multi_head_lin =  nn.Linear(self.num_heads*self.v_dim, self.model_dim)
        self.sc = ScaledDotProductAttention(self.q_dim, self.k_dim, self.v_dim)
        if use_cuda:
            self.sc = self.sc.cuda()

    def forward(self, queries, keys, values):
        for lin_l in self.linear_layers:

            q_linear = lin_l[0](queries)
            k_linear = lin_l[1](keys)
            v_linear = lin_l[2](values)
            output = self.sc(q_linear, k_linear, v_linear)
            self.soft_attention_layers.append(output)

        # Concatenate the different head outputs
        concat = torch.cat(self.soft_attention_layers)

        concat = torch.transpose(concat, dim1=0, dim0=1)
        multihead_output = self.multi_head_lin(concat)

        return multihead_output


# The feed forward network
class FFN(nn.Module):

    def __init__(self, model_dim):
        super(FFN, self).__init__()
        self.model_dim = model_dim

        self.linear_1= nn.Linear(model_dim, model_dim)
        self.relu = nn.ReLU(inplace=True)
        self.linear_final = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        output = self.linear_final(x)
        return output



class Encoder_Layer(nn.Module):

    def __init__(self, model_dim, queries_dim, keys_dim, values_dim, num_heads):
        super(Encoder_Layer, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.queries_dim = queries_dim
        self.keys_dim = keys_dim
        self.value_dim = values_dim
        # Embeddings
        self.multi_headed_attn = MultiheadAttention(model_dim=self.model_dim, queries_dim=self.queries_dim,
                                                    values_dim=self.value_dim, num_heads=self.num_heads
                                                    ,keys_dim=self.keys_dim)

        self.final_linear = FFN(self.model_dim)
        self.layer_norm = LayerNorm(self.model_dim)

        if use_cuda:
            self.multi_headed_attn = self.multi_headed_attn.cuda()
            self.final_linear = self.final_linear.cuda()
            self.layer_norm =  self.layer_norm.cuda()

    def forward(self, encoded_input):

        # x is the input positional embedding-> We first calcualte the multiheaded attention for x
        multi_head_output = self.multi_headed_attn(encoded_input, encoded_input, encoded_input)

        # Add the residual input
        x = encoded_input + multi_head_output
        # Normalize the output
        multi_out = self.layer_norm(x)

        # Feed this input to the feed forward network
        x = self.final_linear(multi_out)
        x = x+multi_out
        output = self.layer_norm(x)

        return output


class Encoder(nn.Module):

    def __init__(self, num_layers, model_dim, queries_dim, keys_dim, values_dim, num_heads, n_vocab):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.queries_dim = queries_dim
        self.keys_dim = keys_dim
        self.values_dim = values_dim
        self.vocab_size = n_vocab
        self.embedding = nn.Embedding(self.vocab_size, self.model_dim)
        # List of Encoder Layers
        self.encoder_layers = nn.ModuleList([Encoder_Layer(self.model_dim, self.queries_dim,
                                                           self.keys_dim, self.values_dim, self.num_heads) for i in range(num_layers)])

        if use_cuda:
            self.encoder_layers = self.encoder_layers.cuda()

    def forward(self, x):
        x = self.embedding(x)
        print(x)
        for i, l in enumerate(self.encoder_layers):
            x = l(x)
            print(x)
        output = x
        return output


class Decoder_Layer(nn.Module):

    def __init__(self, model_dim, queries_dim, values_dim, keys_dim, num_heads):
        super(Decoder_Layer, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.queries_dim = queries_dim
        self.keys_dim = keys_dim
        self.value_dim = values_dim
        self.multi_head_attn_input =   MultiheadAttention(model_dim=self.model_dim, queries_dim=self.queries_dim,
                                                    values_dim=self.value_dim, num_heads=self.num_heads
                                                    ,keys_dim=self.keys_dim, masking=True)

        self.multi_head_attn_encoder = MultiheadAttention(model_dim=self.model_dim, queries_dim=self.queries_dim,
                                                    values_dim=self.value_dim, num_heads=self.num_heads
                                                    ,keys_dim=self.keys_dim)

        self.final_linear_layer = FFN(self.model_dim)
        self.layer_norm = LayerNorm(self.model_dim)

        if use_cuda:
            self.multi_head_attn_input = self.multi_head_attn_input.cuda()
            self.multi_head_attn_encoder = self.multi_head_attn_encoder.cuda()
            self.final_linear_layer = self.final_linear_layer.cuda()
            self.layer_norm =  self.layer_norm.cuda()

    def forward(self, decoder_input, encoder_output):

        # Input x is the positional encoding of the output embedding
        # Apply masked multihead attention on x
        x = self.multi_head_attn_input(decoder_input, decoder_input, decoder_input)
        x =  x + decoder_input
        multi_output = self.layer_norm(x)


        # The multi output goes to the unmasked multihead attention
        x = self.multi_head_attn_encoder(multi_output, encoder_output, encoder_output)
        x = x + multi_output
        multi_unmasked_output = self.layer_norm(x)

        #This output is then fed to the feed forward network
        x = self.final_linear_layer(multi_unmasked_output)
        x = x+ multi_unmasked_output
        x = self.layer_norm(x)

        output = x
        return output


class Decoder(nn.Module):

    def __init__(self, model_dim, queries_dim, values_dim, keys_dim, num_heads, n_vocab, num_layers):
        super(Decoder, self).__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.q_dim = queries_dim
        self.v_dim = values_dim
        self.k_dim = keys_dim
        self.vocab_size = n_vocab

        self.embedding = nn.Embedding(n_vocab, model_dim)
        self.decoder_layers = nn.ModuleList([Decoder_Layer(model_dim=self.model_dim, queries_dim=self.q_dim,
                                                           values_dim=self.v_dim, keys_dim=self.k_dim, num_heads=self.num_heads )
                                             for h in range(self.num_layers)])


        if use_cuda:
            self.decoder_layers = self.decoder_layers.cuda()

    def forward(self, output_sentence, encoder_output):

        x = self.embedding(output_sentence)
        for i, l in enumerate(self.decoder_layers):
          x = l(x, encoder_output)

        output = x
        return output


class Transformer(nn.Module):

    def __init__(self, queries_dim, keys_dim, values_dim, model_dim, num_encoder_layers, num_decoder_layers,
                 n_source_vocab, num_encoder_heads, num_decoder_heads, n_target_vocab, dropout = 0.1):
        super(Transformer, self).__init__()
        self.model_dim = model_dim
        self.queries_dim = queries_dim
        self.values_dim = values_dim
        self.keys_dim = keys_dim
        self.dropout = dropout
        self.num_encoder = num_encoder_layers
        self.num_decoder = num_decoder_layers
        self.num_encoder_heads = num_encoder_heads
        self.num_decoder_heads = num_decoder_heads
        self.n_encoder_vocab = n_source_vocab
        self.n_decoder_vocab = n_target_vocab

        # Encoder
        self.encoder = Encoder(model_dim=self.model_dim, queries_dim=self.queries_dim,
                               values_dim=self.values_dim, keys_dim=self.keys_dim,
                               num_heads=self.num_encoder_heads, num_layers=self.num_encoder,n_vocab=self.n_encoder_vocab)

        # Decoder
        self.decoder = Decoder(model_dim=self.model_dim, queries_dim=self.queries_dim,
                               values_dim= self.values_dim, keys_dim=self.keys_dim,
                               num_heads=self.num_decoder_heads, num_layers=self.num_decoder, n_vocab=self.n_decoder_vocab)

        if use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

        self.target_word = nn.Linear(self.model_dim, self.n_decoder_vocab)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, source, target):

        encoder_output = self.encoder(source)
        if use_cuda:
            encoder_output = encoder_output.cuda()
        decoder_output = self.decoder(target, encoder_output)
        seq_logit =  self.target_word(decoder_output)
        seq_logit = f.softmax(seq_logit)
        return seq_logit.view(-1, seq_logit.size(2))













