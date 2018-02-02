import torch
import torch.nn as nn
import torch.nn.functional as f
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
        x = torch.bmm(queries, keys)
        # Scale the value of x
        x = x/torch.sqrt(self.k_dim)
        # Apply softmax
        x = f.softmax(x, dim=-1)
        # Matrix multiply with values
        output = torch.bmm(x, values)

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
            q_linear =  nn.Linear(self.model_dim, self.q_dim)
            k_linear =  nn.Linear(self.model_dim, self.k_dim)
            v_linear =  nn.Linear(self.model_dim, self.v_linear)
            lin_.append(q_linear)
            lin_.append(k_linear)
            lin_.append(v_linear)
            self.linear_layers.append(lin_)

        self.multi_head_lin =  nn.Linear(self.num_heads*self.v_dim, self.model_dim)

    def forward(self, queries, keys, values):
        for lin_l in self.linear_layers:
            q_linear = lin_l[0](queries)
            k_linear = lin_l[1](keys)
            v_linear = lin_l[2](values)
            sc = ScaledDotProductAttention(self.q_dim, self.k_dim, self.v_dim)
            output = sc(q_linear, k_linear, v_linear)
            self.soft_attention_layers.append(output)

        # Concatenate the different head outputs
        concat = torch.cat(self.soft_attention_layers)
        multihead_output = self.multi_head_lin(concat)

        return multihead_output


# The feed forward network
class FFN(nn.Module):

    def __init__(self, model_dim):
        super(FFN, self).__init__()
        self.model_dim = model_dim

        self.linear_1=  nn.Linear(model_dim, model_dim)
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

        self.multi_headed_attn = MultiheadAttention(model_dim=self.model_dim, queries_dim=self.queries_dim,
                                                    values_dim=self.value_dim, num_heads=self.num_heads
                                                    ,keys_dim=self.keys_dim)
        self.final_linear = FFN(self.model_dim)
        self.layer_norm = LayerNorm(self.model_dim)

    def forward(self, x):

        # x is the input positional embedding-> We first calcualte the multiheaded attention for x
        multi_head_output = self.multi_headed_attn(x)
        # Add the residual input
        x = x+multi_head_output
        # Normalize the output
        multi_out = self.layer_norm(x)

        # Feed this input to the feed forward network
        x = self.final_linear(multi_out)
        x = x+multi_out
        output = self.layer_norm(x)

        return output


class Encoder(nn.Module):

    def __init__(self, num_layers, model_dim, queries_dim, keys_dim, values_dim, num_heads):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.queries_dim = queries_dim
        self.keys_dim = keys_dim
        self.values_dim = values_dim

        # List of Encoder Layers
        self.encoder_layers = nn.ModuleList([Encoder_Layer(self.model_dim, self.queries_dim,
                                                           self.keys_dim, self.values_dim, self.num_heads) for i in num_layers])


    def forward(self, x):
        for i, l in enumerate(self.encoder_layers):
            x = self.encoder_layers(x)

        output = x
        return x


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

    def forward(self, inp, encoder_input):

        # Input x is the positional encoding of the output embedding
        # Apply masked multihead attention on x
        x = self.multi_head_attn_input(inp)
        x =  x+inp
        multi_output = self.layer_norm(x)


        # The multi output goes to the unmasked multihead attention
        x = self.multi_head_attn_encoder(multi_output, encoder_input)
        x = x + multi_output
        multi_unmasked_output = self.layer_norm(x)

        #This output is then fed to the feed forward network
        x = self.final_linear_layer(multi_unmasked_output)
        x = x+ multi_unmasked_output
        x = self.layer_norm(x)

        output = x
        return x








