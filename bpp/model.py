import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / d_k**0.5
    
    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(1), -1e9)
        
    attention_weights = torch.nn.functional.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        output = scaled_dot_product_attention(query, key, value, mask)
        
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        output = self.fc(output)
        
        return output

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedforward(d_model, d_ff)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_output = self.mha(x, x, x, mask)
        x = self.ln1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.ln2(x + ffn_output)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
    


class RNNBinPacking(nn.Module):
    def __init__(self, hidden_size, head_size, num_transformer_layers, num_gru_layers, num_fc_neurons, d_ff):
        super(RNNBinPacking, self).__init__()
        
        self.embed = nn.Linear(2, hidden_size)

        self.transformer_encoder = TransformerEncoder(
            d_model=hidden_size, 
            num_heads=head_size, 
            d_ff=d_ff, 
            num_layers=num_transformer_layers
        )

        self.gru = nn.GRU(hidden_size, hidden_size, num_gru_layers, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, num_fc_neurons),
            nn.ReLU(),
            nn.Linear(num_fc_neurons, 1)
        )
        
    def forward(self, x):
        mask = (x[:, :, 2] == -1).to(x.device)

        order = x[:, :, 2].long()
        order = order.unsqueeze(1).expand(-1, order.size(-1), -1)  # Shape: (batch_size, sequence_length, sequence_length)
        order_mask = (order != order.transpose(1, 2)).to(x.device)

        x = self.embed(x[:, :, :2])

        x = self.transformer_encoder(x, order_mask)

        outputs, hn = self.gru(x)

        # Fetching the last relevant output using the mask
        lengths = (~mask).sum(1).long() - 1  # Minus 1 to get to 0-based index
        last_outputs = outputs[torch.arange(x.size(0), device=x.device), lengths]

        x = self.fc(last_outputs)

        return x

#################################################
# Ablation Study
#################################################



#==============
# PART 1
# Replace Attention Mechanism for Ablation
#==============
    
# class RNNBinPacking(nn.Module):
#     def __init__(self, hidden_size, head_size, num_transformer_layers, num_gru_layers, num_fc_neurons, d_ff):
#         super(RNNBinPacking, self).__init__()
#         self.embed = nn.Sequential(
#             nn.Linear(2, hidden_size),  
#             nn.ReLU(),              
#             nn.Linear(hidden_size, hidden_size) 
#         )
#         self.gru = nn.GRU(hidden_size, hidden_size, num_gru_layers, batch_first=True)
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size, num_fc_neurons),
#             nn.ReLU(),
#             nn.Linear(num_fc_neurons, 1)
#         )
#     def forward(self, x):
#         mask = (x[:, :, 2] == -1).to(x.device)
#         order = x[:, :, 2].long()
#         order = order.unsqueeze(1).expand(-1, order.size(-1), -1)
#         x = self.embed(x[:, :, :2])
#         outputs, hn = self.gru(x)
#         lengths = (~mask).sum(1).long() - 1
#         last_outputs = outputs[torch.arange(x.size(0), device=x.device), lengths]
#         x = self.fc(last_outputs)
#         return x


#----------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------


#==============
# PART 2
# Replace Recurrence Mechanism for Ablation
#==============
    
# class RNNBinPacking(nn.Module):
#     def __init__(self, hidden_size, head_size, num_transformer_layers, num_gru_layers, num_fc_neurons, d_ff):
#         super(RNNBinPacking, self).__init__()
#         self.dim = hidden_size
#         self.embed = nn.Linear(2, hidden_size)
#         self.transformer_encoder = TransformerEncoder(
#             d_model=hidden_size, 
#             num_heads=head_size, 
#             d_ff=d_ff, 
#             num_layers=num_transformer_layers
#         )
#         self.additional_mha = MultiHeadAttention(d_model=hidden_size, num_heads=head_size)
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size, num_fc_neurons),
#             nn.ReLU(),
#             nn.Linear(num_fc_neurons, 1)
#         )
#     def positional_encoding_custom_adjusted(self, positions):
#         batch_size, seq_len = positions.size()
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         positions = positions.to(device)
#         positions = positions - positions[:, 0].unsqueeze(1)
#         position_enc = torch.zeros((batch_size, seq_len, self.dim)).to(device)
#         div_term = torch.exp(torch.arange(0, self.dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.dim)).to(device)
#         position_enc[:, :, 0::2] = torch.sin(positions.unsqueeze(2) * div_term)
#         position_enc[:, :, 1::2] = torch.cos(positions.unsqueeze(2) * div_term)
#         return position_enc
#     def forward(self, x):
#         mask = (x[:, :, 2] == -1).to(x.device)
#         raw_order = x[:, :, 2].long()
#         order = x[:, :, 2].long()
#         order = order.unsqueeze(1).expand(-1, order.size(-1), -1)
#         order_mask = (order != order.transpose(1, 2)).to(x.device)
#         x = self.embed(x[:, :, :2])
#         x = self.transformer_encoder(x, order_mask)
#         x = x + self.positional_encoding_custom_adjusted(raw_order)
#         x = self.additional_mha(x, x, x, mask)  
#         mask_expanded = mask.unsqueeze(-1).expand_as(x)
#         x_masked = torch.masked_fill(x, mask_expanded, 0)
#         lengths = (~mask).sum(dim=1, keepdim=True)
#         x_avg = x_masked.sum(dim=1) / lengths
#         x = self.fc(x_avg)
#         return x