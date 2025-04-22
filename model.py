import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        
        layers.append(nn.Linear(in_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x[x == -1] = 0
        val = self.model(x)
        probs = torch.softmax(val, dim=1)
        return probs


class AuxDrop_MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size,aux_layer_idx,total_features,num_aux_features):
        # aux_drop_idx must be even and less than len(hidden_sizes)
        super(AuxDrop_MLP, self).__init__()
        self.total_features = total_features
        self.num_base_features = total_features - num_aux_features
        self.aux_layer_idx = aux_layer_idx
        self.num_aux_features = num_aux_features

        layers = []
        in_size = input_size

        for i, h in enumerate(hidden_sizes):
            # Skip adding the aux_layer_idx + 1 
            if i == aux_layer_idx + 1:
                in_size = h  # skip this layer
                continue

            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h

        # Final classification layer
        layers.append(nn.Linear(in_size, output_size))
        self.layers = nn.ModuleList(layers)

        # input is (output from aux_layer_idx + aux features), output is hidden_sizes[aux_layer_idx + 1]
        base_output_dim = hidden_sizes[aux_layer_idx]
        self.aux_layer = nn.Linear(base_output_dim + num_aux_features, hidden_sizes[aux_layer_idx + 1])
    def forward(self, x_base, x_aux, aux_mask, dropout_ratio):
        x = x_base
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == self.aux_layer_idx:
                # auxdrop logic
                aux_mask = (x_aux != -1).float()

                # Drop missing features (set them to 0)
                x_aux_masked = x_aux * aux_mask

                # --- Random Dropout for Base Features ---
                expected_aux_drops = (1 - aux_mask).sum(dim=1, keepdim=True)
                base_dropout_ratio = (dropout_ratio * self.total_features - expected_aux_drops) / self.num_base_features
                base_dropout_ratio = base_dropout_ratio.clamp(0.0, 1.0)

                # Sample dropout mask for base features
                base_mask = (torch.rand_like(x) > base_dropout_ratio).float()
                x = x * base_mask  # Apply dropout to base representation

                # Concatenate and pass through aux layer
                x = torch.cat((x, x_aux_masked), dim=1)
                x = self.aux_layer(x)

        probs = F.softmax(x, dim=1)
        return probs