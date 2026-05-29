import torch
import torch.nn as nn
import math

class StackedLSTM(nn.Module):
    """
    Stacked LSTM Baseline Model:
    Flattens the M x N implied volatility grid into a 1D vector (size 49),
    passes it through stacked LSTM layers, and decodes back to the forecast horizon.
    """
    def __init__(self, grid_size=(7, 7), hidden_dim=256, num_layers=3, horizon=1):
        super().__init__()
        self.M, self.N = grid_size
        self.input_dim = self.M * self.N
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon

        # Stacked LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )

        # Fully connected decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.horizon * self.input_dim)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, L, 1, M, N)
        Returns:
            Forecast tensor of shape (B, h, M, N)
        """
        B, L, C, M, N = x.shape
        # Flatten the spatial dimensions: shape (B, L, M*N)
        x_flat = x.view(B, L, -1)

        # LSTM forward pass
        # out shape: (B, L, hidden_dim)
        out, _ = self.lstm(x_flat)

        # Take the hidden state of the last time step: shape (B, hidden_dim)
        last_hidden = out[:, -1, :]

        # Decode to forecast horizon: shape (B, h * M * N)
        decoded = self.decoder(last_hidden)

        # Reshape to output structure: shape (B, h, M, N)
        return decoded.view(B, self.horizon, self.M, self.N)


class ConvLSTMCell(nn.Module):
    """
    Standard ConvLSTM Cell executing spatial 2D convolutions in gates.
    """
    def __init__(self, in_channels, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        # A single convolution computes all gate activations simultaneously
        self.conv = nn.Conv2d(
            in_channels=self.in_channels + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state
        # Concatenate along channel axis: shape (B, in_channels + hidden_dim, M, N)
        combined = torch.cat([x, h_cur], dim=1)
        
        # Apply convolution
        combined_conv = self.conv(combined)
        
        # Split channels into individual gate inputs
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Apply activation functions
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        # Update states
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        )


class ConvLSTM(nn.Module):
    """
    Primary Proposed ConvLSTM Model:
    Combines spatial convolutions inside recurrent states to preserve surface correlations.
    """
    def __init__(self, in_channels=1, hidden_dims=[32, 64, 64], kernel_size=3, num_layers=3, horizon=1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.horizon = horizon

        cell_list = []
        for i in range(self.num_layers):
            cur_in = self.in_channels if i == 0 else self.hidden_dims[i - 1]
            cell_list.append(ConvLSTMCell(
                in_channels=cur_in,
                hidden_dim=self.hidden_dims[i],
                kernel_size=self.kernel_size
            ))
        self.cell_list = nn.ModuleList(cell_list)

        # Batch Normalization layers between ConvLSTM layers
        bn_list = []
        for i in range(self.num_layers - 1):
            bn_list.append(nn.BatchNorm3d(self.hidden_dims[i]))
        self.bn_list = nn.ModuleList(bn_list)

        # Output spatial decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(self.hidden_dims[-1], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, self.horizon, kernel_size=1, padding=0)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, L, C, M, N)
        Returns:
            Forecast tensor of shape (B, h, M, N)
        """
        B, L, C, M, N = x.shape
        device = x.device

        # Process layer-by-layer
        current_input = x
        for i in range(self.num_layers):
            cell = self.cell_list[i]
            h, c = cell.init_hidden(B, (M, N), device)
            
            outputs = []
            for t in range(L):
                # Fetch time step t: shape (B, C_in, M, N)
                x_t = current_input[:, t, :, :, :]
                h, c = cell(x_t, (h, c))
                outputs.append(h)
                
            # Stack along sequence dimension: shape (B, L, hidden_dim, M, N)
            outputs = torch.stack(outputs, dim=1)
            
            # Apply Batch Normalization (if not the last layer)
            if i < self.num_layers - 1:
                # BN3d expects shape (B, C, L, M, N)
                outputs = outputs.permute(0, 2, 1, 3, 4)
                outputs = self.bn_list[i](outputs)
                # Reshape back to sequence-first: (B, L, C, M, N)
                outputs = outputs.permute(0, 2, 1, 3, 4)
                
            current_input = outputs

        # Take last time step output of the final layer: shape (B, hidden_dims[-1], M, N)
        last_hidden = current_input[:, -1, :, :, :]

        # Decode to forecast horizon: shape (B, h, M, N)
        forecasts = self.decoder(last_hidden)
        
        return forecasts


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for sequence data.
    """
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (B, L, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderModel(nn.Module):
    """
    Transformer Encoder Model Variant:
    Applies multi-head self-attention mechanisms to capture temporal correlations of surfaces.
    """
    def __init__(self, grid_size=(7, 7), d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1, horizon=1):
        super().__init__()
        self.M, self.N = grid_size
        self.input_dim = self.M * self.N
        self.d_model = d_model
        self.horizon = horizon

        # Flattened spatial projections
        self.embedding = nn.Linear(self.input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(d_model=self.d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.horizon * self.input_dim)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, L, 1, M, N)
        Returns:
            Forecast tensor of shape (B, h, M, N)
        """
        B, L, C, M, N = x.shape
        # Flatten: shape (B, L, M*N)
        x_flat = x.view(B, L, -1)

        # Embed and apply positional encoding
        embedded = self.embedding(x_flat) # shape (B, L, d_model)
        encoded = self.pos_encoder(embedded)

        # Transformer forward pass
        # out shape: (B, L, d_model)
        out = self.transformer_encoder(encoded)

        # Take last time step output: shape (B, d_model)
        last_step = out[:, -1, :]

        # Decode: shape (B, h * M * N)
        decoded = self.decoder(last_step)

        # Reshape to standard forecast format: shape (B, h, M, N)
        return decoded.view(B, self.horizon, self.M, self.N)
