"""Models for outlier detection"""
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


def reconstruction_loss(x, x_hat):
    return F.mse_loss(input=x_hat, target=x)


def kl_divergence(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)



class ConvBlock(nn.Module):
    def __init__(self, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=kwargs['in_channels'],
                              out_channels=kwargs['out_channels'],
                              kernel_size=kwargs['kernel_size'],
                              stride=kwargs['stride'],
                              padding=kwargs['padding'])
        self.activation_fn = kwargs['activation_fn']
        self.dropout_rate = kwargs['dropout_rate']

        self.use_batch_normalization = kwargs['use_batch_normalization']
        if self.use_batch_normalization:
            self.batch_norm = nn.BatchNorm2d(num_features=kwargs['out_channels'])

    def compute_output_shape(self, input_shape):
        """https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d"""
        assert len(input_shape) == 4
        batch_size, in_channels, in_height, in_width = input_shape

        out_height = ((in_height + (2 * self.conv.padding[0]) -
                       (self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)) - 1) /
                      self.conv.stride[0]) + 1

        out_width = ((in_width + (2 * self.conv.padding[1]) -
                      (self.conv.dilation[1] * (self.conv.kernel_size[1] - 1)) - 1) /
                     self.conv.stride[1]) + 1

        # NOTE: If it’s not divisible, the output size is rounded down.
        return batch_size, self.conv.out_channels, int(out_height), int(out_width)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate)
        if self.use_batch_normalization:
            x = self.batch_norm(x)

        x = self.activation_fn(x)
        return x


class UpConvBlock(nn.Module):
    def __init__(self, **kwargs):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels=kwargs['in_channels'],
                                         out_channels=kwargs['out_channels'],
                                         kernel_size=kwargs['kernel_size'],
                                         stride=kwargs['stride'],
                                         padding=kwargs['padding'])
        self.activation_fn = kwargs['activation_fn']
        self.dropout_rate = kwargs['dropout_rate']

        self.use_batch_normalization = kwargs['use_batch_normalization']
        if self.use_batch_normalization:
            self.batch_norm = nn.BatchNorm2d(num_features=kwargs['out_channels'])

    def compute_output_shape(self, input_shape):
        """https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html"""
        assert len(input_shape) == 4
        batch_size, in_channels, in_height, in_width = input_shape

        out_height = (((in_height - 1) * self.upconv.stride[0]) - (2 * self.upconv.padding[0]) +
                      (self.upconv.dilation[0] * (self.upconv.kernel_size[0] - 1)) +
                      self.upconv.output_padding[0] + 1)

        out_width = (((in_width - 1) * self.upconv.stride[1]) - (2 * self.upconv.padding[1]) +
                     (self.upconv.dilation[1] * (self.upconv.kernel_size[1] - 1)) +
                     self.upconv.output_padding[1] + 1)

        # NOTE: If it’s not divisible, the output size is rounded down.
        return batch_size, self.upconv.out_channels, int(out_height), int(out_width)

    def forward(self, x):
        x = self.upconv(x)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate)
        if self.use_batch_normalization:
            x = self.batch_norm(x)
        x = self.activation_fn(x)
        return x


class ConvolutionalEncoder(nn.Module):
    def __init__(self, num_channels, hidden_dims, **kwargs):
        super(ConvolutionalEncoder, self).__init__()
        self.conv_blocks = nn.ModuleList()

        hidden_dims = [num_channels] + hidden_dims
        for i in range(len(hidden_dims) - 1):
            self.conv_blocks.append(module=ConvBlock(in_channels=hidden_dims[i],
                                                     out_channels=hidden_dims[i+1],
                                                     **kwargs))

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        for conv_block in self.conv_blocks:
            output_shape = conv_block.compute_output_shape(input_shape=output_shape)
        return output_shape

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        return x


class ConvolutionalDecoder(nn.Module):
    def __init__(self, num_channels, hidden_dims, **kwargs):
        super(ConvolutionalDecoder, self).__init__()
        self.upconv_blocks = nn.ModuleList()

        hidden_dims = [num_channels] + hidden_dims
        for i in range(1, len(hidden_dims))[::-1]:
            self.upconv_blocks.append(module=UpConvBlock(in_channels=hidden_dims[i],
                                                         out_channels=hidden_dims[i-1],
                                                         **kwargs))

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        for upconv_block in self.upconv_blocks:
            output_shape = upconv_block.compute_output_shape(input_shape=output_shape)
        return output_shape

    def forward(self, x):
        for upconv_block in self.upconv_blocks:
            x = upconv_block(x)
        return x


class ConvolutionalVAE(nn.Module):
    def __init__(self, batch_size, num_channels, img_size,
                 latent_dim, hidden_dims, gamma=1.0, beta=1.0,
                 kernel_size=4, stride=2, padding=0,
                 encoder_activation_fn=F.leaky_relu, decoder_activation_fn=F.relu,
                 dropout_rate=0.0, use_batch_normalization=True, output_activation_fn=torch.tanh,
                 device=torch.device('cuda')):
        super(ConvolutionalVAE, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.device = device

        self.encoder = ConvolutionalEncoder(num_channels=num_channels,
                                            hidden_dims=hidden_dims,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            activation_fn=encoder_activation_fn,
                                            dropout_rate=dropout_rate,
                                            use_batch_normalization=use_batch_normalization)

        self.base_input_shape = (batch_size, num_channels, img_size, img_size)
        # print('Base Input Shape:', self.base_input_shape)
        self.encoder_output_shape = self.encoder.compute_output_shape(input_shape=self.base_input_shape)
        # print('Encoder Output Shape: ', self.encoder_output_shape)
        self.encoder_out_features = int(np.prod(self.encoder_output_shape[1:]))
        assert self.encoder_out_features >= latent_dim

        self.fc_mu = nn.Linear(in_features=self.encoder_out_features, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=self.encoder_out_features, out_features=latent_dim)
        # print('Mu and LogVar Shape:', (batch_size, latent_dim))

        self.decoder_input_transformation = nn.Linear(in_features=latent_dim, out_features=self.encoder_out_features)
        self.decoder = ConvolutionalDecoder(num_channels=num_channels,
                                            hidden_dims=hidden_dims,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            activation_fn=decoder_activation_fn,
                                            dropout_rate=dropout_rate,
                                            use_batch_normalization=use_batch_normalization)
        # Change the last activation function to to bring values to some bounded range
        # NOTE: sigmoid(.) -> (0, 1), tanh(.) -> (-1, 1)
        self.decoder.upconv_blocks[-1].activation_fn = output_activation_fn

        self.decoder_output_shape = self.decoder.compute_output_shape(input_shape=self.encoder_output_shape)
        # print('Decoder Output Shape:', self.decoder_output_shape)

        if self.decoder_output_shape != self.base_input_shape:
            # print('Using UpSample(.) to resize the final tensor to base input shape!')
            self.upsampler = nn.Upsample(size=(img_size, img_size),
                                         mode='bilinear',
                                         align_corners=True)

    def loss_function(self, x, x_hat, mu, logvar):
        return self.gamma * reconstruction_loss(x=x, x_hat=x_hat) + \
               self.beta * kl_divergence(mu=mu, logvar=logvar)

    def reparameterize(self, mu, logvar):
        eps = torch.randn(size=mu.shape).to(self.device)
        std = torch.sqrt(torch.exp(logvar))
        z = mu + (eps * std)
        return z

    def encode(self, x):                                                # (B, C_in, H_in, W_in)
        features = self.encoder(x)                                      # (B, C_out, H_out, W_out)
        features = features.view(x.shape[0], -1)                        # (B, C_out * H_out * W_out)

        mu, logvar = self.fc_mu(features), self.fc_logvar(features)     # (B, L), (B, L)
        z = self.reparameterize(mu=mu, logvar=logvar)                   # (B, L)

        return z, mu, logvar

    def decode(self, z):                                                # (B, L)
        z = self.decoder_input_transformation(z)                        # (B, C_out * H_out * W_out)
        z = z.view(-1, *self.encoder_output_shape[1:])                  # (B, C_out, H_out, W_out)
        x_hat = self.decoder(z)                                         # (B, ~C_in, ~H_in, ~W_in)

        if self.decoder_output_shape != self.base_input_shape:
            x_hat = self.upsampler(x_hat)                               # (B, ~C_in, ~H_in, ~W_in)

        return x_hat

    def forward(self, x):                                               # (B, C_in, H_in, W_in)
        z, mu, logvar = self.encode(x)                                  # (B, L)
        x_hat = self.decode(z)                                          # (B, ~C_in, ~H_in, ~W_in)

        return x_hat, mu, logvar


class Block(nn.Module):
    def __init__(self, **kwargs):
        super(Block, self).__init__()
        self.linear = nn.Linear(in_features=kwargs['in_features'], out_features=kwargs['out_features'])
        self.activation_fn = kwargs['activation_fn']
        self.dropout_rate = kwargs['dropout_rate']

        self.use_batch_normalization = kwargs['use_batch_normalization']
        if self.use_batch_normalization:
            self.batch_norm = nn.BatchNorm1d(num_features=kwargs['out_features'])

    def forward(self, x):
        x = self.linear(x)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate)
        if self.use_batch_normalization:
            x = self.batch_norm(x)

        x = self.activation_fn(x)
        return x


class Encoder(nn.Module):
    def __init__(self, num_features, hidden_dims, **kwargs):
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList()

        hidden_dims = [num_features] + hidden_dims
        for i in range(len(hidden_dims) - 1):
            self.blocks.append(module=Block(in_features=hidden_dims[i],
                                            out_features=hidden_dims[i+1],
                                            **kwargs))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_features, hidden_dims, **kwargs):
        super(Decoder, self).__init__()
        self.blocks = nn.ModuleList()

        hidden_dims = [num_features] + hidden_dims
        for i in range(1, len(hidden_dims))[::-1]:
            self.blocks.append(module=Block(in_features=hidden_dims[i],
                                            out_features=hidden_dims[i-1],
                                            **kwargs))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class VAE(nn.Module):
    def __init__(self, input_shape, latent_dim, hidden_dims, gamma=1.0, beta=1.0,
                 encoder_activation_fn=F.leaky_relu, decoder_activation_fn=F.relu,
                 dropout_rate=0.0, use_batch_normalization=True, output_activation_fn=torch.tanh,
                 device=torch.device('cuda')):
        super(VAE, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.device = device

        assert len(input_shape) == 2
        batch_size, num_features = input_shape

        self.encoder = Encoder(num_features=num_features,
                               hidden_dims=hidden_dims,
                               activation_fn=encoder_activation_fn,
                               dropout_rate=dropout_rate,
                               use_batch_normalization=use_batch_normalization)

        self.fc_mu = nn.Linear(in_features=hidden_dims[-1], out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=hidden_dims[-1], out_features=latent_dim)

        self.decoder_input_transformation = nn.Linear(in_features=latent_dim,
                                                      out_features=hidden_dims[-1])
        self.decoder = Decoder(num_features=num_features,
                               hidden_dims=hidden_dims,
                               activation_fn=decoder_activation_fn,
                               dropout_rate=dropout_rate,
                               use_batch_normalization=use_batch_normalization)
        # Change the last activation function to to bring values to some bounded range
        # NOTE: sigmoid(.) -> (0, 1), tanh(.) -> (-1, 1)
        self.decoder.blocks[-1].activation_fn = output_activation_fn

    def loss_function(self, x, x_hat, mu, logvar):
        return self.gamma * reconstruction_loss(x=x, x_hat=x_hat) + \
               self.beta * kl_divergence(mu=mu, logvar=logvar)

    def reparameterize(self, mu, logvar):
        eps = torch.randn(size=mu.shape).to(self.device)
        std = torch.sqrt(torch.exp(logvar))
        z = mu + (eps * std)
        return z

    def encode(self, x):  # (B, D)
        features = self.encoder(x)  # (B, H_out)

        mu, logvar = self.fc_mu(features), self.fc_logvar(features)  # (B, L), (B, L)
        z = self.reparameterize(mu=mu, logvar=logvar)  # (B, L)

        return z, mu, logvar

    def decode(self, z):  # (B, L)
        z = self.decoder_input_transformation(z)  # (B, H_out)
        x_hat = self.decoder(z)  # (B, D)

        return x_hat

    def forward(self, x):  # (B, D)
        z, mu, logvar = self.encode(x)  # (B, L)
        x_hat = self.decode(z)  # (B, D)

        return x_hat, mu, logvar


class RecurrentBlock(nn.Module):
    def __init__(self, **kwargs):
        super(RecurrentBlock, self).__init__()
        self.device = kwargs['device']
        self.hidden_size = kwargs['hidden_size']
        self.return_sequences = kwargs['return_sequences']
        self.lstm = nn.LSTM(input_size=kwargs['input_size'],
                            hidden_size=kwargs['hidden_size'],
                            num_layers=1,
                            dropout=kwargs['dropout_rate'],
                            bidirectional=kwargs['bidirectional'])
        self.activation_fn = kwargs['activation_fn']

        self.use_batch_normalization = kwargs['use_batch_normalization']
        if self.use_batch_normalization:
            self.batch_norm = nn.BatchNorm1d(num_features=kwargs['hidden_size'])

    def forward(self, x):
        h0 = Variable(torch.zeros(2 if self.use_bidirectional else 1,
                                  x.shape[0],
                                  self.hidden_size)).to(self.device)   # (L * 2 OR L, B, H)
        c0 = Variable(torch.zeros(2 if self.use_bidirectional else 1,
                                  x.shape[0],
                                  self.hidden_size)).to(self.device)   # (L * 2 OR L, B, H)

        x, _ = self.lstm(x, (h0, c0))  # (B, P, H*), (2 x (B, B, H*))

        if self.use_batch_normalization:
            # https://discuss.pytorch.org/t/how-does-the-batch-normalization-work-for-sequence-data/30839/3
            x = x.permute(0, 2, 1)
            x = self.batch_norm(x)
            x = x.permute(0, 2, 1)

        x = self.activation_fn(x)

        if not self.return_sequences:
            x = x[:, -1, :]

        return x


class RecurrentEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, **kwargs):
        super(RecurrentEncoder, self).__init__()
        self.recurrent_blocks = nn.ModuleList()

        hidden_dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims) - 1):
            self.recurrent_blocks.append(module=RecurrentBlock(input_size=hidden_dims[i],
                                                               hidden_size=hidden_dims[i+1],
                                                               return_sequences=True if i != len(hidden_dims) - 2 else False,
                                                               **kwargs))

    def forward(self, x):
        for recurrent_block in self.recurrent_blocks:
            x = recurrent_block(x)
        return x


class RecurrentDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, **kwargs):
        super(RecurrentDecoder, self).__init__()
        self.recurrent_blocks = nn.ModuleList()

        hidden_dims = [input_dim] + hidden_dims
        for i in range(1, len(hidden_dims))[::-1]:
            self.recurrent_blocks.append(module=RecurrentBlock(input_size=hidden_dims[i],
                                                               hidden_size=hidden_dims[i-1],
                                                               return_sequences=True if i != 1 else False,
                                                               **kwargs))

    def forward(self, x):
        for recurrent_block in self.recurrent_blocks:
            x = recurrent_block(x)
        return x


class RecurrentVAE(nn.Module):
    def __init__(self, sequence_length, latent_dim, embedding_dim, hidden_dims,
                 bidirectional=False, gamma=1.0, beta=1.0,
                 encoder_activation_fn=F.leaky_relu, decoder_activation_fn=F.relu,
                 dropout_rate=0.0, use_batch_normalization=True, output_activation_fn=torch.tanh,
                 device=torch.device('cuda')):
        super(RecurrentVAE, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.device = device

        self.embeddings = nn.Embedding(num_embeddings=sequence_length, embedding_dim=embedding_dim)
        self.encoder = RecurrentEncoder(input_dim=embedding_dim,
                                        hidden_dims=hidden_dims,
                                        bidirectional=bidirectional,
                                        activation_fn=encoder_activation_fn,
                                        dropout_rate=dropout_rate,
                                        use_batch_normalization=use_batch_normalization,
                                        device=device)

        self.fc_mu = nn.Linear(in_features=hidden_dims[-1], out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=hidden_dims[-1], out_features=latent_dim)

        self.decoder_input_transformation = nn.Linear(in_features=latent_dim,
                                                      out_features=hidden_dims[-1])
        self.decoder = RecurrentDecoder(input_dim=embedding_dim,
                                        hidden_dims=hidden_dims,
                                        bidirectional=bidirectional,
                                        activation_fn=decoder_activation_fn,
                                        dropout_rate=dropout_rate,
                                        use_batch_normalization=use_batch_normalization,
                                        device=device)
        # Change the last activation function to to bring values to some bounded range
        # NOTE: sigmoid(.) -> (0, 1), tanh(.) -> (-1, 1)
        self.decoder.recurrent_blocks[-1].activation_fn = output_activation_fn

    def loss_function(self, x, x_hat, mu, logvar):
        return self.gamma * reconstruction_loss(x=x, x_hat=x_hat) + \
               self.beta * kl_divergence(mu=mu, logvar=logvar)

    def reparameterize(self, mu, logvar):
        eps = torch.randn(size=mu.shape).to(self.device)
        std = torch.sqrt(torch.exp(logvar))
        z = mu + (eps * std)
        return z

    def encode(self, x):  # (B, D)
        features = self.encoder(x)  # (B, H_out)

        mu, logvar = self.fc_mu(features), self.fc_logvar(features)  # (B, L), (B, L)
        z = self.reparameterize(mu=mu, logvar=logvar)  # (B, L)

        return z, mu, logvar

    def decode(self, z):  # (B, L)
        z = self.decoder_input_transformation(z)  # (B, H_out)
        x_hat = self.decoder(z)  # (B, D)

        return x_hat

    def forward(self, x):  # (B, D)
        z, mu, logvar = self.encode(x)  # (B, L)
        x_hat = self.decode(z)  # (B, D)

        return x_hat, mu, logvar

