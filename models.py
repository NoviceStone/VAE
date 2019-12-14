import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, data_type="binary"):
        super(VAE, self).__init__()
        # Encoder: layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)
        # Decoder: layers
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc41 = nn.Linear(hidden_size, input_size)
        self.fc42 = nn.Linear(hidden_size, input_size)
        # data_type: can be "binary" or "real"
        self.data_type = data_type

    def encode(self, x):
        h1 = torch.tanh(self.fc1(x))
        mean, log_var = self.fc21(h1), self.fc22(h1)
        return mean, log_var

    @staticmethod
    def reparameterize(mean, log_var):
        mu, sigma = mean, torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        return z

    def decode(self, z):
        h3 = torch.tanh(self.fc3(z))
        if self.data_type == "real":
            mean, log_var = torch.sigmoid(self.fc41(h3)), self.fc42(h3)
            return mean, log_var
        else:
            logits = self.fc41(h3)
            probs = torch.sigmoid(logits)
            return probs

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.reparameterize(z_mean, z_logvar)
        return z_mean, z_logvar, self.decode(z)
