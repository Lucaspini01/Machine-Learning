import torch, torch.nn as nn, torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, D, k):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, k)
        )
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self, D, k):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, D)
        )
    def forward(self, z): return self.net(z)

class Autoencoder(nn.Module):
    def __init__(self, D, k):
        super().__init__()
        self.enc = Encoder(D, k)
        self.dec = Decoder(D, k)
    def forward(self, x):
        z = self.enc(x)
        xr = self.dec(z)
        return xr, z
    

