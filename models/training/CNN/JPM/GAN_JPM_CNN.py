import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


def preprocess_financial_data(df, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(df)

    sequences = []
    for i in range(len(scaled_data) - sequence_length):
        sequences.append(scaled_data[i : i + sequence_length])

    return np.array(sequences), scaler


model_type: str = "GAN_CNN"
stock_name: str = "JPM"
datetime_now: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

production_data_path: Path = Path("data/30_prod")

csv_market_dataset_name: str = f"{stock_name}.csv"
df: pd.DataFrame = pd.read_csv(
    filepath_or_buffer=production_data_path / csv_market_dataset_name,
    sep=",",
)

df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")

assert pd.api.types.is_datetime64_any_dtype(df["Date"])

print(f"Number of days in the dataset: {df.shape[0]}")
print(f"Starting date of the dataset: {df.loc[0, 'Date'].strftime('%d %b %Y')}")
print(f"Ending date of the dataset: {df.loc[len(df) - 1, 'Date'].strftime('%d %b %Y')}")

column_objective: str = "Close"

features_: NDArray[np.float64] = df[column_objective].values

manual_seed: int = 42

torch.manual_seed(manual_seed)
np.random.seed(manual_seed)
torch.use_deterministic_algorithms(True)

sequence_length: int = 60
sequences, scaler = preprocess_financial_data(features_.reshape(-1, 1), sequence_length)

real_data = torch.FloatTensor(sequences)


class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


dataset = TimeSeriesDataset(real_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


class Generator(nn.Module):
    def __init__(self, latent_dim, seq_length=60, n_features=1):
        super(Generator, self).__init__()
        self.seq_length = seq_length
        self.n_features = n_features
        self.init_length = seq_length // 4
        self.init_channels = 128

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.init_channels * self.init_length),
            nn.BatchNorm1d(self.init_channels * self.init_length),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(
                self.init_channels, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(64, n_features, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(z.size(0), self.init_channels, self.init_length)
        out = self.deconv(out)
        out = out.permute(0, 2, 1)
        return out


class Discriminator(nn.Module):
    def __init__(self, seq_length=60, n_features=1):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
        )

        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(256 * 7, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.permute(0, 2, 1)
        features = self.model(x)
        validity = self.fc(features)
        return validity


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
seq_length = 60
n_features = 1
lr = 0.0002
epochs = 100

generator = Generator(latent_dim, seq_length, n_features).to(device)
discriminator = Discriminator(seq_length, n_features).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()

G_losses = []
D_losses = []

for epoch in range(epochs):
    for i, real_samples in enumerate(dataloader):
        real_samples = real_samples.to(device)
        batch_size = real_samples.size(0)

        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        optimizer_D.zero_grad()
        discriminator(real_samples)

        real_loss = criterion(discriminator(real_samples), valid)

        z = torch.randn(batch_size, latent_dim, device=device)
        fake_samples = generator(z)
        fake_loss = criterion(discriminator(fake_samples.detach()), fake)

        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        optimizer_G.zero_grad()

        z = torch.randn(batch_size, latent_dim, device=device)
        gen_samples = generator(z)
        g_loss = criterion(discriminator(gen_samples), valid)

        g_loss.backward()
        optimizer_G.step()

        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())

    if epoch % 100 == 0:
        print(
            f"Epoch [{epoch}/{epochs}] Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}"  # noqa E501
        )

torch.save(
    generator.state_dict(), f"GENERATOR_{model_type}_{stock_name}_{datetime_now}.pth"
)
torch.save(
    discriminator.state_dict(),
    f"DISCRIMINATOR_{model_type}_{stock_name}_{datetime_now}.pth",
)

print("Ended")
