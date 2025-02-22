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


model_type: str = "GAN_LSTM"
stock_name: str = "AAPL"
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
    def __init__(self, latent_dim, seq_length, n_features):
        super().__init__()
        self.hidden_dim = 128

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.hidden_dim),
        )

        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, n_features), nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.unsqueeze(1).repeat(1, sequence_length, 1)

        lstm_out, _ = self.lstm(x)

        return self.output_layer(lstm_out)


class Discriminator(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.hidden_dim = 64

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.classifier(last_output)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
seq_length = 60
n_features = 1
lr = 0.0002
epochs = 1

generator = Generator(latent_dim, seq_length, n_features).to(device)
discriminator = Discriminator(n_features).to(device)

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
