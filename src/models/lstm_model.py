import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


class TimeSeriesDataset(Dataset):
    """Simple sliding window dataset for LSTM."""

    def __init__(self, data, seq_len=12):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return x, y


class SalesLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)
        # take last time step
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out.squeeze(-1)


class LSTMForecaster:
    def __init__(self, seq_len=12, hidden_size=64, num_layers=2,
                 lr=0.001, epochs=100, batch_size=8):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.scaler = MinMaxScaler()
        self.model = None
        self.train_losses = []

    def fit(self, train_series):
        """Fit LSTM on training series."""
        if len(train_series) <= self.seq_len:
            raise ValueError(
                f"Training series length ({len(train_series)}) must be "
                f"greater than seq_len ({self.seq_len})"
            )
        values = train_series.values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(values).flatten()

        dataset = TimeSeriesDataset(scaled, seq_len=self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = SalesLSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            self.train_losses.append(avg_loss)

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")

        self._train_scaled = scaled
        self._train_index = train_series.index
        return self

    def predict(self, steps, test_index=None):
        """
        Multi-step forecast. Uses recursive prediction
        (feed predictions back as input).
        """
        self.model.eval()

        # start with the last seq_len values from training
        current_seq = list(self._train_scaled[-self.seq_len:])
        predictions = []

        with torch.no_grad():
            for _ in range(steps):
                x = torch.FloatTensor(current_seq[-self.seq_len:]).unsqueeze(0)
                pred = self.model(x).item()
                predictions.append(pred)
                current_seq.append(pred)

        # inverse transform
        preds_array = np.array(predictions).reshape(-1, 1)
        preds_inv = self.scaler.inverse_transform(preds_array).flatten()

        if test_index is not None:
            return pd.Series(preds_inv, index=test_index)
        return preds_inv

    def plot_loss(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(self.train_losses)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("LSTM Training Loss")
        plt.tight_layout()
        return fig
