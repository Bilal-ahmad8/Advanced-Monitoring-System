import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from src.utils.common import read_yaml
from src.constant import PARAMS_FILE_PATH

params = read_yaml(PARAMS_FILE_PATH)
params = params.model_params

torch.manual_seed(42)

def create_sequences(df, seq_len):
    sequences = []
    for i in range(len(df) - seq_len+1):
        seq = df[i:i+seq_len]
        sequences.append(seq)
    sequences = torch.tensor(sequences, dtype = torch.float32)
    return sequences , sequences.clone()

class CustomDataset(Dataset):
    def __init__(self, data, seq):
        self.x, self.y = create_sequences(data, seq)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
class LSTMAutoEncoder(nn.Module):
  def __init__(self,input_dim=6, hidden_dim=params.hidden_dim, num_layers=params.num_layers,biDirect_bool=params.biDirect_bool,
                                dp_ratio_1=params.dp_ratio_1, dp_ratio_2=params.dp_ratio_2, inside_dp_ratio=params.inside_dp_ratio,
                                latent_dim=params.latent_dim):
    super().__init__()
    self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                           dropout=inside_dp_ratio,batch_first=True, bidirectional=biDirect_bool)

    self.dropout = nn.Dropout(dp_ratio_1)
    self.fc = nn.Linear(hidden_dim*2 if biDirect_bool else hidden_dim, latent_dim)

    self.decoder = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, num_layers=num_layers,
                           dropout=inside_dp_ratio, batch_first=True)

    self.dropout2 = nn.Dropout(dp_ratio_2)
    self.fc2 = nn.Linear(hidden_dim, input_dim)

  def forward(self, x):
    _, (h_n, _) = self.encoder(x)

    if self.encoder.bidirectional:
      h_n_combined = torch.cat((h_n[-2], h_n[-1]), dim=1)
    else:
      h_n_combined = h_n[-1]

    h_n_combined = self.dropout(h_n_combined)
    fc1_out = self.fc(h_n_combined)
    latent = fc1_out.unsqueeze(1).repeat(1, x.size(1), 1)

    decoded, _ = self.decoder(latent)
    decoded = self.dropout2(decoded)
    final_decoded = self.fc2(decoded)

    return final_decoded