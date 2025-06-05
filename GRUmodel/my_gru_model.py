import torch 
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# ---------- Модель ----------
class GRUModel(torch.nn.Module):
    def __init__(self, input_dim, embeddings_for_features, hidden_size=256, output_size=1, num_layers=2, dropout=0.05, bidirectional = False):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.bidirectional = bidirectional
        
        # Embedding
        self.embedding = torch.nn.ModuleList([
            torch.nn.Embedding(
                embeddings_for_features[item][0], 
                padding_idx=0, #embeddings_for_features[item][0]-1,
                embedding_dim=embeddings_for_features[item][1]
            ) for item in embeddings_for_features
        ])
        # GRU слой
        self.gru = torch.nn.GRU(
            input_size=sum([embeddings_for_features[item][1] for item in embeddings_for_features]),
            hidden_size=hidden_size,
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=bidirectional
        )

        # Инициализация весов GRU
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

        # Полносвязные слои для выхода
        self.fc = torch.nn.Linear(hidden_size * (2 if self.bidirectional else 1), hidden_size//2)
        self.bn1 = torch.nn.BatchNorm1d(self.hidden_size // 2)
        self.dropout = torch.nn.Dropout(self.dropout_prob)
        self.relu = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(hidden_size//2, hidden_size//4)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_size // 4)
        self.dropout2 = torch.nn.Dropout(self.dropout_prob)

        self.fc_out = torch.nn.Linear(hidden_size//4, output_size)

        # Инициализация весов полносвязных слоев
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)
        torch.nn.init.zeros_(self.fc.bias)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.zeros_(self.fc_out.bias)
        

    def forward(self, x, lengths):
        #Формируем эмбеддинг
        embeddings = [embedding(x[:, :, i]) for i, embedding in enumerate(self.embedding)]
        all_embedding = torch.cat(embeddings, dim=-1)

        packed_input = pack_padded_sequence(
            all_embedding,
            lengths.cpu(),  # Длины должны быть на CPU для pack_padded_sequence
            batch_first=True,
            enforce_sorted=False  # Данные не обязательно отсортированы по длине
        )

        #Формируем скрытый слой
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size).to(x.device)
        
        # Пропускаем через GRU
        out, _ = self.gru(packed_input, h0)  # out: [batch_size, seq_len, hidden_size]
        
        out, _ = pad_packed_sequence(out, batch_first=True)


        # Берем выход с последнего временного шага
        # out = out[:, -1, :]  # [batch_size, hidden_size]
        last_indices = lengths - 1
        batch_indices = torch.arange(batch_size, device=x.device)
        out = out[batch_indices, last_indices, :]
        
        # Пропускаем через полносвязные слои
        out = self.dropout(self.relu(self.bn1(self.fc(out))))
        out = self.dropout2(self.relu(self.bn2(self.fc2(out))))
        out = self.fc_out(out)
        return out.squeeze(-1)