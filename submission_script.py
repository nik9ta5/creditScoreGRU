import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from datetime import datetime

# --------------------------------- Загрузка архитектуры модели ---------------------------------
from GRUmodel.my_gru_model import GRUModel

TEST_DATA_DIR = "../alpha/test_data" #Директория с test data (установить свой путь)

DEVICE = torch.device("cuda") if torch.cuda.is_available else torch.device('cpu')
RANDOM_STATE = 42
torch.seed = RANDOM_STATE

FEATURES_FOR_DELETE = [
    "enc_col_5",
    "enc_col_20", 
    "enc_col_21",
    "enc_col_22",
    "enc_col_24",
    "enc_col_41",
    "enc_col_43",
    "enc_col_44",
    "enc_col_47",
    "enc_col_48",
    # ---- Еще исключил ----
    'enc_col_7',
    'enc_col_10',
    'enc_col_13',
    'enc_col_16',
    'enc_col_19',
    'enc_col_36',
    'enc_col_39'
]


class CustomDataset_Test(Dataset):
    ''' Класс для формирования датасета для тестирования '''
    def __init__(self, dataframe, set_for_ids):
        self.dataframe = dataframe
        self.set_for_ids = set_for_ids # Сет с ID-шниками, которые входят в данный датасет
        #Для того, чтобы сделать отображение из [0, len(set_for_ids-1)] в (set_for_ids)
        self.index_to_id = {idx : curr_id for idx, curr_id in enumerate(self.set_for_ids)}

    def __len__(self):
        return len(self.set_for_ids) #Количество уникальных ID-шников

    def __getitem__(self, idx):  
        client_id = self.index_to_id[idx] # Получаем ID клиента
        data = self.dataframe[self.dataframe['id'] == client_id].sort_values("rn").values[:, 2:]
        seq_length = data.shape[0]
        return {"data" : data, "lengths" : seq_length, "ids" : client_id}
    

def collate_fn3_test(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    # Извлекаем данные, длины и цели
    data = [torch.tensor(item['data'], dtype=torch.int64) for item in batch]
    lengths = torch.tensor([item['lengths'] for item in batch], dtype=torch.int64)
    ids = torch.tensor([item['ids'] for item in batch], dtype=torch.float32)  # Предполагаем, что targets — float для BCEWithLogitsLoss
    # Паддинг последовательностей до максимальной длины в батче
    padded_data = pad_sequence(data, batch_first=True, padding_value=0)  # (batch_size, max_seq_len, input_dim)
    # Переводим на устройство
    padded_data = padded_data.to(DEVICE)
    lengths = lengths.to(DEVICE)
    ids = ids.to(DEVICE)
    return {"data": padded_data, "lengths": lengths, "ids": ids}


def eval_for_test_model(model, dataloader):
    model.eval()
    all_preds = []
    all_ids = []
    for batch in tqdm(dataloader, desc="test"):
        if batch is None:
            continue
        with torch.no_grad():  # Отключаем вычисление градиентов
            pred = torch.sigmoid(model(batch['data'], batch['lengths'])) #Добавил сигмоиду во время инференса, чтобы интерпретировать как вероятности
            pred = pred.squeeze(-1)
        # Сохраняем предсказания и метки
        all_preds.append(pred.cpu().numpy())  
        all_ids.append(batch['ids'].cpu().numpy()) 
        
    print(f"all_preds len: {len(all_preds)}")
    print(f"all_ids len: {len(all_ids)}")
    return np.concat(all_preds), np.concat(all_ids)







# ------------------- FOR SUBMISSION -------------------

if __name__ == "__main__":

    STEP = "TRAIN" # TRAIN or VAL or TEST
    VALIDATION_SIZE = 0.1 #Какую часть от общего датасета взять для валидации
    SUB_TRAIN_SIZE = 0.5 #Какую часть от тренировочной взять для обучения (использовал меньше данных, чем было)

    BATCH_SIZE_TRAIN = 256
    BATCH_SIZE_VAL = 256
    EPOCHS = 2
    LR = 1e-3
    L2 = 1e-5

    #Должны совпадать с параметрами используемыми при обучении
    BIDERECTIONAL = False

    HIDDEN_SIZE = 512
    NUM_LAYERS_GRU = 3
    DROPOUT = 0.05
    
    #Считываем с файла размерности для эмбеддингов (PS: был сформирован в my_train_pipe.py)
    with open(f"./embeddings_dims.json", "r", encoding='UTF-8') as file:
        json_string = file.read()
        embedding_dim_for_features = json.loads(json_string)
        # print(embedding_dim_for_features)
    

    # Файлы для предсказаний
    test_files_data = os.listdir(TEST_DATA_DIR)
    assert len(test_files_data) == 2, "req 2 test data files"

    # ID-шники, для которых нужно сделать предсказания
    test_target_ids = pd.read_csv("./test_target.csv")

    test_ids = test_target_ids['id'].unique()

    test_rows_df = []
    for filename in test_files_data: #Идем по всем файлам
        tmp_file = pd.read_parquet(f"{TEST_DATA_DIR}/{filename}")
        tmp_file = tmp_file.drop(columns=FEATURES_FOR_DELETE)
        test_rows_df.append(tmp_file)
        break

    #Полный датасет
    test_rows_df = pd.concat(test_rows_df, ignore_index=True)
    FEATURE_NUM = test_rows_df.shape[1] - 2

    # ------------ Загрузить модель для предсказаний ------------
    path2load_model = './models/training_01-06-2025_23-22-35_final_10548/model_roc_0.77.pt'
    
    GRUmodel = GRUModel(FEATURE_NUM, embedding_dim_for_features, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS_GRU, dropout=DROPOUT, bidirectional=BIDERECTIONAL).to(DEVICE)
    GRUmodel.load_state_dict(torch.load(path2load_model))
    

    print('\n\n ---------- SUBMISSION ----------\n\n')

    test_dataset = CustomDataset_Test(test_rows_df, test_ids)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn3_test)

    all_preds, all_ids = eval_for_test_model(GRUmodel, test_dataloader)

    # # assert len(all_preds) == len(test_ids) and len(all_ids) == len(test_ids), "Equals dim"

    print(len(all_preds))
    print(len(all_ids))

    sub_frame = pd.DataFrame({
         "id" : all_ids.astype(int),
         "target" : all_preds
     })

    sub_frame = sub_frame.sort_values('id')
    sub_frame = sub_frame.reset_index(drop=True)
    print(sub_frame.head())
    print(sub_frame.shape)
    curr_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    sub_frame.to_csv(f'submission_{curr_time}.csv', index=False)

    print("DONE")