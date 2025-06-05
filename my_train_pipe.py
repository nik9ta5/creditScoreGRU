import os
import torch
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import gc
from IPython.display import clear_output
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from datetime import datetime
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --------------------------------- Загрузка архитектуры модели ---------------------------------
from GRUmodel.my_gru_model import GRUModel

# --------------------------------- DATA ---------------------------------
TRAIN_DATA_DIR = "./train_data" #Директория с train data (установить свой путь)
TEST_DATA_DIR = "./test_data" #Директория с test data (установить свой путь)

TRAIN_TARGET = './train_target.csv' # ID и Метки классов
TEST_TARGET = './test_target.csv' # ID для кого определить веротяности

#Файл с длинной истории и таргетом
ALL_TRAIN_TARGET_INFO = "./info_len_taget.csv"

# --------------------------------- VARIABLES ---------------------------------
DEVICE = torch.device("cuda") if torch.cuda.is_available else torch.device('cpu')
RANDOM_STATE = 42
torch.seed = RANDOM_STATE

# ---------- Признаки для удаления ----------
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




# ---------- Определение размерности эмбеддингов для всех категориальных признаков ----------
def get_embeddings_dim_for_all_fearutes(dataframe, val_dataframe, logger):
    features_df = dataframe.columns
    embedding_dim_for_features = {}
    
    logger.info(f"""\n ----- Embeddings ----- \n""")
    for feature in features_df:
        if feature == 'id' or feature == 'rn':
            continue
        curr_len = max((dataframe[feature].unique().max() + 1), (val_dataframe[feature].unique().max() + 1)) #Будем брать максимум и с запасом + 1
        embedding_dim_for_features[feature] = (curr_len, min(600, round(1.6 * curr_len**0.56))) #Изменим подсчет размерности эмбеддингов min(600, round(1.6 * n_cat**0.56)) было: int(curr_len*0.7)
        
        logger.info(f"TRAIN: {feature} - {dataframe[feature].unique()}\nTEST: {feature} - {val_dataframe[feature].unique()}")

    return embedding_dim_for_features


# ---------- Датасет ----------
class CustomDataset(Dataset):
    def __init__(self, dataframe, target_frame_info, set_for_ids):
        self.dataframe = dataframe
        self.target_frame_info = target_frame_info
        self.set_for_ids = set_for_ids # Сет с ID-шниками, которые входят в данный датасет
        self.index_to_id = {idx : curr_id for idx, curr_id in enumerate(self.set_for_ids)}

    def __len__(self):
        return len(self.set_for_ids) #Количество уникальных ID-шников

    def __getitem__(self, idx):  
        client_id = self.index_to_id[idx] # Получаем ID клиента
        data = self.dataframe[self.dataframe['id'] == client_id].sort_values("rn").values[:, 2:]
        seq_length = data.shape[0]
        target = self.target_frame_info.loc[client_id, 'target']
        return {"data" : data, "lengths" : seq_length, "targets" : target}
    

# Функция для формирование батча из DataLoader'a
def collate_fn2(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    # Извлекаем данные, длины и цели
    data = [torch.tensor(item['data'], dtype=torch.int64) for item in batch]
    lengths = torch.tensor([item['lengths'] for item in batch], dtype=torch.int64)
    targets = torch.tensor([item['targets'] for item in batch], dtype=torch.float32)  # Предполагаем, что targets — float для BCEWithLogitsLoss
    # Паддинг последовательностей до максимальной длины в батче
    padded_data = pad_sequence(data, batch_first=True, padding_value=0)  # (batch_size, max_seq_len, input_dim)
    # Переводим на устройство
    padded_data = padded_data.to(DEVICE)
    lengths = lengths.to(DEVICE)
    targets = targets.to(DEVICE)
    return {"data": padded_data, "lengths": lengths, "targets": targets}


# ---------- Оценка модели (ROC-AUC) ----------
def eval_model(model, val_dataloader):
    model.eval()

    all_preds = []
    all_targets = []
    for batch in tqdm(val_dataloader, desc="validation"):
        if batch is None:
            continue

        # Прогон через модель
        with torch.no_grad():  # Отключаем вычисление градиентов
            pred = model(batch['data'], batch['lengths'])
            pred = pred.squeeze(-1)

        # Сохраняем предсказания и метки
        all_preds.append(pred.cpu().numpy())  
        all_targets.append(batch['targets'].cpu().numpy())
        
    print(f"all_targets len: {len(all_targets)}")
    print(f"all_preds len: {len(all_preds)}")

    roc_auc = roc_auc_score(np.concat(all_targets), np.concat(all_preds))
    print(f"ROC-AUC Score: {roc_auc:.3f}")
    return roc_auc


# ---------- Training ----------
def train_model(model, dataloader, val_dataloader, loss_fn, opt, scheduler = None, epochs=1, validation_step = 50, logger=None, dir_save_model="default", timestamp="default"):
    batch_step = 0
    max_roc = 0

    logger.info(f"""\n---------- Training ----------\n""")
    pathCurrTrain = f'{dir_save_model}/train_model_{timestamp}'
    logger.info(f"""path2dir_save_models: {pathCurrTrain}""")
    os.makedirs(pathCurrTrain, exist_ok=True)
    for epoch in range(epochs):
        for batch in tqdm(dataloader, desc="train dataloader"):
            if batch is None:
                continue
            
            pred = model(batch['data'], batch['lengths'])
            pred = pred.squeeze(-1)

            loss = loss_fn(pred.cpu(), batch['targets'].cpu())
            loss.backward()
            opt.step()
            opt.zero_grad()

            if (batch_step + 1) % validation_step == 0:
                print(f"epoch: {epoch} Trainig loss: {loss}")
                roc = eval_model(model, val_dataloader)
                if scheduler:
                    scheduler.step(roc)
                logger.info(f"epoch: {epoch}. Trainig loss: {loss}. MAX-ROC: {max_roc}. ROC-AUC: {roc}")
                if roc > max_roc:
                    max_roc = roc
                    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                    dir_path = f"{pathCurrTrain}/training_{timestamp}_epoch_{epoch}_{batch_step}/"
                    os.makedirs(dir_path, exist_ok=True)
                    filenameModel = f"{dir_path}model_roc_{roc:.2f}.pt"
                    torch.save(model.state_dict(), filenameModel)
                    logger.info(f"model with ROC: {roc} SAVE PATH: {filenameModel}")
                model.train()
            batch_step += 1

    print('------------- Result -------------')
    logger.info(f"------------- Result Train -------------")
    print(f"trainig loss: {loss}")
    logger.info(f"trainig loss: {loss}")
    roc = eval_model(model, val_dataloader)
    logger.info(f"FINAL ROC: {roc}")

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    dir_path = f"{pathCurrTrain}/training_{timestamp}_final_{batch_step}/"
    os.makedirs(dir_path, exist_ok=True)
    filenameModel = f"{dir_path}model_roc_{roc:.2f}.pt"
    torch.save(model.state_dict(), filenameModel)
    logger.info(f"model with ROC: {roc} SAVE PATH: {filenameModel}")



# ------------------- FOR TRAIN -------------------

if __name__ == "__main__":

    PATH_FOR_SAVE_MODELS = './models' #Директория для сохранения моделей
    PATH_FOR_LOG_FILES = './logs' #Директория для файлов логов
    os.makedirs(PATH_FOR_SAVE_MODELS, exist_ok=True)
    os.makedirs(PATH_FOR_LOG_FILES, exist_ok=True)

    # ------------ Переменные обучения ------------
    STEP = "TRAIN" # TRAIN or VAL or TEST
    VALIDATION_SIZE = 0.1 #Какую часть от общего датасета взять для валидации
    SUB_TRAIN_SIZE = 0.5 #Какую часть от тренировочной взять для обучения (использовал меньше данных, чем было)

    BATCH_SIZE_TRAIN = 256
    BATCH_SIZE_VAL = 256
    EPOCHS = 2
    LR = 1e-3
    L2 = 1e-5

    BIDERECTIONAL = False

    HIDDEN_SIZE = 512
    NUM_LAYERS_GRU = 3
    DROPOUT = 0.05

    VALIDATION_STEPS = 1000

    TIMESTAMP = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    #Создаем файл логгов
    LOG_FILENAME = f"{PATH_FOR_LOG_FILES}/{STEP}_model_{TIMESTAMP}.log"
    logging.basicConfig(
        level=logging.INFO,  # Уровень логирования
        format='%(asctime)s - %(levelname)s - %(message)s',  # Формат сообщений
        handlers=[
            logging.FileHandler(LOG_FILENAME),  # Запись в файл
            # logging.StreamHandler()  # Вывод в консоль (опционально)
        ]
    )
    my_logger = logging.getLogger()
    my_logger.info(f"""
{STEP} - {TIMESTAMP}
RANDOM_STATE = {RANDOM_STATE}
DEVICE = {DEVICE}

VALIDATION_SIZE = {VALIDATION_SIZE}
SUB_TRAIN_SIZE = {SUB_TRAIN_SIZE}

BATCH_SIZE_TRAIN = {BATCH_SIZE_TRAIN}
BATCH_SIZE_VAL = {BATCH_SIZE_VAL}
EPOCHS = {EPOCHS}
LR = {LR}
L2 = {L2}

BIDERECTIONAL = {BIDERECTIONAL}

HIDDEN_SIZE = {HIDDEN_SIZE}
NUM_LAYERS_GRU = {NUM_LAYERS_GRU}

VALIDATION_STEPS = {VALIDATION_STEPS}

FEATURES_FOR_DELETE = {FEATURES_FOR_DELETE}
""")

    all_files_data = os.listdir(TRAIN_DATA_DIR) #Список с именами всех тренировочных файлов
    assert len(all_files_data) == 12, "req 12 data files"
    
    # ---------- Формирование тренировочной и валидационной частей ----------
    target_frame_info = pd.read_csv(ALL_TRAIN_TARGET_INFO) #Фрейм со всеми клиентами, таргетами и длиннами историй
    target_frame_info = target_frame_info.drop(columns=['Unnamed: 0']) #Дропаем лишний признак
    target_frame_info = target_frame_info.sort_values('id')

    #Формируем тренировочную и валидационную выборки
    train_user_ids, val_user_ids = train_test_split(
        target_frame_info,
        test_size=VALIDATION_SIZE,
        stratify=target_frame_info['target'],
        random_state=RANDOM_STATE,
        shuffle=True
    )

    #Сформируем еще подвыборку для обучения (small)
    sub_train_user_ids, sub_val_user_ids = train_test_split(
        train_user_ids,
        train_size=SUB_TRAIN_SIZE,
        stratify=train_user_ids['target'],
        random_state=RANDOM_STATE,
        shuffle=True
    )


    # ---------- Используем подвыборку для обучения ----------
    train_user_ids = sub_train_user_ids

    # ---------- Для веса ошибки ----------
    neg_cl = target_frame_info['target'].value_counts().iloc[0]
    pol_cl = target_frame_info['target'].value_counts().iloc[1]
    pos_weight = torch.tensor([neg_cl / pol_cl], dtype=torch.float32)

    #Считывал разом все (хватало оперативки, но понимаю, что недостаток)
    all_rows_df = []
    for filename in all_files_data: #Идем по всем файлам
        tmp_file = pd.read_parquet(f"{TRAIN_DATA_DIR}/{filename}")
        tmp_file = tmp_file.drop(columns=FEATURES_FOR_DELETE)
        all_rows_df.append(tmp_file)
        break # Удалить когда загружать все

    all_df = pd.concat(all_rows_df, ignore_index=True)

    FEATURE_NUM = all_df.shape[1] - 2 #Потому что еще дропнуть ID и RN
    my_logger.info(f"FEATURE_NUM = {FEATURE_NUM}")

    train_dataset = all_df[all_df['id'].isin(train_user_ids['id'].unique())]
    val_dataset = all_df[all_df['id'].isin(val_user_ids['id'].unique())]
    # all_df = None 

    ids_for_train = set(train_dataset['id'].unique())
    ids_for_val = set(val_dataset['id'].unique())

    my_logger.info(f"""
    shape uniq clients:         {target_frame_info.shape}
    shape train dataset:        {train_dataset.shape}
    shape val dataset:          {val_dataset.shape}
    shape sub train dataset:    {sub_train_user_ids.shape}
    
    cnt ids for train:          {len(ids_for_train)}
    cnt ids for val:            {len(ids_for_val)}                        
    """)

    #Проверка, что ID-шники не пересекаются (чтобы не использовать одного клиента для валидации и обучения)
    assert len(ids_for_train.intersection(ids_for_val)) == 0, f"ID: {ids_for_train.intersection(ids_for_val)} train and val part"

    train_dataset = CustomDataset(train_dataset, target_frame_info, ids_for_train)
    val_dataset = CustomDataset(val_dataset, target_frame_info, ids_for_val)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, collate_fn=collate_fn2)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE_VAL, shuffle=False, collate_fn=collate_fn2)
 
    # ------------------- FOR SUBMISSION -------------------

    # Для предсказания
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

    # --- Получаем размерности для эмбеддинов (для каждого признака) --- 
    #Особенность: тестовые данные содержат признаки, имеющие значения, отличающие от тренировочных
    #Если определять размерности эмбеддингов только по тренировочным данным - на тесте будут возникать ошибки
    embedding_dim_for_features = get_embeddings_dim_for_all_fearutes(all_df, test_rows_df, logger=my_logger)
    
    # Сохранить размерности эмбеддингов в первый раз, чтобы дальше не вычислять каждый раз
    # with open(f"./embeddings_dims.json", "w", encoding='UTF-8') as file:
    #     embedding_dim_for_features = {
    #         item:(int(embedding_dim_for_features[item][0]), embedding_dim_for_features[item][1]) for item in embedding_dim_for_features}
    #     json.dump(embedding_dim_for_features, file)


    #Инициализация модели
    GRUmodel = GRUModel(FEATURE_NUM, embedding_dim_for_features, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS_GRU, dropout=DROPOUT, bidirectional=BIDERECTIONAL).to(DEVICE)
    
    # Загрузка весов модели - если продолжить обучение с определенного сохранения
#     path2continue = './models/training_01-06-2025_23-22-35_final_10548/model_roc_0.77.pt'
#     GRUmodel.load_state_dict(torch.load(path2continue))
#     my_logger.info(f'''
# continue: {path2continue}

# model:
# {str(GRUmodel)}
# ''')

    # Функция потерь, оптимизатор
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) # Если выход не сигмоида (Убрал сигмоиду на выходе)
    # pos_weight - увеличивает штраф за ошибки в предсказании меньшего класса. (то есть наших дефолтов - 1)
    opt = torch.optim.Adam(GRUmodel.parameters(), lr=LR)
    # scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.75, patience=2, verbose=True)

    #Запуск обучения
    train_model(GRUmodel, train_dataloader, val_dataloader, loss_fn, opt, epochs=EPOCHS, scheduler=None, validation_step=VALIDATION_STEPS, logger=my_logger, timestamp=TIMESTAMP, dir_save_model=PATH_FOR_SAVE_MODELS)

    print("DONE")