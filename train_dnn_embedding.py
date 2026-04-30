# Добавление Embedding слоя для категориальных фичей

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import config
from data_loader import load_train_data
from preprocess import (
    fill_missing_age, fill_missing_embarked, fill_missing_fare,
    encode_sex, extract_title, create_family_size,
    create_age_group, create_fare_group
)


# ========== 1. ДАТАСЕТ С EMBEDDING ==========
class TitanicEmbeddingDataset(Dataset):
    def __init__(self, df, is_train=True):
        # Категориальные признаки (для Embedding)
        self.categorical_features = df[['Pclass', 'Sex', 'Title', 'AgeGroup', 'FareGroup', 'Embarked']].copy()

        # Числовые признаки
        self.numerical_features = df[['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']].copy()

        # Размерности категориальных признаков
        self.cat_dims = {
            'Pclass': 4,
            'Sex': 2,
            'Title': 5,
            'AgeGroup': 5,
            'FareGroup': 4,
            'Embarked': 4,
        }

        # ЗАПОЛНЯЕМ ПРОПУСКИ (на всякий случай)
        self.categorical_features = self.categorical_features.fillna(0)
        self.numerical_features = self.numerical_features.fillna(self.numerical_features.median())

        # Нормализуем числовые
        self.scaler = StandardScaler()
        self.numerical_features = self.scaler.fit_transform(self.numerical_features)

        # Категориальные в индексы (убеждаемся, что нет NaN)
        for col in self.categorical_features.columns:
            # Преобразуем в int, предварительно заполнив пропуски
            self.categorical_features[col] = self.categorical_features[col].fillna(0)
            self.categorical_features[col] = self.categorical_features[col].astype('int64')

        self.categorical = torch.tensor(self.categorical_features.values, dtype=torch.long)
        self.numerical = torch.tensor(self.numerical_features, dtype=torch.float32)

        if is_train and 'Survived' in df.columns:
            self.target = torch.tensor(df['Survived'].values, dtype=torch.float32)
            self.has_target = True
        else:
            self.has_target = False

    def __len__(self):
        return len(self.categorical)

    def __getitem__(self, idx):
        if self.has_target:
            return self.categorical[idx], self.numerical[idx], self.target[idx]
        else:
            return self.categorical[idx], self.numerical[idx]


# ========== 2. НЕЙРОСЕТЬ С EMBEDDING ==========
class TitanicDNNWithEmbedding(nn.Module):
    def __init__(self, cat_dims, embedding_dim=8, numerical_dim=5, hidden_sizes=[128, 64], dropout_rate=0.3):
        super().__init__()

        # Embedding слои
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embedding_dim) for dim in cat_dims.values()
        ])

        cat_embedding_dim = len(cat_dims) * embedding_dim  # 48

        self.fc1 = nn.Linear(cat_embedding_dim + numerical_dim, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_sizes[1], 1)
        self.relu = nn.ReLU()

    def forward(self, categorical, numerical):
        embedded = [emb(categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        embedded = torch.cat(embedded, dim=1)
        x = torch.cat([embedded, numerical], dim=1)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


# ========== 3. ПОДГОТОВКА ДАННЫХ ==========
print("=" * 60)
print("DNN С EMBEDDING (PyTorch)")
print("Пункт чеклиста 7.9 — со звёздочкой")
print("=" * 60)

df = load_train_data()
df = fill_missing_age(df)
df = fill_missing_embarked(df)
df = fill_missing_fare(df)
df = encode_sex(df)
df = extract_title(df)
df = create_family_size(df)
df = create_age_group(df)
df = create_fare_group(df)

# Кодируем Embarked в числа
embarked_map = {'C': 0, 'Q': 1, 'S': 2}
df['Embarked'] = df['Embarked'].map(embarked_map)

# Ещё раз убеждаемся, что нет пропусков
df = df.fillna(0)

print(f"✅ Данные загружены, форма: {df.shape}")

# Проверим, что нет NaN в нужных колонках
cat_cols = ['Pclass', 'Sex', 'Title', 'AgeGroup', 'FareGroup', 'Embarked']
for col in cat_cols:
    print(f"  {col}: пропусков {df[col].isna().sum()}")

# ========== 4. K-FOLD ОБУЧЕНИЕ ==========
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['Survived'])):
    print(f"\n--- Фолд {fold + 1} ---")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_dataset = TitanicEmbeddingDataset(train_df, is_train=True)
    val_dataset = TitanicEmbeddingDataset(val_df, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = TitanicDNNWithEmbedding(
        cat_dims=train_dataset.cat_dims,
        embedding_dim=8,
        numerical_dim=5,
        hidden_sizes=[128, 64],
        dropout_rate=0.3
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(50):
        model.train()
        for cat, num, target in train_loader:
            optimizer.zero_grad()
            outputs = model(cat, num)
            loss = criterion(outputs.squeeze(), target)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    predictions = []
    with torch.no_grad():
        for cat, num in val_loader:
            outputs = model(cat, num)
            preds = (torch.sigmoid(outputs) > 0.5).int().squeeze()
            predictions.extend(preds.numpy())

    y_val = val_df['Survived'].values
    acc = accuracy_score(y_val, predictions)
    fold_accuracies.append(acc)
    print(f"Точность на фолде {fold + 1}: {acc:.4f}")

# ========== 5. РЕЗУЛЬТАТЫ ==========
mean_acc = np.mean(fold_accuracies)
print("\n" + "=" * 60)
print("ИТОГИ DNN С EMBEDDING")
print("=" * 60)
print(f"Средняя точность по 5 фолдам: {mean_acc:.4f}")
print("=" * 60)