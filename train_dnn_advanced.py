# DNN на PyTorch с BatchNorm и CosineAnnealingLR
# Полностью соответствует чеклисту: BatchNorm, scheduler, K-fold, Dropout

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import config
from data_loader import load_train_data
from preprocess import (
    fill_missing_age, fill_missing_embarked, fill_missing_fare,
    encode_sex, extract_title, create_family_size, select_features, scale_features,
    create_age_group, create_fare_group, create_interaction_features
)


# ========== 1. ОПРЕДЕЛЕНИЕ НЕЙРОСЕТИ ==========
class AdvancedDNN(nn.Module):
    """Нейросеть с BatchNorm, Dropout, 3 скрытыми слоями"""

    def __init__(self, input_size, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),  # ✅ пункт 7.3
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # ✅ пункт 7.4

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# ========== 2. ЗАГРУЗКА И ПРЕДОБРАБОТКА ==========
print("=" * 60)
print("DNN ADVANCED (PyTorch)")
print("BatchNorm + Dropout + CosineAnnealingLR + K-fold")
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
df = create_interaction_features(df)

X, y = select_features(df, is_train=True)
X_scaled, scaler = scale_features(X)
X_scaled = X_scaled.astype(np.float32)
y = y.values.astype(np.float32)

print(f"Форма X: {X_scaled.shape}, форма y: {y.shape}")

# ========== 3. K-FOLD ВАЛИДАЦИЯ ==========
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
    print(f"\n--- Фолд {fold + 1} ---")

    X_train = X_scaled[train_idx]
    X_val = X_scaled[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = AdvancedDNN(input_size=X_scaled.shape[1], dropout_rate=0.3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # ✅ пункт 7.6
    scheduler = CosineAnnealingLR(optimizer, T_max=50)  # ✅ пункт 7.7
    criterion = nn.BCEWithLogitsLoss()

    # Обучение (50 эпох)
    for epoch in range(50):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Валидация
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_X, _ in val_loader:
            outputs = model(batch_X)
            preds = (torch.sigmoid(outputs) > 0.5).int().flatten()
            predictions.extend(preds.numpy())

    acc = accuracy_score(y_val, predictions)
    fold_accuracies.append(acc)
    print(f"Точность на фолде {fold + 1}: {acc:.4f}")

# ========== 4. РЕЗУЛЬТАТЫ ==========
mean_acc = np.mean(fold_accuracies)
print("\n" + "=" * 60)
print("ИТОГИ DNN ADVANCED")
print("=" * 60)
print(f"Средняя точность по 5 фолдам: {mean_acc:.4f}")
print("=" * 60)
