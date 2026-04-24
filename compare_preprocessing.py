# Сравнение качества модели ДО и ПОСЛЕ предобработки

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import config
from data_loader import load_train_data
from preprocess import (
    fill_missing_age, fill_missing_embarked, fill_missing_fare,
    encode_sex, extract_title, create_family_size, select_features
)

print("=" * 60)
print("СРАВНЕНИЕ МЕТРИКИ ДО И ПОСЛЕ ПРЕДОБРАБОТКИ")
print("=" * 60)

# Загружаем сырые данные
df_raw = load_train_data()
print(f"\nСырые данные: {df_raw.shape[0]} строк, {df_raw.shape[1]} столбцов")

# === 1. МОДЕЛЬ НА СЫРЫХ ДАННЫХ (минимальная предобработка) ===
print("\n" + "=" * 50)
print("1. МОДЕЛЬ НА СЫРЫХ ДАННЫХ")
print("=" * 50)

# Копируем, чтобы не испортить оригинал
df_raw_copy = df_raw.copy()

# Только самая необходимая предобработка (иначе модель не обучить)
# Заполняем пропуски в Age простой медианой (не групповой)
df_raw_copy['Age'].fillna(df_raw_copy['Age'].median(), inplace=True)
# Заполняем пропуски в Embarked самой частой категорией
df_raw_copy['Embarked'].fillna(df_raw_copy['Embarked'].mode()[0], inplace=True)
# Заполняем пропуски в Fare медианой
df_raw_copy['Fare'].fillna(df_raw_copy['Fare'].median(), inplace=True)
# Кодируем Sex
df_raw_copy['Sex'] = df_raw_copy['Sex'].map({'male': 0, 'female': 1})

# Выбираем признаки для сырых данных (без новых фичей Title, FamilySize)
feature_cols_raw = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X_raw = df_raw_copy[feature_cols_raw].copy()
y_raw = df_raw_copy['Survived'].copy()

print(f"Признаки: {feature_cols_raw}")
print(f"Форма X_raw: {X_raw.shape}")

# Stratified K-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
raw_accuracies = []

for train_idx, val_idx in skf.split(X_raw, y_raw):
    X_train_fold = X_raw.iloc[train_idx]
    X_val_fold = X_raw.iloc[val_idx]
    y_train_fold = y_raw.iloc[train_idx]
    y_val_fold = y_raw.iloc[val_idx]

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=config.RANDOM_SEED)
    rf.fit(X_train_fold, y_train_fold)

    y_pred = rf.predict(X_val_fold)
    acc = accuracy_score(y_val_fold, y_pred)
    raw_accuracies.append(acc)

mean_raw_acc = np.mean(raw_accuracies)
print(f"\nСредняя точность на сырых данных: {mean_raw_acc:.4f} (по 5 фолдам)")

# === 2. МОДЕЛЬ НА ОБРАБОТАННЫХ ДАННЫХ (полная предобработка) ===
print("\n" + "=" * 50)
print("2. МОДЕЛЬ НА ОБРАБОТАННЫХ ДАННЫХ")
print("=" * 50)

# Полная предобработка
df_processed = df_raw.copy()
df_processed = fill_missing_age(df_processed)
df_processed = fill_missing_embarked(df_processed)
df_processed = fill_missing_fare(df_processed)
df_processed = encode_sex(df_processed)
df_processed = extract_title(df_processed)
df_processed = create_family_size(df_processed)

X_proc, y_proc = select_features(df_processed, is_train=True)
print(f"Признаки: {X_proc.columns.tolist()}")
print(f"Форма X_proc: {X_proc.shape}")

proc_accuracies = []

for train_idx, val_idx in skf.split(X_proc, y_proc):
    X_train_fold = X_proc.iloc[train_idx]
    X_val_fold = X_proc.iloc[val_idx]
    y_train_fold = y_proc.iloc[train_idx]
    y_val_fold = y_proc.iloc[val_idx]

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=config.RANDOM_SEED)
    rf.fit(X_train_fold, y_train_fold)

    y_pred = rf.predict(X_val_fold)
    acc = accuracy_score(y_val_fold, y_pred)
    proc_accuracies.append(acc)

mean_proc_acc = np.mean(proc_accuracies)
print(f"\nСредняя точность на обработанных данных: {mean_proc_acc:.4f} (по 5 фолдам)")

# === 3. СРАВНЕНИЕ ===
print("\n" + "=" * 50)
print("СРАВНЕНИЕ")
print("=" * 50)
print(f"До предобработки:    {mean_raw_acc:.4f}")
print(f"После предобработки: {mean_proc_acc:.4f}")
print(
    f"Улучшение:           {mean_proc_acc - mean_raw_acc:.4f} (+{(mean_proc_acc - mean_raw_acc) / mean_raw_acc * 100:.1f}%)")

if mean_proc_acc > mean_raw_acc:
    print("\n Предобработка улучшила качество модели!")
else:
    print("\n Предобработка не дала улучшения (возможно, нужно больше признаков)")