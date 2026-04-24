# Сравните скор на лидерборде у 1 и 5 фолдов из чеклиста

# Сравнение 1 фолда и 5 фолдов (пункт чеклиста "Валидация")
# Сохраняет два файла для отправки на Kaggle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import config
from data_loader import load_train_data, load_test_data
from preprocess import (
    fill_missing_age, fill_missing_embarked, fill_missing_fare,
    encode_sex, extract_title, create_family_size, select_features
)
from sklearn.preprocessing import StandardScaler


def scale_features_simple(X_train, X_test=None):
    """Простое масштабирование для сравнения"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    return X_train_scaled, scaler


# ========== 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ==========
print("=" * 60)
print("СРАВНЕНИЕ 1 ФОЛДА И 5 ФОЛДОВ")
print("=" * 60)

# Обучающие данные
df_train = load_train_data()
df_train = fill_missing_age(df_train)
df_train = fill_missing_embarked(df_train)
df_train = fill_missing_fare(df_train)
df_train = encode_sex(df_train)
df_train = extract_title(df_train)
df_train = create_family_size(df_train)
X, y = select_features(df_train, is_train=True)

# Тестовые данные (для предсказаний)
df_test = load_test_data()
passenger_ids = df_test['PassengerId'].copy()
df_test = fill_missing_age(df_test)
df_test = fill_missing_embarked(df_test)
df_test = fill_missing_fare(df_test)
df_test = encode_sex(df_test)
df_test = extract_title(df_test)
df_test = create_family_size(df_test)
X_test = select_features(df_test, is_train=False)

print(f"Признаки: {X.columns.tolist()}")
print(f"Обучающая выборка: {X.shape[0]} строк")
print(f"Тестовая выборка: {X_test.shape[0]} строк")

# ========== 2. ПОДХОД 1: 1 ФОЛД (train_test_split) ==========
print("\n" + "=" * 60)
print("ПОДХОД 1: 1 ФОЛД (train_test_split)")
print("=" * 60)

# Разделяем 1 раз
X_train_1, X_val_1, y_train_1, y_val_1 = train_test_split(
    X, y, test_size=config.VAL_SIZE, random_state=config.RANDOM_SEED, stratify=y
)

# Масштабируем
X_train_scaled_1, X_val_scaled_1, scaler_1 = scale_features_simple(X_train_1, X_val_1)

# Обучаем
rf_1 = RandomForestClassifier(**config.RF_PARAMS)
rf_1.fit(X_train_scaled_1, y_train_1)

# Оценка на валидации
y_pred_val_1 = rf_1.predict(X_val_scaled_1)
acc_1 = accuracy_score(y_val_1, y_pred_val_1)
print(f"Точность на валидации (1 фолд): {acc_1:.4f}")

# Предсказание на тестовых данных
X_test_scaled_1 = scaler_1.transform(X_test)
predictions_1 = rf_1.predict(X_test_scaled_1)

# Сохраняем для отправки на Kaggle
import os

os.makedirs(config.SUBMISSIONS_DIR, exist_ok=True)

submission_1 = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': predictions_1
})
submission_1.to_csv(f"{config.SUBMISSIONS_DIR}/submission_1fold.csv", index=False)
print(f" Файл для 1 фолда сохранён: {config.SUBMISSIONS_DIR}/submission_1fold.csv")

# ========== 3. ПОДХОД 2: 5 ФОЛДОВ (StratifiedKFold) ==========
print("\n" + "=" * 60)
print("ПОДХОД 2: 5 ФОЛДОВ (StratifiedKFold)")
print("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)

# Массив для хранения предсказаний на тесте (усредняем по фолдам)
test_predictions = np.zeros(len(df_test))

fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx]

    # Масштабируем
    X_train_scaled_fold, X_val_scaled_fold, scaler_fold = scale_features_simple(X_train_fold, X_val_fold)

    # Обучаем
    rf_fold = RandomForestClassifier(**config.RF_PARAMS)
    rf_fold.fit(X_train_scaled_fold, y_train_fold)

    # Оценка на валидации
    y_pred_val_fold = rf_fold.predict(X_val_scaled_fold)
    acc_fold = accuracy_score(y_val_fold, y_pred_val_fold)
    fold_accuracies.append(acc_fold)
    print(f"Фолд {fold + 1}: точность на валидации = {acc_fold:.4f}")

    # Предсказание на тесте
    X_test_scaled_fold = scaler_fold.transform(X_test)
    test_predictions += rf_fold.predict(X_test_scaled_fold) / 5  # усредняем по 5 фолдам

mean_acc_5fold = np.mean(fold_accuracies)
print(f"\nСредняя точность по 5 фолдам: {mean_acc_5fold:.4f}")

# Сохраняем для отправки на Kaggle
submission_5fold = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': (test_predictions > 0.5).astype(int)
})
submission_5fold.to_csv(f"{config.SUBMISSIONS_DIR}/submission_5fold.csv", index=False)
print(f" Файл для 5 фолдов сохранён: {config.SUBMISSIONS_DIR}/submission_5fold.csv")

# ========== 4. СРАВНЕНИЕ ==========
print("\n" + "=" * 60)
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 60)
print(f"1 фолд (простое разделение):        {acc_1:.4f}")
print(f"5 фолдов (средняя по кросс-валидации): {mean_acc_5fold:.4f}")
print(f"Разница: {abs(acc_1 - mean_acc_5fold):.4f}")

print("\n" + "=" * 60)
print(" Созданы два файла для отправки на Kaggle:")
print(f"   - {config.SUBMISSIONS_DIR}/submission_1fold.csv")
print(f"   - {config.SUBMISSIONS_DIR}/submission_5fold.csv")
print("\nЗагрузи их на Kaggle и сравни скоры на лидерборде.")
print("=" * 60)

## Вывод по валидации

- **1 фолд (train_test_split):** точность 81.56%
- **5 фолдов (StratifiedKFold):** средняя точность 82.94%   #Дало более высокую и стабильную оценку качества модели

