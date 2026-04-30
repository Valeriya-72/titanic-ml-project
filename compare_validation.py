# Сравнение 1 и 5 фолдов (Titanic)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import config
from data_loader import load_train_data, load_test_data
from preprocess import (
    fill_missing_age, fill_missing_embarked, fill_missing_fare,
    encode_sex, extract_title, create_family_size
)


def scale_features_simple(X_train, X_test=None):
    """Простое масштабирование"""
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

df_train = load_train_data()
df_train = fill_missing_age(df_train)
df_train = fill_missing_embarked(df_train)
df_train = fill_missing_fare(df_train)
df_train = encode_sex(df_train)
df_train = extract_title(df_train)
df_train = create_family_size(df_train)

# Выбираем признаки (основные, без новых фич)
feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Title', 'FamilySize']
X = df_train[feature_cols].copy()
y = df_train['Survived'].copy()

df_test = load_test_data()
passenger_ids = df_test['PassengerId'].copy()
df_test = fill_missing_age(df_test)
df_test = fill_missing_embarked(df_test)
df_test = fill_missing_fare(df_test)
df_test = encode_sex(df_test)
df_test = extract_title(df_test)
df_test = create_family_size(df_test)
X_test = df_test[feature_cols].copy()

print(f"Признаки: {feature_cols}")
print(f"Обучающая выборка: {X.shape[0]} строк")
print(f"Тестовая выборка: {X_test.shape[0]} строк")

# ========== 2. ПОДХОД 1: 1 ФОЛД ==========
print("\n" + "=" * 60)
print("ПОДХОД 1: 1 ФОЛД (train_test_split)")
print("=" * 60)

X_train_1, X_val_1, y_train_1, y_val_1 = train_test_split(
    X, y, test_size=config.VAL_SIZE, random_state=config.RANDOM_SEED, stratify=y
)

X_train_scaled_1, X_val_scaled_1, scaler_1 = scale_features_simple(X_train_1, X_val_1)

rf_1 = RandomForestClassifier(**config.RF_PARAMS)
rf_1.fit(X_train_scaled_1, y_train_1)

y_pred_val_1 = rf_1.predict(X_val_scaled_1)
acc_1 = accuracy_score(y_val_1, y_pred_val_1)
print(f"Точность на валидации (1 фолд): {acc_1:.4f}")

X_test_scaled_1 = scaler_1.transform(X_test)
predictions_1 = rf_1.predict(X_test_scaled_1)

import os
os.makedirs(config.SUBMISSIONS_DIR, exist_ok=True)

submission_1 = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': predictions_1
})
submission_1.to_csv(f"{config.SUBMISSIONS_DIR}/submission_1fold.csv", index=False)
print(f" Файл для 1 фолда сохранён: {config.SUBMISSIONS_DIR}/submission_1fold.csv")

# ========== 3. ПОДХОД 2: 5 ФОЛДОВ ==========
print("\n" + "=" * 60)
print("ПОДХОД 2: 5 ФОЛДОВ (StratifiedKFold)")
print("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
test_predictions = np.zeros(len(df_test))
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx]

    X_train_scaled_fold, X_val_scaled_fold, scaler_fold = scale_features_simple(X_train_fold, X_val_fold)

    rf_fold = RandomForestClassifier(**config.RF_PARAMS)
    rf_fold.fit(X_train_scaled_fold, y_train_fold)

    y_pred_val_fold = rf_fold.predict(X_val_scaled_fold)
    acc_fold = accuracy_score(y_val_fold, y_pred_val_fold)
    fold_accuracies.append(acc_fold)
    print(f"Фолд {fold+1}: точность = {acc_fold:.4f}")

    X_test_scaled_fold = scaler_fold.transform(X_test)
    test_predictions += rf_fold.predict(X_test_scaled_fold) / 5

mean_acc_5fold = np.mean(fold_accuracies)
print(f"\nСредняя точность по 5 фолдам: {mean_acc_5fold:.4f}")

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

if mean_acc_5fold > acc_1:
    print(" 5-фолдовая валидация дала более высокую и стабильную оценку")
else:
    print(" Оба подхода показали сопоставимые результаты")