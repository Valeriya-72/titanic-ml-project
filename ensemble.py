# Ансамбли моделей: Voting (голосование), Averaging (усреднение), Stacking с Ridge (линейная регрессия с L2-регуляризацией)

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import config
from data_loader import load_train_data
from preprocess import (
    fill_missing_age, fill_missing_embarked, fill_missing_fare,
    encode_sex, extract_title, create_family_size, select_features, scale_features,
    create_age_group, create_fare_group, create_interaction_features
)

print("=" * 60)
print("АНСАМБЛИ ПО ЧЕКЛИСТУ")
print("=" * 60)

# ========== 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ ==========
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

# ========== 2. ЛУЧШИЕ МОДЕЛИ ДЛЯ АНСАМБЛЯ ==========
models = [
    ('lgbm', LGBMClassifier(n_estimators=200, max_depth=5, num_leaves=31,
                            random_state=config.RANDOM_SEED, verbose=-1)),
    ('xgb', XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                          random_state=config.RANDOM_SEED, use_label_encoder=False, eval_metric='logloss')),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=config.RANDOM_SEED)),
]

# ========== 3. K-FOLD ВАЛИДАЦИЯ ==========
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)

# ========== 4. VOTING (мягкое голосование) ==========
print("\n" + "=" * 60)
print("1. VOTING (мягкое голосование)")
print("=" * 60)

voting_clf = VotingClassifier(estimators=models, voting='soft')
voting_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
    X_train = X_scaled[train_idx]
    X_val = X_scaled[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    voting_accuracies.append(acc)
    print(f"  Фолд {fold + 1}: точность = {acc:.4f}")

mean_voting_acc = np.mean(voting_accuracies)
print(f"\n✅ Voting: средняя точность = {mean_voting_acc:.4f}")

# ========== 5. AVERAGING (усреднение вероятностей) ==========
print("\n" + "=" * 60)
print("2. AVERAGING (усреднение вероятностей)")
print("=" * 60)

avg_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
    X_train = X_scaled[train_idx]
    X_val = X_scaled[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    probas = []
    for name, model in models:
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        probas.append(proba)

    avg_proba = np.mean(probas, axis=0)
    y_pred = (avg_proba > 0.5).astype(int)
    acc = accuracy_score(y_val, y_pred)
    avg_accuracies.append(acc)
    print(f"  Фолд {fold + 1}: точность = {acc:.4f}")

mean_avg_acc = np.mean(avg_accuracies)
print(f"\n✅ Averaging: средняя точность = {mean_avg_acc:.4f}")

# ========== 6. STACKING (через Ridge = линейная регрессия с L2) ==========
print("\n" + "=" * 60)
print("3. STACKING (метамодель — RidgeClassifier)")
print("=" * 60)

stacking_ridge = StackingClassifier(estimators=models, final_estimator=RidgeClassifier(alpha=1.0))
ridge_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
    X_train = X_scaled[train_idx]
    X_val = X_scaled[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    stacking_ridge.fit(X_train, y_train)
    y_pred = stacking_ridge.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    ridge_accuracies.append(acc)
    print(f"  Фолд {fold + 1}: точность = {acc:.4f}")

mean_ridge_acc = np.mean(ridge_accuracies)
print(f"\n✅ Stacking (Ridge): средняя точность = {mean_ridge_acc:.4f}")

# ========== 7. ИТОГОВОЕ СРАВНЕНИЕ ПО ЧЕКЛИСТУ ==========
print("\n" + "=" * 60)
print("ИТОГОВОЕ СРАВНЕНИЕ АНСАМБЛЕЙ (ПО ЧЕКЛИСТУ)")
print("=" * 60)
print(f"Voting:           {mean_voting_acc:.4f}")
print(f"Averaging:        {mean_avg_acc:.4f}")
print(f"Stacking (Ridge): {mean_ridge_acc:.4f}")

best = max(mean_voting_acc, mean_avg_acc, mean_ridge_acc)
if best == mean_voting_acc:
    best_name = "Voting"
elif best == mean_avg_acc:
    best_name = "Averaging"
else:
    best_name = "Stacking (Ridge)"

print(f"\n✅ Лучший ансамбль: {best_name} с точностью {best:.4f}")
print("=" * 60)
