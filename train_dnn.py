# Deep Neural Network (DNN) через MLPClassifier (sklearn)

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import config
from data_loader import load_train_data
from preprocess import (
    fill_missing_age, fill_missing_embarked, fill_missing_fare,
    encode_sex, extract_title, create_family_size, select_features, scale_features
)

print("=" * 60)
print("DNN (MLPClassifier из sklearn)")
print("=" * 60)

# Загружаем и обрабатываем данные
df = load_train_data()
df = fill_missing_age(df)
df = fill_missing_embarked(df)
df = fill_missing_fare(df)
df = encode_sex(df)
df = extract_title(df)
df = create_family_size(df)

X, y = select_features(df, is_train=True)
X_scaled, scaler = scale_features(X)

print(f"Признаки: {X.columns.tolist()}")
print(f"Форма X: {X_scaled.shape}")

# ========== ЭКСПЕРИМЕНТЫ ==========
# 1. Базовая MLP из двух слоёв
# 2. Добавление больше слоёв
# 3. Dropout (через alpha)
# 4. Разные размеры слоёв
# 5. Разные оптимизаторы
# 6. Разные параметры (LR, batchsize, эпохи)

experiments = [
    # Простая MLP: 2 скрытых слоя (64 и 32)
    {"name": "MLP 2 слоя (64, 32), Adam",
     "hidden_sizes": (64, 32), "activation": "relu", "solver": "adam",
     "alpha": 0.0001, "batch_size": 32, "max_iter": 100, "learning_rate_init": 0.001},

    # MLP: 3 слоя (128, 64, 32)
    {"name": "MLP 3 слоя (128, 64, 32), Adam",
     "hidden_sizes": (128, 64, 32), "activation": "relu", "solver": "adam",
     "alpha": 0.0001, "batch_size": 32, "max_iter": 100, "learning_rate_init": 0.001},

    # MLP с большим dropout (через alpha)
    {"name": "MLP 3 слоя, усиленная регуляризация (alpha=0.01)",
     "hidden_sizes": (128, 64, 32), "activation": "relu", "solver": "adam",
     "alpha": 0.01, "batch_size": 32, "max_iter": 100, "learning_rate_init": 0.001},

    # Другая активация (tanh)
    {"name": "MLP с tanh активацией",
     "hidden_sizes": (128, 64, 32), "activation": "tanh", "solver": "adam",
     "alpha": 0.0001, "batch_size": 32, "max_iter": 100, "learning_rate_init": 0.001},

    # SGD оптимизатор
    {"name": "MLP с SGD оптимизатором",
     "hidden_sizes": (128, 64, 32), "activation": "relu", "solver": "sgd",
     "alpha": 0.0001, "batch_size": 32, "max_iter": 100, "learning_rate_init": 0.01},

    # Меньший learning rate
    {"name": "LR = 0.0001",
     "hidden_sizes": (128, 64, 32), "activation": "relu", "solver": "adam",
     "alpha": 0.0001, "batch_size": 32, "max_iter": 100, "learning_rate_init": 0.0001},

    # Больше эпох (max_iter)
    {"name": "Эпохи = 200",
     "hidden_sizes": (128, 64, 32), "activation": "relu", "solver": "adam",
     "alpha": 0.0001, "batch_size": 32, "max_iter": 200, "learning_rate_init": 0.001},

    # Большой batch size
    {"name": "Batch size = 64",
     "hidden_sizes": (128, 64, 32), "activation": "relu", "solver": "adam",
     "alpha": 0.0001, "batch_size": 64, "max_iter": 100, "learning_rate_init": 0.001},
]

# K-fold валидация
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)

all_results = []

for exp in experiments:
    print("\n" + "=" * 60)
    print(f"ЭКСПЕРИМЕНТ: {exp['name']}")
    print(f"Параметры: hidden_sizes={exp['hidden_sizes']}, activation={exp['activation']}, "
          f"solver={exp['solver']}, alpha={exp['alpha']}, batch_size={exp['batch_size']}, "
          f"max_iter={exp['max_iter']}, lr={exp['learning_rate_init']}")
    print("=" * 60)

    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        X_train = X_scaled[train_idx]
        X_val = X_scaled[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        mlp = MLPClassifier(
            hidden_layer_sizes=exp['hidden_sizes'],
            activation=exp['activation'],
            solver=exp['solver'],
            alpha=exp['alpha'],
            batch_size=exp['batch_size'],
            max_iter=exp['max_iter'],
            learning_rate_init=exp['learning_rate_init'],
            random_state=config.RANDOM_SEED,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False
        )

        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        fold_accuracies.append(acc)
        print(f"  Фолд {fold + 1}: точность = {acc:.4f}")

    mean_acc = np.mean(fold_accuracies)
    print(f"\n✅ Средняя точность: {mean_acc:.4f}")

    all_results.append({
        "name": exp['name'],
        "accuracy": mean_acc
    })

# Итоги
print("\n" + "=" * 60)
print("ИТОГИ ВСЕХ ЭКСПЕРИМЕНТОВ")
print("=" * 60)

all_results.sort(key=lambda x: x['accuracy'], reverse=True)
for i, res in enumerate(all_results):
    print(f"{i + 1}. {res['name']}: {res['accuracy']:.4f}")

best_dnn = all_results[0]
print("\n" + "=" * 60)
print(f"ЛУЧШАЯ DNN МОДЕЛЬ: {best_dnn['name']}")
print(f"Средняя точность: {best_dnn['accuracy']:.4f}")

# Сравнение с LightGBM (84.17%)
print("\n" + "=" * 60)
print("СРАВНЕНИЕ С ЛУЧШЕЙ ML-МОДЕЛЬЮ")
print("=" * 60)
print(f"LightGBM (лучшая ML): 0.8417")
print(f"DNN (лучшая):         {best_dnn['accuracy']:.4f}")

diff = best_dnn['accuracy'] - 0.8417
if diff > 0:
    print(f" DNN лучше на {diff:.4f}")
else:
    print(f" DNN хуже на {abs(diff):.4f}")