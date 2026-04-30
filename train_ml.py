# Обучение моделей машинного обучения с K-fold валидацией

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import accuracy_score
import config
from data_loader import load_train_data
from preprocess import (
    fill_missing_age, fill_missing_embarked, fill_missing_fare,
    encode_sex, extract_title, create_family_size, select_features, scale_features
)
from preprocess import (
    fill_missing_age, fill_missing_embarked, fill_missing_fare,
    encode_sex, extract_title, create_family_size, select_features, scale_features,
    create_age_group, create_fare_group, create_interaction_features  # ← добавить эти три
)

# ========== 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ ==========
print("=" * 60)
print("ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ")
print("=" * 60)

df = load_train_data()

# Применяем все функции предобработки
df = fill_missing_age(df)
df = fill_missing_embarked(df)
df = fill_missing_fare(df)
df = encode_sex(df)
df = extract_title(df)
df = create_family_size(df)

# НОВЫЕ ФИЧИ (Feature Engineering)
df = create_age_group(df)
df = create_fare_group(df)
df = create_interaction_features(df)

# Выбираем признаки
X, y = select_features(df, is_train=True)


print(f"Признаки: {X.columns.tolist()}")
print(f"Форма X: {X.shape}, форма y: {y.shape}")

# ========== 2. СТРАТИФИЦИРОВАННАЯ K-FOLD ВАЛИДАЦИЯ ==========
print("\n" + "=" * 60)
print(f"СТРАТИФИЦИРОВАННАЯ K-FOLD ВАЛИДАЦИЯ (K={config.N_FOLDS})")
print("=" * 60)

# Создаём объект для K-fold (перемешиваем, сохраняем пропорции классов)
skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)

# ========== 3. KNN С РАЗНЫМИ k ==========
print("\n" + "=" * 60)
print("KNN (K NEAREST NEIGHBORS)")
print("=" * 60)

best_k = None
best_mean_acc = 0

for k in config.KNN_NEIGHBORS:
    fold_accuracies = []

    # Проходим по каждому фолду
    for train_idx, val_idx in skf.split(X, y):
        # Разделяем данные
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        # Масштабируем (важно для KNN!)
        X_train_scaled, X_val_scaled, _ = scale_features(X_train_fold, X_val_fold)

        # Обучаем KNN
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train_fold)

        # Предсказываем и считаем точность
        y_pred = knn.predict(X_val_scaled)
        acc = accuracy_score(y_val_fold, y_pred)
        fold_accuracies.append(acc)

    # Средняя точность по всем фолдам
    mean_acc = np.mean(fold_accuracies)
    print(f"k = {k}: средняя точность = {mean_acc:.4f} (по {config.N_FOLDS} фолдам)")

    if mean_acc > best_mean_acc:
        best_mean_acc = mean_acc
        best_k = k

print(f"\n Лучший k = {best_k}, средняя точность = {best_mean_acc:.4f}")

# ========== 4. RANDOM FOREST ==========
print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

rf_accuracies = []

for train_idx, val_idx in skf.split(X, y):
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx]

    # Random Forest не требует масштабирования
    rf = RandomForestClassifier(**config.RF_PARAMS)
    rf.fit(X_train_fold, y_train_fold)

    y_pred = rf.predict(X_val_fold)
    acc = accuracy_score(y_val_fold, y_pred)
    rf_accuracies.append(acc)

mean_rf_acc = np.mean(rf_accuracies)
print(f"Random Forest: средняя точность = {mean_rf_acc:.4f} (по {config.N_FOLDS} фолдам)")

# ========== 5. ЛИНЕЙНАЯ МОДЕЛЬ (RidgeClassifier) ==========
print("\n" + "=" * 60)
print("RIDGE CLASSIFIER (ЛИНЕЙНАЯ МОДЕЛЬ)")
print("=" * 60)

ridge_accuracies = []

for train_idx, val_idx in skf.split(X, y):
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx]

    # Линейным моделям нужно масштабирование
    X_train_scaled, X_val_scaled, _ = scale_features(X_train_fold, X_val_fold)

    ridge = RidgeClassifier(alpha=1.0, random_state=config.RANDOM_SEED)
    ridge.fit(X_train_scaled, y_train_fold)

    y_pred = ridge.predict(X_val_scaled)
    acc = accuracy_score(y_val_fold, y_pred)
    ridge_accuracies.append(acc)

mean_ridge_acc = np.mean(ridge_accuracies)
print(f"RidgeClassifier: средняя точность = {mean_ridge_acc:.4f} (по {config.N_FOLDS} фолдам)")

# ========== 5. XGBOOST (ГРАДИЕНТНЫЙ БУСТИНГ) ==========
print("\n" + "=" * 60)
print("XGBOOST (ГРАДИЕНТНЫЙ БУСТИНГ)")
print("=" * 60)

from xgboost import XGBClassifier

xgb_accuracies = []

for train_idx, val_idx in skf.split(X, y):
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx]

    # XGBoost не требует масштабирования
    xgb = XGBClassifier(**config.XGB_PARAMS, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train_fold, y_train_fold)

    y_pred = xgb.predict(X_val_fold)
    acc = accuracy_score(y_val_fold, y_pred)
    xgb_accuracies.append(acc)

mean_xgb_acc = np.mean(xgb_accuracies)
print(f"XGBoost: средняя точность = {mean_xgb_acc:.4f} (по {config.N_FOLDS} фолдам)")

# ========== 7. LASSO (L1-РЕГУЛЯРИЗАЦИЯ) ==========
print("\n" + "=" * 60)
print("LASSO (L1-РЕГУЛЯРИЗАЦИЯ)")
print("=" * 60)

from sklearn.linear_model import Lasso

# Lasso решает задачу регрессии, а нам нужна классификация.
# Поэтому используем Lasso с порогом 0.5: если предсказание > 0.5 → класс 1
# Или лучше использовать LogisticRegression с penalty='l1'

from sklearn.linear_model import LogisticRegression

# Logistic Regression с L1-регуляризацией (Lasso)
lasso_accuracies = []
alphas = [0.001, 0.01, 0.1, 1.0, 10.0]  # разные значения силы регуляризации

for alpha in alphas:
    fold_accuracies = []
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        # Логистическая регрессия с L1 (Lasso)
        lr_l1 = LogisticRegression(penalty='l1', solver='saga', C=1 / alpha,
                                   max_iter=1000, random_state=config.RANDOM_SEED)
        lr_l1.fit(X_train_fold, y_train_fold)

        y_pred = lr_l1.predict(X_val_fold)
        acc = accuracy_score(y_val_fold, y_pred)
        fold_accuracies.append(acc)

    mean_acc = np.mean(fold_accuracies)
    lasso_accuracies.append(mean_acc)
    print(f"alpha = {alpha} (сила регуляризации): средняя точность = {mean_acc:.4f}")

best_lasso_idx = np.argmax(lasso_accuracies)
best_lasso_alpha = alphas[best_lasso_idx]
best_lasso_acc = lasso_accuracies[best_lasso_idx]
print(f"\n Лучший alpha для Lasso: {best_lasso_alpha}, точность = {best_lasso_acc:.4f}")

# ========== 8. ELASTICNET (L1 + L2 РЕГУЛЯРИЗАЦИЯ) ==========
print("\n" + "=" * 60)
print("ELASTICNET (L1 + L2 РЕГУЛЯРИЗАЦИЯ)")
print("=" * 60)

elasticnet_accuracies = []
alphas = [0.001, 0.01, 0.1, 1.0]
l1_ratios = [0.2, 0.5, 0.8]  # пропорция между L1 и L2

for alpha in alphas:
    for l1_ratio in l1_ratios:
        fold_accuracies = []
        for train_idx, val_idx in skf.split(X, y):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]

            # Логистическая регрессия с ElasticNet
            lr_en = LogisticRegression(penalty='elasticnet', solver='saga',
                                       C=1 / alpha, l1_ratio=l1_ratio,
                                       max_iter=1000, random_state=config.RANDOM_SEED)
            lr_en.fit(X_train_fold, y_train_fold)

            y_pred = lr_en.predict(X_val_fold)
            acc = accuracy_score(y_val_fold, y_pred)
            fold_accuracies.append(acc)

        mean_acc = np.mean(fold_accuracies)
        elasticnet_accuracies.append((alpha, l1_ratio, mean_acc))
        print(f"alpha = {alpha}, l1_ratio = {l1_ratio}: точность = {mean_acc:.4f}")

best_en = max(elasticnet_accuracies, key=lambda x: x[2])
print(f"\n Лучший ElasticNet: alpha = {best_en[0]}, l1_ratio = {best_en[1]}, точность = {best_en[2]:.4f}")

# ========== 9. ЭКСПЕРИМЕНТ С НОРМАЛИЗАЦИЕЙ (StandardScaler vs MinMaxScaler vs RobustScaler) ==========
print("\n" + "=" * 60)
print("ЭКСПЕРИМЕНТ: РАЗНЫЕ ТИПЫ НОРМАЛИЗАЦИИ ДЛЯ RIDGE")
print("=" * 60)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

scaler_types = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler()
}

for scaler_name, scaler in scaler_types.items():
    fold_accuracies = []
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        # Масштабируем
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)

        # Обучаем Ridge
        ridge = RidgeClassifier(alpha=1.0, random_state=config.RANDOM_SEED)
        ridge.fit(X_train_scaled, y_train_fold)

        y_pred = ridge.predict(X_val_scaled)
        acc = accuracy_score(y_val_fold, y_pred)
        fold_accuracies.append(acc)

    mean_acc = np.mean(fold_accuracies)
    print(f"{scaler_name}: средняя точность = {mean_acc:.4f} (по {config.N_FOLDS} фолдам)")

    # ========== 9. CATBOOST ==========
    print("\n" + "=" * 60)
    print("CATBOOST (ГРАДИЕНТНЫЙ БУСТИНГ НА ДЕРЕВЬЯХ)")
    print("=" * 60)

    try:
        from catboost import CatBoostClassifier

        # Пробуем разные гиперпараметры
        catboost_params_list = [
            {"depth": 4, "iterations": 100, "learning_rate": 0.05},
            {"depth": 6, "iterations": 100, "learning_rate": 0.05},
            {"depth": 8, "iterations": 100, "learning_rate": 0.05},
            {"depth": 6, "iterations": 200, "learning_rate": 0.03},
            {"depth": 6, "iterations": 150, "learning_rate": 0.07, "l2_leaf_reg": 3},
        ]

        best_catboost_acc = 0
        best_catboost_params = None

        for params in catboost_params_list:
            fold_accuracies = []
            for train_idx, val_idx in skf.split(X, y):
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]

                cb = CatBoostClassifier(
                    **params,
                    random_seed=config.RANDOM_SEED,
                    verbose=False,
                    cat_features=[]  # все признаки уже числовые
                )
                cb.fit(X_train_fold, y_train_fold)

                y_pred = cb.predict(X_val_fold)
                acc = accuracy_score(y_val_fold, y_pred)
                fold_accuracies.append(acc)

            mean_acc = np.mean(fold_accuracies)
            print(
                f"depth={params['depth']}, iterations={params['iterations']}, lr={params.get('learning_rate', 0.05)}: точность = {mean_acc:.4f}")

            if mean_acc > best_catboost_acc:
                best_catboost_acc = mean_acc
                best_catboost_params = params

        print(
            f"\n✅ Лучший CatBoost: depth={best_catboost_params['depth']}, iterations={best_catboost_params['iterations']}, точность = {best_catboost_acc:.4f}")

    except ImportError:
        print("❌ CatBoost не установлен. Установи командой: pip install catboost")
        best_catboost_acc = 0

    # ========== 10. LIGHTGBM ==========
    print("\n" + "=" * 60)
    print("LIGHTGBM (ГРАДИЕНТНЫЙ БУСТИНГ НА ДЕРЕВЬЯХ)")
    print("=" * 60)

    try:
        from lightgbm import LGBMClassifier

        # Пробуем разные гиперпараметры
        lgbm_params_list = [
            {"num_leaves": 31, "max_depth": 5, "n_estimators": 100, "learning_rate": 0.05},
            {"num_leaves": 63, "max_depth": 7, "n_estimators": 100, "learning_rate": 0.05},
            {"num_leaves": 31, "max_depth": 5, "n_estimators": 200, "learning_rate": 0.03},
            {"num_leaves": 63, "max_depth": 7, "n_estimators": 200, "learning_rate": 0.03},
            {"num_leaves": 127, "max_depth": 10, "n_estimators": 150, "learning_rate": 0.05},
            {"num_leaves": 31, "max_depth": 5, "n_estimators": 150, "learning_rate": 0.07, "subsample": 0.8},
        ]

        best_lgbm_acc = 0
        best_lgbm_params = None

        for params in lgbm_params_list:
            fold_accuracies = []
            for train_idx, val_idx in skf.split(X, y):
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]

                lgbm = LGBMClassifier(
                    **params,
                    random_state=config.RANDOM_SEED,
                    verbose=-1
                )
                lgbm.fit(X_train_fold, y_train_fold)

                y_pred = lgbm.predict(X_val_fold)
                acc = accuracy_score(y_val_fold, y_pred)
                fold_accuracies.append(acc)

            mean_acc = np.mean(fold_accuracies)
            print(
                f"num_leaves={params['num_leaves']}, max_depth={params['max_depth']}, n_estimators={params['n_estimators']}, lr={params.get('learning_rate', 0.05)}: точность = {mean_acc:.4f}")

            if mean_acc > best_lgbm_acc:
                best_lgbm_acc = mean_acc
                best_lgbm_params = params

        print(
            f"\n✅ Лучший LightGBM: num_leaves={best_lgbm_params['num_leaves']}, max_depth={best_lgbm_params['max_depth']}, n_estimators={best_lgbm_params['n_estimators']}, точность = {best_lgbm_acc:.4f}")

    except ImportError:
        print("❌ LightGBM не установлен. Установи командой: pip install lightgbm")
        best_lgbm_acc = 0

# ========== 11. РЕШАЮЩЕЕ ДЕРЕВО (DECISION TREE) ==========
print("\n" + "=" * 60)
print("РЕШАЮЩЕЕ ДЕРЕВО (DECISION TREE)")
print("=" * 60)

from sklearn.tree import DecisionTreeClassifier

# Пробуем разные глубины деревьев и минимальное количество образцов в листе
tree_params_list = [
    {"max_depth": 3, "min_samples_split": 2},
    {"max_depth": 5, "min_samples_split": 2},
    {"max_depth": 7, "min_samples_split": 2},
    {"max_depth": 10, "min_samples_split": 2},
    {"max_depth": None, "min_samples_split": 2},
    {"max_depth": 5, "min_samples_split": 5},
    {"max_depth": 5, "min_samples_split": 10},
]

best_tree_acc = 0
best_tree_params = None

for params in tree_params_list:
    fold_accuracies = []
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        dt = DecisionTreeClassifier(
            **params,
            random_state=config.RANDOM_SEED
        )
        dt.fit(X_train_fold, y_train_fold)

        y_pred = dt.predict(X_val_fold)
        acc = accuracy_score(y_val_fold, y_pred)
        fold_accuracies.append(acc)

    mean_acc = np.mean(fold_accuracies)
    print(
        f"max_depth={params['max_depth']}, min_samples_split={params['min_samples_split']}: точность = {mean_acc:.4f}")

    if mean_acc > best_tree_acc:
        best_tree_acc = mean_acc
        best_tree_params = params

print(
    f"\n✅ Лучшее дерево: max_depth={best_tree_params['max_depth']}, min_samples_split={best_tree_params['min_samples_split']}, точность = {best_tree_acc:.4f}")

# ========== СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ ==========
print("\n" + "=" * 60)
print("СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ")
print("=" * 60)
print(f"KNN (k={best_k}):                       {best_mean_acc:.4f}")
print(f"Random Forest:                         {mean_rf_acc:.4f}")
print(f"Ridge Classifier:                      {mean_ridge_acc:.4f}")
print(f"XGBoost:                               {mean_xgb_acc:.4f}")
print(f"Lasso (L1):                            {best_lasso_acc:.4f}")
print(f"ElasticNet:                            {best_en[2]:.4f}")
print(f"CatBoost:                              {best_catboost_acc:.4f}")
print(f"LightGBM:                              {best_lgbm_acc:.4f}")
print(f"Decision Tree:                         {best_tree_acc:.4f}")

all_models = [
    (best_mean_acc, "KNN"), (mean_rf_acc, "Random Forest"),
    (mean_ridge_acc, "Ridge"), (mean_xgb_acc, "XGBoost"),
    (best_lasso_acc, "Lasso"), (best_en[2], "ElasticNet"),
    (best_catboost_acc, "CatBoost"), (best_lgbm_acc, "LightGBM"),
    (best_tree_acc, "Decision Tree")
]
best_model = max(all_models, key=lambda x: x[0])
print(f"\n✅ Лучшая модель: {best_model[1]} с точностью {best_model[0]:.4f}")
print("=" * 60)