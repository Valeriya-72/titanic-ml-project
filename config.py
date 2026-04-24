# Настройки для всего проекта: пути к данным и гиперпараметры моделей

# ===== ПУТИ К ФАЙЛАМ =====
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
EDA_OUTPUT_DIR = "eda_output"
MODELS_DIR = "models"
SUBMISSIONS_DIR = "submissions"

# ===== НАСТРОЙКИ ОБУЧЕНИЯ =====
RANDOM_SEED = 42
VAL_SIZE = 0.2

# ===== ВАЛИДАЦИЯ =====
N_FOLDS = 5

# ===== ПАРАМЕТРЫ МОДЕЛЕЙ =====
KNN_NEIGHBORS = [3, 5, 7, 9, 11]

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": RANDOM_SEED
}

XGB_PARAMS = {
    "n_estimators": 150,
    "max_depth": 5,
    "learning_rate": 0.05,
    "random_state": RANDOM_SEED
}

# ===== ЛИНЕЙНЫЕ МОДЕЛИ С РЕГУЛЯРИЗАЦИЕЙ =====
LASSO_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0]      # сила регуляризации L1
ELASTICNET_ALPHAS = [0.001, 0.01, 0.1, 1.0]       # общая сила регуляризации
ELASTICNET_L1_RATIOS = [0.2, 0.5, 0.8]            # пропорция L1 / L2