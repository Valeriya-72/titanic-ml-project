# Загружает данные из CSV-файлов

import pandas as pd
import config


def load_train_data():
    """Загружает обучающую выборку train.csv"""
    df = pd.read_csv(config.TRAIN_PATH)
    print(f"Обучающая выборка: {df.shape[0]} строк, {df.shape[1]} столбцов")
    return df


def load_test_data():
    """Загружает тестовую выборку test.csv"""
    df = pd.read_csv(config.TEST_PATH)
    print(f"Тестовая выборка: {df.shape[0]} строк, {df.shape[1]} столбцов")
    return df


# Проверка (запустится только при запуске этого файла)
if __name__ == "__main__":
    print("=" * 40)
    print("ПРОВЕРКА ЗАГРУЗКИ ДАННЫХ")
    print("=" * 40)

    train = load_train_data()
    print("\nПервые 5 строк train.csv:")
    print(train.head())

    test = load_test_data()
    print("\nПервые 5 строк test.csv:")
    print(test.head())

    print("\n✅ Всё работает!")
