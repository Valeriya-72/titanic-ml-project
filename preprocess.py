# Предобработка данных перед обучением моделей
# Пункт 2 чеклиста "Предобработка"

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler


def fill_missing_age(df):
    """    Заполняет пропуски в возрасте (Age) медианой по полу и классу.  """
    # Группируем по полу и классу, вычисляем медиану возраста
    age_medians = df.groupby(['Sex', 'Pclass'])['Age'].median()

    # Заполняем пропуски
    for (sex, pclass), median_age in age_medians.items():
        mask = (df['Sex'] == sex) & (df['Pclass'] == pclass) & (df['Age'].isna())
        df.loc[mask, 'Age'] = median_age

    # Оставшиеся пропуски (если есть) заполняем общей медианой
    df['Age'].fillna(df['Age'].median(), inplace=True)
    return df


def fill_missing_embarked(df):
    """   Заполняет пропуски в порте посадки самым частым значением ('S')   """
    df['Embarked'].fillna('S', inplace=True)
    return df


def fill_missing_fare(df):
    """   Заполняет пропуски в стоимости билета медианой по классу   """
    for pclass in df['Pclass'].unique():
        median_fare = df[df['Pclass'] == pclass]['Fare'].median()
        df.loc[(df['Pclass'] == pclass) & (df['Fare'].isna()), 'Fare'] = median_fare

    # Если остались пропуски (на всякий случай)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    return df


def encode_sex(df):
    """   Превращает 'male'/'female' в 0/1 для использования в модели   """
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    return df


def extract_title(df):
    """
    Извлекает титул из имени (Mr, Mrs, Miss, Master и т.д.)
    Титул — очень сильный признак выживания.
    """
    # Ищем слово с заглавной буквы перед точкой
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

    # Редкие титулы объединяем в одну группу
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                   'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Mme', 'Mlle', 'Ms']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')

    # Превращаем титулы в числа
    title_mapping = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
    df['Title'] = df['Title'].map(title_mapping)
    return df


def create_family_size(df):
    """
    Создаёт признак 'FamilySize' = SibSp + Parch + 1
    (сколько всего родственников на борту, включая самого пассажира)
    """
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    return df


def select_features(df, is_train=True):
    """Выбирает признаки для обучения модели"""
    # РАСШИРЕННЫЙ список признаков (с новыми фичами)
    feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                       'Title', 'FamilySize', 'AgeGroup', 'FareGroup', 'Pclass_Sex', 'Title_Age']

    X = df[feature_columns].copy()

    if is_train and 'Survived' in df.columns:
        y = df['Survived'].copy()
        return X, y
    else:
        return X


def scale_features(X_train, X_test=None, scaler_type='standard', scaler=None):
    """
    Нормализует данные разными способами.

    scaler_type: 'standard', 'minmax', 'robust'
    scaler: если передан готовый scaler, используем его (для тестовых данных)
    """
    if scaler is not None:
        # Используем уже обученный scaler (для тестовых данных)
        X_train_scaled = scaler.transform(X_train)
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            return X_train_scaled, X_test_scaled, scaler
        return X_train_scaled, scaler

    # Обучаем новый scaler
    if scaler_type == 'standard':
        scaler_obj = StandardScaler()
    elif scaler_type == 'minmax':
        scaler_obj = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler_obj = RobustScaler()
    else:
        scaler_obj = StandardScaler()

    X_train_scaled = scaler_obj.fit_transform(X_train)

    if X_test is not None:
        X_test_scaled = scaler_obj.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler_obj
    else:
        return X_train_scaled, scaler_obj


def create_age_group(df):
    """
    Разбивает возраст на категории (Feature Engineering)
    0: ребёнок (0-12), 1: подросток (12-18), 2: молодой взрослый (18-35),
    3: взрослый (35-60), 4: пожилой (60+)
    """
    bins = [0, 12, 18, 35, 60, 100]
    labels = [0, 1, 2, 3, 4]
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    # Заполняем пропуски (если вдруг остались) самой частой группой
    df['AgeGroup'].fillna(2, inplace=True)
    df['AgeGroup'] = df['AgeGroup'].astype(int)
    return df


def create_fare_group(df):
    """Разбивает стоимость билета на 4 группы по квартилям"""
    df['FareGroup'] = pd.qcut(df['Fare'], 4, labels=False, duplicates='drop')
    df['FareGroup'] = df['FareGroup'].astype(int)
    return df


def create_interaction_features(df):
    """
    Создаёт комбинации важных признаков (интеракции)
    Это помогает моделям улавливать нелинейные зависимости
    """
    # Взаимодействие класса билета и пола (женщины в 1 классе выживали часто)
    df['Pclass_Sex'] = df['Pclass'] * df['Sex']

    # Взаимодействие титула и возраста (грубо)
    df['Title_Age'] = df['Title'] * (df['Age'] // 10).astype(int)

    return df


# ========== БЫСТРАЯ ПРОВЕРКА  ==========
if __name__ == "__main__":
    from data_loader import load_train_data

    print("=" * 50)
    print("ПРОВЕРКА ПРЕДОБРАБОТКИ")
    print("=" * 50)

    df = load_train_data()

    # Применяем все функции предобработки (включая новые)
    df = fill_missing_age(df)
    df = fill_missing_embarked(df)
    df = fill_missing_fare(df)
    df = encode_sex(df)
    df = extract_title(df)
    df = create_family_size(df)

    # НОВЫЕ ФИЧИ
    df = create_age_group(df)
    df = create_fare_group(df)
    df = create_interaction_features(df)

    X, y = select_features(df, is_train=True)

    print("\n✅ Предобработка завершена!")
    print(f"Форма X после обработки: {X.shape}")
    print(f"Форма y: {y.shape}")
    print(f"\nПервые 5 строк обработанных признаков:")
    print(X.head())

    print("\n" + "=" * 50)
    print("✅ Всё работает! Теперь с новыми фичами.")
