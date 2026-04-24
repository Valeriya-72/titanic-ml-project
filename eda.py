# Выводит информацию о данных, считает пропуски, показывает статистики, строит простые графики (распределение возраста, выживаемости по классу и полу)
# Exploration Data Analysis (EDA)
# Анализ данных перед обучением моделей

import matplotlib.pyplot as plt
import seaborn as sns
import os
import config
from data_loader import load_train_data
import numpy as np

# Загружаем данные
df = load_train_data()

# ========== 1. ИНФОРМАЦИЯ О ДАННЫХ ==========
print("=" * 50)
print("1. ИНФОРМАЦИЯ О ДАННЫХ")
print("=" * 50)
print(df.info())

# ========== 1.1 КОНСТАНТНЫЕ ПРИЗНАКИ ==========
print("\n" + "=" * 50)
print("1.1 КОНСТАНТНЫЕ ПРИЗНАКИ")
print("=" * 50)

constant_features = []
for col in df.columns:
    if df[col].nunique() <= 1:
        constant_features.append(col)
        print(f"  {col}: единственное значение = {df[col].iloc[0]}")

if not constant_features:
    print("  Нет константных признаков (все содержат различающиеся значения)")

# ========== 2. ПРОПУСКИ ==========
print("\n" + "=" * 50)
print("2. ПРОПУСКИ В ДАННЫХ")
print("=" * 50)
print(df.isnull().sum()[df.isnull().sum() > 0])

# ========== 3. СТАТИСТИКИ ДЛЯ ЧИСЛОВЫХ ПРИЗНАКОВ ==========
print("\n" + "=" * 50)
print("3. СТАТИСТИКИ ЧИСЛОВЫХ ПРИЗНАКОВ")
print("=" * 50)
print(df.describe())

# ========== 4. РАСПРЕДЕЛЕНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ ==========
print("\n" + "=" * 50)
print("4. РАСПРЕДЕЛЕНИЕ ВЫЖИВШИХ (Survived)")
print("=" * 50)
print(df['Survived'].value_counts(normalize=True))

# ========== 5. ВИЗУАЛИЗАЦИЯ ==========
# Создаём папку для графиков (если её нет)
os.makedirs(config.EDA_OUTPUT_DIR, exist_ok=True)

# 5.1 Гистограмма возраста
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'].dropna(), bins=30, kde=True)
plt.title('Распределение возраста пассажиров')
plt.xlabel('Возраст')
plt.ylabel('Количество')
plt.savefig(f"{config.EDA_OUTPUT_DIR}/age_distribution.png")
plt.close()
print(f"\n График 1 сохранён: {config.EDA_OUTPUT_DIR}/age_distribution.png")

# 5.2 Выживаемость по классу
plt.figure(figsize=(6, 4))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Выживаемость в зависимости от класса билета')
plt.ylabel('Доля выживших')
plt.savefig(f"{config.EDA_OUTPUT_DIR}/survival_by_class.png")
plt.close()
print(f" График 2 сохранён: {config.EDA_OUTPUT_DIR}/survival_by_class.png")

# 5.3 Выживаемость по полу
plt.figure(figsize=(6, 4))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Выживаемость в зависимости от пола')
plt.ylabel('Доля выживших')
plt.savefig(f"{config.EDA_OUTPUT_DIR}/survival_by_sex.png")
plt.close()
print(f" График 3 сохранён: {config.EDA_OUTPUT_DIR}/survival_by_sex.png")

print("\n" + "=" * 50)

# ========== 6. АНАЛИЗ КОРРЕЛЯЦИИ ПРИЗНАКОВ ==========
print("\n" + "=" * 50)
print("6. КОРРЕЛЯЦИЯ ПРИЗНАКОВ")
print("=" * 50)

# Берём только числовые признаки для корреляции
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = df[numeric_cols].corr()

# Выводим корреляцию с целевой переменной Survived
print("Корреляция признаков с Survived:")
survived_corr = correlation_matrix['Survived'].sort_values(ascending=False)
for feature, corr in survived_corr.items():
    print(f"  {feature}: {corr:.4f}")

# Поиск сильно коррелирующих пар признаков (исключая Survived)
print("\nСильно коррелирующие пары признаков (>0.5 или <-0.5):")
high_corr = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr = correlation_matrix.iloc[i, j]
        if abs(corr) > 0.5:
            col1 = correlation_matrix.columns[i]
            col2 = correlation_matrix.columns[j]
            high_corr.append((col1, col2, corr))
            print(f"  {col1} ↔ {col2}: {corr:.4f}")

if not high_corr:
    print("  Нет сильно коррелирующих пар")

# Тепловая карта корреляции (график)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, mask=np.triu(np.ones_like(correlation_matrix, dtype=bool)))
plt.title('Матрица корреляции признаков')
plt.tight_layout()
plt.savefig(f"{config.EDA_OUTPUT_DIR}/correlation_matrix.png")
plt.close()
print(f"\n График корреляции сохранён: {config.EDA_OUTPUT_DIR}/correlation_matrix.png")

# ========== 7. АНАЛИЗ ВЫБРОСОВ ==========
print("\n" + "=" * 50)
print("7. АНАЛИЗ ВЫБРОСОВ")
print("=" * 50)

# Анализ выбросов для Fare
Q1_fare = df['Fare'].quantile(0.25)
Q3_fare = df['Fare'].quantile(0.75)
IQR_fare = Q3_fare - Q1_fare
lower_bound_fare = Q1_fare - 1.5 * IQR_fare
upper_bound_fare = Q3_fare + 1.5 * IQR_fare
outliers_fare = df[(df['Fare'] < lower_bound_fare) | (df['Fare'] > upper_bound_fare)]
print(f"\nFare (стоимость билета):")
print(f"  Нижняя граница: {lower_bound_fare:.2f}")
print(f"  Верхняя граница: {upper_bound_fare:.2f}")
print(f"  Выбросов: {len(outliers_fare)} ({len(outliers_fare)/len(df)*100:.1f}%)")

# Анализ выбросов для Age
Q1_age = df['Age'].quantile(0.25)
Q3_age = df['Age'].quantile(0.75)
IQR_age = Q3_age - Q1_age
lower_bound_age = Q1_age - 1.5 * IQR_age
upper_bound_age = Q3_age + 1.5 * IQR_age
outliers_age = df[(df['Age'] < lower_bound_age) | (df['Age'] > upper_bound_age)]
print(f"\nAge (возраст):")
print(f"  Нижняя граница: {lower_bound_age:.2f}")
print(f"  Верхняя граница: {upper_bound_age:.2f}")
print(f"  Выбросов: {len(outliers_age)} ({len(outliers_age)/len(df)*100:.1f}%)")

# График выбросов (boxplot)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(y=df['Fare'], ax=axes[0])
axes[0].set_title('Boxplot Fare (стоимость билета)')
sns.boxplot(y=df['Age'], ax=axes[1])
axes[1].set_title('Boxplot Age (возраст)')
plt.tight_layout()
plt.savefig(f"{config.EDA_OUTPUT_DIR}/outliers_boxplot.png")
plt.close()
print(f"\n График выбросов сохранён: {config.EDA_OUTPUT_DIR}/outliers_boxplot.png")


print(" EDA ЗАВЕРШЁН! Все графики сохранены.")
print("=" * 50)