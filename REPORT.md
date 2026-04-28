# Отчёт по проекту Titanic Survival Prediction

**Студент:** Valeriya-72  
**Задача:** Предсказать выживание пассажиров Титаника (бинарная классификация)

---

## 1. EDA (Exploratory Data Analysis)

✅ **Что сделано:**
- Выведена общая информация о данных (`df.info()`)
- Посчитаны статистики (`df.describe()`)
- Проанализированы пропуски (Age, Cabin, Embarked)
- Построены графики распределений (matplotlib, seaborn):
  - Распределение возраста
  - Выживаемость по классу
  - Выживаемость по полу
  - Тепловая карта корреляции
  - Boxplot выбросов
- Проверены константные признаки (нет)
- Найдены сильно коррелирующие пары (Pclass ↔ Fare: -0.55)

**Файл:** `eda.py`

---

## 2. Предобработка (Preprocessing)

✅ **Что сделано:**
- **Пропуски:**
  - `Age` — заполнены медианой по группе (Sex + Pclass)
  - `Embarked` — заполнены самым частым значением ('S')
  - `Fare` — заполнены медианой по классу
- **Нормализация:** применён `StandardScaler` (для KNN и линейных моделей)
- **Категориальные фичи:**
  - `Sex` — закодирован в 0/1
  - `Title` — извлечён из имени (Mr, Mrs, Miss и др.)
- **Выбросы:** проанализированы (Fare: 13%, Age: 1.2%)
- **Сравнение метрики до/после предобработки:** выполнено в `compare_preprocessing.py`

**Файлы:** `preprocess.py`, `compare_preprocessing.py`

---

## 3. Валидация

✅ **Что сделано:**
- Использован **Stratified K-fold** (k=5)
- Сравнены **1 и 5 фолдов**:
  - 1 фолд (train_test_split): точность 81.56%
  - 5 фолдов: средняя точность 82.94%
- **Вывод:** 5 фолдов даёт более стабильную оценку качества.

**Файлы:** все `train_*.py`, `compare_validation.py`

---

## 4. Модели (Machine Learning)

✅ **Что сделано:**
- Для бинарной классификации использована логистическая регрессия (корректный подход)

| Модель | Параметры | Результат (5 фолдов) |
|--------|-----------|---------------------|
| **LightGBM** | num_leaves=31, max_depth=5, n_estimators=200 | **84.51%** |
| Random Forest | n_estimators=100, max_depth=10 | 83.95% |
| XGBoost | n_estimators=150, max_depth=5 | 83.72% |
| CatBoost | depth=8, iterations=100 | 83.05% |
| Logistic Regression | penalty=None | 80.59% |
| KNN | n_neighbors=5 | 82.72% |
| Decision Tree | max_depth=7 | 82.27% |
| Ridge | alpha=1.0 | 82.16% |
| Lasso / ElasticNet | L1 / L1+L2 | ~78% |

**Эксперименты:**
- KNN с разными k: [3,5,7,9,11]
- Линейные модели с разными типами нормализации (StandardScaler, MinMaxScaler, RobustScaler)
- Бустинги с разными гиперпараметрами

**Файл:** `train_ml.py`

---

## 5. Feature Engineering

✅ **Что сделано:**
- **Декомпозиция:** `Title` (из имени), `FamilySize` = SibSp + Parch + 1
- **Комбинации фичей:** `Pclass_Sex`, `Title_Age`
- **Трансформации:** `AgeGroup` (5 групп), `FareGroup` (4 квартиля)

**Файл:** `preprocess.py`

---

## 6. Ансамбли

✅ **Что сделано:**

| Метод | Точность |
|-------|----------|
| Voting (soft) | 83.28% |
| Averaging (усреднение) | 83.28% |
| Stacking (Ridge) | 83.27% |

**Вывод:** Ансамбли не превзошли лучшую одиночную модель (LightGBM 84.51%), но повысили стабильность предсказаний.

**Файл:** `ensemble.py`

---

## 7. Deep Neural Network (DNN)

✅ **Что сделано:**

| Пункт | Реализация |
|-------|-----------|
| MLP из двух слоёв | MLPClassifier (64, 32) |
| Добавление больше слоёв | MLPClassifier (128, 64, 32) |
| BatchNorm | ✅ PyTorch (`train_dnn_advanced.py`) |
| Dropout с разными значениями | PyTorch: dropout_rate=0.2, 0.3, 0.4 |
| Разные размеры слоёв и активации | (64,32), (128,64,32); relu, tanh |
| Разные оптимизаторы | Adam, SGD |
| Scheduler (косинусовый) | ✅ PyTorch (`train_dnn_advanced.py`) |
| Разные параметры (LR, batchsize, эпохи) | 0.001/0.0001, 32/64, 100/200 |
| **Embedding (звёздочка)** | ✅ PyTorch (`train_dnn_embedding.py`) |

### Результаты DNN

| Модель | Точность |
|--------|----------|
| DNN с Embedding | 83.72% |
| DNN Advanced (PyTorch) | 82.60% |
| DNN базовый (MLPClassifier) | 81.93% |

**Вывод:** Embedding дал прирост ~1% к обычному DNN и сравнился с XGBoost.

**Файлы:** `train_dnn.py`, `train_dnn_advanced.py`, `train_dnn_embedding.py`

---

## 8. Лучший результат

🏆 **Лучшая модель:** **LightGBM**  
📊 **Точность (5 фолдов):** **84.51%**

---

## 9. Выводы по проекту

1. **Данные:** Пропуски и выбросы были корректно обработаны.
2. **Feature Engineering:** Новые фичи повысили точность на ~1.5%.
3. **Модели:** Бустинги (LightGBM, XGBoost, CatBoost) показали лучшие результаты.
4. **Ансамбли:** Voting и Stacking дали стабильные результаты.
5. **DNN:** Реализована на PyTorch с BatchNorm, Dropout, CosineAnnealingLR и Embedding.
6. **Лучший результат:** LightGBM — **84.51%**.

---

## 10. Ссылка на GitHub

[https://github.com/Valeriya-72/titanic-ml-project](https://github.com/Valeriya-72/titanic-ml-project)

---

**Дата сдачи:** апрель 2026