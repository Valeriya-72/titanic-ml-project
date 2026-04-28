# Titanic Survival Prediction

Проект по предсказанию выживания пассажиров Титаника (бинарная классификация).

## 📊 Лучший результат

🏆 **LightGBM** — **84.51%** (Stratified K-fold, 5 фолдов)

## 🧠 Ключевые этапы

| Этап | Что сделано |
|------|-------------|
| **EDA** | Анализ пропусков, распределений, корреляций, выбросов |
| **Preprocessing** | Заполнение пропусков, нормализация, кодирование категорий |
| **Feature Engineering** | `Title`, `FamilySize`, `AgeGroup`, `FareGroup`, комбинации `Pclass_Sex`, `Title_Age` |
| **Validation** | Stratified K-fold (5), сравнение 1 и 5 фолдов |
| **Models** | Linear (Ridge, Lasso, ElasticNet), KNN, Tree, Random Forest, CatBoost, LightGBM, XGBoost |
| **DNN** | PyTorch: BatchNorm, Dropout, CosineAnnealingLR, **Embedding** |
| **Ensembles** | Voting, Averaging, Stacking (Ridge) |

## 📁 Структура проекта

- `eda.py` — разведочный анализ
- `preprocess.py` — предобработка и feature engineering
- `train_ml.py` — обучение ML-моделей
- `train_dnn.py` — DNN на sklearn
- `train_dnn_advanced.py` — DNN на PyTorch (BatchNorm, scheduler)
- `train_dnn_embedding.py` — DNN с Embedding
- `ensemble.py` — ансамбли
- `compare_validation.py` — сравнение 1 и 5 фолдов
- `compare_preprocessing.py` — сравнение до/после предобработки

## 🚀 Запуск проекта

1. Установить зависимости:  
   `pip install -r requirements.txt`
2. Запустить EDA:  
   `python eda.py`
3. Обучить ML-модели:  
   `python train_ml.py`
4. Обучить DNN:  
   `python train_dnn_advanced.py`
5. Запустить ансамбли:  
   `python ensemble.py`

## 📈 Результаты моделей

| Модель | Точность (5 фолдов) |
|--------|---------------------|
| LightGBM | **84.51%** |
| Random Forest | 83.95% |
| XGBoost | 83.72% |
| DNN с Embedding | 83.72% |
| CatBoost | 83.05% |
| KNN | 82.72% |
| Decision Tree | 82.27% |
| Ridge | 82.16% |

## 📎 Ссылки

- [GitHub репозиторий](https://github.com/Valeriya-72/titanic-ml-project)
- [REPORT.md — подробный отчёт](REPORT.md)

---

**Дата выполнения:** апрель 2026