# Прогноз спроса на велосипеды Divvy

**[Полный отчёт об эксперименте и результатах →](REPORT.md)**

Тестовое задание: прогноз числа стартов поездок на станции в **следующий час** по поездкам Chicago Divvy за 2024 год. Для строки с `timestamp = t` целевая переменная — число стартов в час `t`, а `demand_lag_1` — фактическое число стартов в час `t−1`. Это режим почасового обновления прогноза после закрытия предыдущего часа.

## Что сделано

- Сбор почасового спроса из всех 12 CSV потоковыми порциями. Исходный подготовленный parquet содержал только январь–май.
- Исправлено выравнивание цели и лагов. Старые метрики из `forecasting_metrics.csv` с новыми напрямую не сопоставимы.
- Добавлены лаги 1, 2, 3, 24, 25, 48, 168, 169 часов, скользящие средние за 3, 24 и 168 часов, календарные и циклические признаки. Все исторические признаки вычисляются с `shift(1)` или большим сдвигом.
- Ограниченный подбор 2 конфигураций HistGradientBoosting и 2 конфигураций CatBoost на июльском окне. CatBoost использует ID станции как категорию; HistGradientBoosting использует координаты без ID станции.
- Проверка на трёх хронологических окнах: 1–14 августа, 1–14 октября, 1–14 ноября. Для каждого окна обучение использует только более ранние часы. Декабрь оставлен для однократной финальной проверки.
- Метрики MAE, RMSE, WAPE, R², а также отдельные результаты для станций с высокой активностью, часов пик, нулевого спроса, будней и выходных. Стандартное отклонение считается по трём folds.
- Число строк обучения ограничено равномерной случайной выборкой **только внутри доступного прошлого**. Это ускоряет подбор и ограничивает расход RAM; в отчёте показано, сколько строк было доступно и использовано.

## Запуск

Из корня проекта, в PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\data\downloader.py
.\.venv\Scripts\python.exe src\features\build_demand.py
.\.venv\Scripts\python.exe src\features\create_features.py
.\.venv\Scripts\python.exe src\forecasting\train.py --max-train-rows 800000 --tune-rows 400000 --threads 8
.\.venv\Scripts\python.exe src\forecasting\training_size_experiment.py --rows 2000000
.\.venv\Scripts\python.exe src\forecasting\training_size_experiment.py --rows 4000000
.\.venv\Scripts\python.exe src\forecasting\train.py --reuse-results --final-train-rows 2000000
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Сырые файлы ожидаются в `data/raw/divvy/2024??-divvy-tripdata.csv`. Для другого расположения можно передать `--raw-dir`. Параметры `--max-train-rows` и `--tune-rows` позволяют обменять время обучения на количество использованных наблюдений. Если CatBoost недоступен, скрипт выполняет поиск только среди HistGradientBoosting.

## Результаты

Краткие числа и выводы: [results/reports/SUMMARY.md](results/reports/SUMMARY.md). Обзор статей и похожих проектов: [RESEARCH.md](RESEARCH.md).

| Файл | Назначение |
| --- | --- |
| `results/reports/tuning.csv` | Подбор параметров на июльском окне |
| `results/reports/fold_metrics.csv` | Метрики по folds и сегментам |
| `results/reports/model_comparison.csv` | Средние метрики, разброс и итоговый рейтинг |
| `results/reports/training_size_experiment.csv` | Влияние объёма исторических строк на качество и время |
| `results/reports/holdout_metrics.csv` | Проверка выбранной модели и baseline на декабре |
| `results/reports/selected_model.json` | Конфигурация, признаки, правило выбора и путь к модели |
| `results/figures/model_comparison.png` | Сравнение MAE и RMSE по folds |
| `results/models/best_model.*` | Модель, обученная только на данных до 1 декабря |

Правило выбора среди обученных моделей: сначала считается `mean(MAE) + 0.25 × std(MAE) + 0.05 × mean(RMSE)`. Среди моделей с итогом не хуже 1% от лучшего выбирается самая быстрая в обучении. Это учитывает качество, устойчивость и стоимость повторного обучения. Baseline служат точкой сравнения и не участвуют в выборе финальной обученной модели. Если подбор и folds уже завершились, `--reuse-results` пересчитает итоговый выбор и holdout без повторного обучения всех кандидатов.

## Условия и ограничения

Панель станций начинается после первой зарегистрированной поездки. Часы без стартов получают нулевую цель. Станция включается в прогнозы в течение 30 дней после последней поездки, чтобы не объявлять закрытие станции по будущим данным; это эвристика при отсутствии официального календаря работы станций. Метрика на множестве нулевых часов может выглядеть хорошей даже при ошибках на загруженных станциях, поэтому приведены сегментные метрики. Число зарегистрированных поездок не равно истинному спросу, если на станции не было доступных велосипедов. Для прогноза на несколько часов вперёд без промежуточных фактических наблюдений нужен отдельный рекурсивный или прямой многогоризонтный эксперимент.

## Исследовательская основа

- [scikit-learn: Lagged features for time series forecasting](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html) — временное разделение и проверка лаговых моделей.
- [scikit-learn: Time-related feature engineering](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html) — циклические признаки и временная валидация.
- [Divvy bike rebalancing, GitHub](https://github.com/rishi-kumar0208/divvy-bike-rebalancing) — станционная панель после открытия, лаги без будущих данных, категориальный ID станции.
- [LightGBM: Features](https://lightgbm.readthedocs.io/en/latest/Features.html) — обоснование гистограммного бустинга и категориальных признаков на больших таблицах; здесь проверены доступные HistGradientBoosting и CatBoost.
- [Feng et al., ACM TKDD 2024](https://doi.org/10.1145/3690388) — пространственная и временная неоднородность спроса и операционная связь с ребалансировкой.
