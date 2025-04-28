import pandas as pd
import numpy as np
import os
import warnings
import joblib
import time
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


def calculate_metrics(y_true, y_pred):
    """Calculates MAE, RMSE, sMAPE (0-200 scale), and R2."""
    y_pred = np.nan_to_num(y_pred); y_true = np.nan_to_num(y_true)
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred): return {'MAE': np.nan, 'RMSE': np.nan, 'sMAPE': np.nan, 'R2': np.nan}
    mae = mean_absolute_error(y_true, y_pred); rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    numerator = np.abs(y_pred - y_true); denominator = (np.abs(y_true) + np.abs(y_pred))
    smape_vals = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator!=0)
    smape = np.mean(smape_vals) * 200
    try:
        if np.var(y_true) < 1e-9: r2 = np.nan
        else: r2 = r2_score(y_true, y_pred)
    except ValueError: r2 = np.nan
    return {'MAE': mae, 'RMSE': rmse, 'sMAPE': smape, 'R2': r2}


class DataManagerSimplified:
    """
    Загружает и подготавливает базовый DataFrame (продажи + календарь + цены)
    для указанного магазина и набора файлов. НЕ СОЗДАЕТ ПРОИЗВОДНЫЕ ПРИЗНАКИ.
    """
    def __init__(self, store_id):
        self.store_id = store_id
        # Жестко заданный маппинг item_id для STORE_1
        self.item_id_map = {
            'STORE_1_064': 0, 'STORE_1_065': 1, 'STORE_1_090': 2, 'STORE_1_252': 3,
            'STORE_1_325': 4, 'STORE_1_339': 5, 'STORE_1_376': 6, 'STORE_1_546': 7,
            'STORE_1_547': 8, 'STORE_1_555': 9, 'STORE_1_584': 10, 'STORE_1_586': 11,
            'STORE_1_587': 12, 'STORE_1_714': 13, 'STORE_1_727': 14
        }
        self.item_id_map_reverse = {v: k for k, v in self.item_id_map.items()}
        print(f"DataManagerSimplified инициализирован для магазина: {self.store_id}")

    def _load_external_data(self, calendar_path, prices_path):
        """Загружает календарь и цены из указанных путей."""
        print(f"  Загрузка календаря из: {calendar_path}")
        print(f"  Загрузка цен из: {prices_path}")
        calendar_df, prices_df = None, None
        try:
            calendar_df = pd.read_csv(calendar_path, parse_dates=['date'])
            required_cal = ['date', 'date_id', 'wm_yr_wk', 'wday', 'month', 'year'] # Базовые календарные поля
            if not all(c in calendar_df.columns for c in required_cal): raise ValueError("Календарь: отсутствуют колонки")
            print(f"    Календарь загружен: {calendar_df.shape}")

            prices_full = pd.read_csv(prices_path)
            prices_df = prices_full[prices_full['store_id'] == self.store_id].copy()
            if 'store_id' in prices_df.columns: prices_df.drop('store_id', axis=1, inplace=True)
            required_price = ['item_id', 'wm_yr_wk', 'sell_price']
            if not all(c in prices_df.columns for c in required_price): raise ValueError("Цены: отсутствуют колонки")
            print(f"    Цены загружены и отфильтрованы: {prices_df.shape}")
            return calendar_df, prices_df
        except Exception as e: print(f"  Ошибка загрузки внешних данных: {e}"); return None, None

    def _preprocess_sales(self, sales_path, calendar_df):
        """Загружает, фильтрует продажи и добавляет дату, используя календарь."""
        print(f"  Предобработка продаж из: {os.path.basename(sales_path)}...")
        if calendar_df is None: print("  Ошибка: Календарь не предоставлен."); return None
        try:
            sales_df = pd.read_csv(sales_path); sales_store_df = sales_df[sales_df['store_id'] == self.store_id].copy()
            if sales_store_df.empty: print(f"  Ошибка: Нет данных для {self.store_id}"); return None
            if 'store_id' in sales_store_df.columns: sales_store_df.drop('store_id', axis=1, inplace=True)
            required_sales = ['item_id', 'date_id', 'cnt']
            if not all(c in sales_store_df.columns for c in required_sales): raise ValueError("Продажи: отсутствуют колонки")

            date_map = calendar_df.set_index('date_id')['date']
            sales_store_df['date'] = sales_store_df['date_id'].map(date_map)
            initial_rows = len(sales_store_df); sales_store_df.dropna(subset=['date'], inplace=True)
            if len(sales_store_df) < initial_rows: print(f"    Удалено {initial_rows - len(sales_store_df)} строк с ненайденными датами.")

            # Оставляем ТОЛЬКО нужные колонки
            result_df = sales_store_df[['item_id', 'date', 'cnt']].reset_index(drop=True)
            result_df['date'] = pd.to_datetime(result_df['date'])
            print(f"    Продажи предобработаны: {result_df.shape}")
            return result_df
        except Exception as e: print(f"  Ошибка обработки продаж: {e}"); return None

    def get_base_data(self, sales_path, calendar_path, prices_path):
        """
        Основной метод для получения базового DataFrame (продажи + календарь + цены).

        Возвращает:
            pd.DataFrame or None: Базовый DataFrame или None в случае ошибки.
        """
        print(f"\n--- Получение базовых данных для sales: {os.path.basename(sales_path)} ---")
        calendar_df, prices_df = self._load_external_data(calendar_path, prices_path)
        if calendar_df is None or prices_df is None: return None
        sales_processed_df = self._preprocess_sales(sales_path, calendar_df)
        if sales_processed_df is None: return None

        # --- Объединение данных ---
        print("  Объединение данных...")
        # Выбираем нужные колонки календаря
        cal_cols = ['date', 'wm_yr_wk', 'wday', 'month', 'year'] # Базовые
        calendar_subset = calendar_df[cal_cols].drop_duplicates(subset=['date']).copy()
        # Объединяем продажи и календарь
        base_df = pd.merge(sales_processed_df, calendar_subset, on='date', how='left')
        # Объединяем с ценами
        prices_to_merge = prices_df[['item_id', 'wm_yr_wk', 'sell_price']].drop_duplicates()
        base_df = pd.merge(base_df, prices_to_merge, on=['item_id', 'wm_yr_wk'], how='left')
        # Заполняем пропуски цен
        base_df['sell_price'] = base_df.groupby('item_id')['sell_price'].transform(lambda x: x.ffill().bfill())
        base_df['sell_price'].fillna(0, inplace=True)
        # Заполняем пропуски в календарных данных
        cal_fill_cols = ['wday', 'month', 'year', 'wm_yr_wk']
        for col in cal_fill_cols:
             if col in base_df.columns and base_df[col].isnull().any():
                  base_df[col] = base_df[col].fillna(method='ffill').fillna(method='bfill')
                  if base_df[col].isnull().any(): base_df[col].fillna(0, inplace=True)
        # Удаляем промежуточную колонку
        base_df = base_df.drop(columns=['wm_yr_wk'], errors='ignore')
        # Сортируем
        base_df.sort_values(by=['item_id', 'date'], inplace=True, ignore_index=True)
        print(f"  Базовый DataFrame готов: {base_df.shape}")
        return base_df


class LGBMForecaster:
    """
    Обучает модель LightGBM, генерирует признаки, сохраняет/загружает
    состояние и делает прогнозы. Использует УПРОЩЕННЫЙ набор признаков.
    """
    def __init__(self, store_id, model_dir="models_lgbm_simple_v3"): # Новая папка
        self.store_id = store_id
        self.model_dir = model_dir
        self.model_lgbm = None
        self.features = [] # Список имен признаков для модели
        self.categorical_features = [] # Список имен категориальных признаков
        self.history_df = None # DataFrame с историей для расчета лагов
        self.is_fitted = False
        # Жестко заданный маппинг item_id (должен быть консистентен с DataManager)
        self.item_id_map = {
            'STORE_1_064': 0, 'STORE_1_065': 1, 'STORE_1_090': 2, 'STORE_1_252': 3,
            'STORE_1_325': 4, 'STORE_1_339': 5, 'STORE_1_376': 6, 'STORE_1_546': 7,
            'STORE_1_547': 8, 'STORE_1_555': 9, 'STORE_1_584': 10, 'STORE_1_586': 11,
            'STORE_1_587': 12, 'STORE_1_714': 13, 'STORE_1_727': 14
        }
        self.item_id_map_reverse = {v: k for k, v in self.item_id_map.items()}
        os.makedirs(self.model_dir, exist_ok=True)
        print(f"LGBMForecaster инициализирован для магазина: {self.store_id}. Папка модели: '{self.model_dir}'")

    def _get_numeric_item_id(self, item_id_str):
        """Применяет маппинг для получения числового ID."""
        return self.item_id_map.get(item_id_str, -1)

    def _create_features_internal(self, df, target_col='cnt', is_train=True):
        """
        Внутренний метод для создания признаков.
        Исправлена логика расчета лагов/окон для инференса v3.
        """
        print(f"  Создание признаков LGBM. Режим обучения: {is_train}. Входная форма: {df.shape}")
        df_processed = df.copy()
        if not all(c in df_processed.columns for c in ['item_id', 'date']): raise ValueError("Нужны 'item_id' и 'date'.")
        df_processed['date'] = pd.to_datetime(df_processed['date'])

        # 1. Календарные признаки
        df_processed['dayofweek'] = df_processed['date'].dt.dayofweek; df_processed['year'] = df_processed['date'].dt.year
        df_processed['dayofmonth'] = df_processed['date'].dt.day; df_processed['weekofyear'] = df_processed['date'].dt.isocalendar().week.astype(int)
        df_processed['dayofyear'] = df_processed['date'].dt.dayofyear

        # 2. Лаговые и скользящие признаки
        lags = [7, 14, 21, 28, 35, 90]; windows = [7, 14, 28]; base_shift = 7
        lag_roll_cols = [f'{target_col}_lag_{lag}' for lag in lags] + \
                        [f'{target_col}_roll_mean_{w}' for w in windows] + \
                        [f'{target_col}_roll_std_{w}' for w in windows]

        if is_train and target_col in df_processed.columns:
            # --- Режим ОБУЧЕНИЯ: Рассчитываем на текущем df ---
            print(f"    Расчет лагов/окон из текущего df (режим обучения)...")
            df_processed = df_processed.sort_values(by=['item_id', 'date'])
            for lag in lags: df_processed[f'{target_col}_lag_{lag}'] = df_processed.groupby('item_id')[target_col].shift(lag)
            for window in windows:
                shifted = df_processed.groupby('item_id')[target_col].shift(base_shift)
                df_processed[f'{target_col}_roll_mean_{window}'] = shifted.rolling(window, min_periods=1).mean()
                df_processed[f'{target_col}_roll_std_{window}'] = shifted.rolling(window, min_periods=2).std()
            # --- КОНЕЦ РЕЖИМА ОБУЧЕНИЯ ---

        elif not is_train and self.history_df is not None and target_col in self.history_df.columns:
            # --- Режим ПРОГНОЗА: Используем явное объединение с историей ---
            print(f"    Расчет лагов/окон с использованием history_df (режим прогноза)...")
            hist_subset = self.history_df[['item_id', 'date', target_col]].copy()
            hist_subset['date'] = pd.to_datetime(hist_subset['date'])
            # Берем структуру будущего (без target_col)
            future_structure = df_processed[['item_id', 'date']].copy()
            future_structure['date'] = pd.to_datetime(future_structure['date'])

            # Объединяем историю и структуру будущего
            combined_df = pd.concat([hist_subset, future_structure], ignore_index=True)\
                           .sort_values(['item_id', 'date'])\
                           .drop_duplicates(['item_id', 'date'], keep='last') # keep='last' сохранит NaN из future, если даты пересекаются

            print(f"    Размер объединенного df для расчета: {combined_df.shape}")

            # Рассчитываем лаги и окна на ОБЪЕДИНЕННОМ df
            lag_roll_features_dict = {}
            for lag in lags: lag_roll_features_dict[f'{target_col}_lag_{lag}'] = combined_df.groupby('item_id')[target_col].shift(lag)
            for window in windows:
                shifted = combined_df.groupby('item_id')[target_col].shift(base_shift)
                lag_roll_features_dict[f'{target_col}_roll_mean_{window}'] = shifted.rolling(window, min_periods=1).mean()
                lag_roll_features_dict[f'{target_col}_roll_std_{window}'] = shifted.rolling(window, min_periods=2).std()

            # Создаем DataFrame только с лагами/окнами
            df_lag_roll = pd.DataFrame(lag_roll_features_dict)
            df_lag_roll['date'] = combined_df['date']
            df_lag_roll['item_id'] = combined_df['item_id']

            # Выбираем строки, соответствующие ТОЛЬКО будущим датам (из df_processed)
            # и мержим их к df_processed
            features_to_merge = df_lag_roll[df_lag_roll['date'].isin(df_processed['date'])].copy()
            df_processed = pd.merge(df_processed, features_to_merge, on=['item_id', 'date'], how='left')
            # --- КОНЕЦ РЕЖИМА ПРОГНОЗА ---
        else:
            print(f"    Внимание: Невозможно рассчитать лаги/окна. Пропуск.")
            # Создаем пустые колонки, чтобы модель не ругалась при predict
            for col in lag_roll_cols: df_processed[col] = 0

        # 3. Числовой ID Товара
        df_processed['item_id_numeric'] = df_processed['item_id'].apply(self._get_numeric_item_id)

        # 4. Удаление исходного ID
        cols_to_drop = ['item_id', 'event_name_1', 'event_type_1']
        df_processed.drop(columns=[col for col in cols_to_drop if col in df_processed.columns], inplace=True, errors='ignore')

        # 5. Заполнение пропусков и оптимизация типов
        feature_cols_final = [col for col in df_processed.columns if col not in [target_col, 'date']]
        print(f"    Заполнение NaN и оптимизация типов для {len(feature_cols_final)} признаков...")
        for col in feature_cols_final:
            if pd.api.types.is_numeric_dtype(df_processed[col].dtype):
                if df_processed[col].isnull().any():
                    df_processed[col] = df_processed[col].fillna(0) # Заполняем нулями (особенно важно для лагов/окон)

            # Оптимизация памяти
            if df_processed[col].dtype == 'float64': df_processed[col] = df_processed[col].astype('float32')
            elif pd.api.types.is_integer_dtype(df_processed[col].dtype):
                 max_v, min_v = df_processed[col].max(), df_processed[col].min()
                 if max_v < 127 and min_v >= -128: df_processed[col] = df_processed[col].astype('int8')
                 elif max_v < 32767 and min_v >= -32768: df_processed[col] = df_processed[col].astype('int16')
                 elif max_v < 2147483647 and min_v >= -2147483648: df_processed[col] = df_processed[col].astype('int32')

        print(f"  Создание признаков LGBM завершено. Форма: {df_processed.shape}")
        return df_processed

    def fit(self, train_base_df, lgbm_params=None):
        """
        Обучает модель LightGBM на предоставленном базовом DataFrame.

        Args:
            train_base_df (pd.DataFrame): DataFrame с базовыми данными для обучения
                                          (результат DataManager.get_base_data).
            lgbm_params (dict, optional): Параметры для LightGBM.
        """
        print(f"\n--- Запуск обучения LGBM для {self.store_id} ---")
        fit_start_time = time.time()
        TARGET = 'cnt'

        if train_base_df is None or TARGET not in train_base_df.columns:
            print("  Ошибка: Предоставлен некорректный train_base_df."); return

        # 1. Создание признаков для обучения
        try:
            # Сохраняем историю ДО создания признаков (нужны item_id, date, cnt, sell_price)
            hist_cols = ['item_id', 'date', TARGET, 'sell_price']
            if all(c in train_base_df.columns for c in hist_cols):
                 self.history_df = train_base_df[hist_cols].copy()
                 print(f"  History df сохранен для использования в predict: {self.history_df.shape}")
            else:
                 print("  Внимание: Не удалось сохранить history_df (отсутствуют колонки). Predict может не сработать.")
                 self.history_df = None # Явно устанавливаем None

            # Создаем признаки, передавая базовый df
            train_features_df = self._create_features_internal(train_base_df, target_col=TARGET, is_train=True)

            # Удаляем NaN после создания лагов/окон
            initial_rows = len(train_features_df)
            train_features_df.dropna(inplace=True) # Удаляем строки с любыми NaN
            print(f"  Удалено {initial_rows - len(train_features_df)} строк из-за NaNs в признаках.")
            print(f"  Размер данных для обучения после dropna: {train_features_df.shape}")
            if train_features_df.empty: print("  Ошибка: Нет данных для обучения после удаления NaN."); return

            # Определяем финальный список признаков и категорий
            self.features = [col for col in train_features_df.columns if col not in [TARGET, 'date']]
            potential_cats = ['item_id_numeric', 'dayofweek', 'year', 'dayofmonth', 'weekofyear', 'dayofyear', 'wday']
            self.categorical_features = [col for col in potential_cats if col in self.features]

            X_train = train_features_df[self.features]
            y_train = train_features_df[TARGET]

        except Exception as e: print(f"  Ошибка при создании признаков или подготовке X/y: {e}"); return

        print(f"  Обучение на {len(X_train)} строках, {len(self.features)} признаках.")
        print(f"  Категориальные признаки: {self.categorical_features}")

        # 2. Обучение модели
        if lgbm_params is None:
             lgbm_params = { 'objective': 'regression_l1', 'n_estimators': 57, 'learning_rate': 0.1,
                            'num_leaves': 31, 'feature_fraction': 0.8, 'bagging_fraction': 0.9,
                            'bagging_freq': 1, 'verbose': -1, 'n_jobs': -1, 'seed': 42, 'boosting_type': 'gbdt'}
             print("  Используются параметры LGBM по умолчанию/подобранные.")
        self.model_lgbm = lgb.LGBMRegressor(**lgbm_params)
        try:
            print("  Обучение модели LGBM...")
            self.model_lgbm.fit(X_train, y_train, categorical_feature=self.categorical_features)
            self.is_fitted = True
            fit_end_time = time.time()
            print(f"  Модель LGBM успешно обучена за {fit_end_time - fit_start_time:.2f} сек.")
            # Обновляем список признаков на случай внутренних изменений LGBM
            if hasattr(self.model_lgbm, 'booster_') and hasattr(self.model_lgbm.booster_, 'feature_name'):
                 self.features = self.model_lgbm.booster_.feature_name()
        except Exception as e: print(f"  Ошибка обучения модели LGBM: {e}"); self.is_fitted = False

        # 3. Сохранение модели и атрибутов
        if self.is_fitted: self.save_model()

    def predict(self, predict_base_df): # <<< Принимает готовый базовый df
        """
        Генерирует прогнозы на основе предоставленного базового DataFrame.
        Ожидает df с колонками 'item_id', 'date', 'sell_price', 'wday', 'month', 'year'.
        """
        if not self.is_fitted: raise RuntimeError("Модель не обучена.")
        if self.history_df is None: raise RuntimeError("Отсутствуют исторические данные (history_df).")
        if predict_base_df is None or predict_base_df.empty: print("Ошибка: predict_base_df не предоставлен или пуст."); return None
        required_cols = ['item_id', 'date', 'sell_price', 'wday', 'month', 'year']
        if not all(c in predict_base_df.columns for c in required_cols):
             missing = [c for c in required_cols if c not in predict_base_df.columns]
             print(f"Ошибка: predict_base_df не хватает колонок: {missing}."); return None

        print(f"\n--- Генерация прогнозов LGBM ---")
        predict_start_time = time.time()
        TARGET = 'cnt' # Имя целевой колонки для create_features

        # 1. Создание признаков для прогноза, используя predict_base_df и self.history_df
        print("  Создание признаков для прогноза...")
        try:
            predict_features_df = self._create_features_internal(
                predict_base_df.copy(), # Передаем копию базового df
                target_col=TARGET,
                is_train=False # Указываем режим прогноза
            )
        except Exception as e: print(f"  Ошибка при создании признаков для прогноза: {e}"); return None

        # 2. Проверка и выравнивание признаков
        missing_features = [f for f in self.features if f not in predict_features_df.columns]
        if missing_features:
             print(f"  Внимание: Отсутствуют признаки: {missing_features}. Заполняем нулями.")
             for f in missing_features: predict_features_df[f] = 0
        try:
             X_pred = predict_features_df[self.features] # Выбираем и упорядочиваем
        except KeyError as e: print(f"  Ошибка: Колонки, ожидаемые моделью, не найдены: {e}"); return None

        # 3. Прогнозирование
        print(f"  Прогнозирование на {X_pred.shape[0]} строках...")
        try:
             y_pred = self.model_lgbm.predict(X_pred)
             y_pred = np.maximum(0, y_pred)
        except Exception as e: print(f"  Ошибка во время прогнозирования: {e}"); return None

        # 4. Формирование результата
        result_df = pd.DataFrame({
            'date': predict_base_df['date'].values, # Берем из исходного базового df
            'item_id': predict_base_df['item_id'].values, # Берем из исходного базового df
            'forecast': y_pred
        })
        predict_end_time = time.time()
        print(f"  Генерация прогнозов завершена за {predict_end_time - predict_start_time:.2f} сек.")
        return result_df

    def save_model(self):
        """Сохраняет обученную модель и необходимые атрибуты."""
        if not self.is_fitted: print("  Модель не обучена, сохранение невозможно."); return
        print(f"  Сохранение компонентов модели в {self.model_dir}...")
        paths = {
            'model': os.path.join(self.model_dir, f"lgbm_model_{self.store_id}.joblib"),
            'features': os.path.join(self.model_dir, f"features_{self.store_id}.joblib"),
            'cats': os.path.join(self.model_dir, f"categorical_features_{self.store_id}.joblib"),
            'history': os.path.join(self.model_dir, f"history_df_{self.store_id}.joblib")
        }
        objects_to_save = {
            'model': self.model_lgbm, 'features': self.features, 'cats': self.categorical_features,
             'history': self.history_df
        }
        save_errors = 0
        for name, obj in objects_to_save.items():
            if obj is not None:
                try: joblib.dump(obj, paths[name])
                except Exception as e: print(f"    Ошибка сохранения {name}: {e}"); save_errors += 1
            # Не ругаемся, если нет history или events
        if save_errors == 0: print(f"  Компоненты модели сохранены.")
        else: print(f"  {save_errors} ошибки при сохранении.")

    def load_model(self):
        """Загружает обученную модель и необходимые атрибуты."""
        print(f"  Загрузка компонентов модели из {self.model_dir}...")
        paths = {
            'model': os.path.join(self.model_dir, f"lgbm_model_{self.store_id}.joblib"),
            'features': os.path.join(self.model_dir, f"features_{self.store_id}.joblib"),
            'cats': os.path.join(self.model_dir, f"categorical_features_{self.store_id}.joblib"),
            'history': os.path.join(self.model_dir, f"history_df_{self.store_id}.joblib")
        }
        loaded_correctly = True
        # Атрибуты, которые нужно загрузить
        attributes_to_load = ['model_lgbm', 'features', 'categorical_features',
                              'history_df']
        # Соответствие ключей путей и имен атрибутов
        paths_keys = ['model', 'features', 'cats',
                       'history']

        for name_key, attr_name in zip(paths_keys, attributes_to_load):
            path = paths[name_key]
            if os.path.exists(path):
                 try: setattr(self, attr_name, joblib.load(path)); print(f"    {attr_name} загружен.")
                 except Exception as e: print(f"    Ошибка загрузки {name_key}: {e}"); loaded_correctly = False; setattr(self, attr_name, None)
            else:
                 print(f"    Файл для {name_key} ('{path}') не найден.")
                 # History может быть опциональной при загрузке, но обязательной для predict
                 if attr_name != 'history_df': loaded_correctly = False
                 setattr(self, attr_name, None)

        # is_fitted зависит от наличия всех этих компонентов
        self.is_fitted = loaded_correctly and all(getattr(self, attr, None) is not None for attr in attributes_to_load)

        if self.is_fitted: print("  Компоненты модели успешно загружены.")
        else: print("  Не удалось загрузить все необходимые компоненты. Может потребоваться переобучение.")
        return self.is_fitted

    def evaluate(self, y_true_df, y_pred_df):
        """Оценивает качество прогноза."""
        print("\n--- Оценка качества прогноза ---"); required_true = ['date', 'item_id', 'cnt']; required_pred = ['date', 'item_id', 'forecast']
        if not all(c in y_true_df.columns for c in required_true): print("Ошибка: y_true_df не хватает колонок."); return None
        if not all(c in y_pred_df.columns for c in required_pred): print("Ошибка: y_pred_df не хватает колонок."); return None
        if y_true_df.empty or y_pred_df.empty: print("Ошибка: Входные DataFrame пусты."); return None
        try:
            y_true_df['date'] = pd.to_datetime(y_true_df['date']); y_pred_df['date'] = pd.to_datetime(y_pred_df['date'])
            merged = pd.merge(y_true_df[required_true], y_pred_df[required_pred], on=['date', 'item_id'], how='inner')
        except Exception as e: print(f"Ошибка при слиянии данных для оценки: {e}"); return None
        if merged.empty: print("  Ошибка: Нет совпадающих данных для оценки."); return None
        merged.sort_values(['item_id', 'date'], inplace=True); min_date, max_date = merged['date'].min(), merged['date'].max()
        available_days = (max_date - min_date).days + 1; print(f"  Период оценки: {min_date.date()} по {max_date.date()} ({available_days} дн.)")
        results = {}; horizons = {'week': 7, 'month': 30, 'quarter': 90}
        for name, days in horizons.items():
            results[name] = {k: np.nan for k in ['MAE', 'RMSE', 'sMAPE', 'R2']}
            if available_days >= days:
                print(f"  Расчет метрик для горизонта: {name} ({days} дн.)"); horizon_df = merged[merged['date'] <= min_date + pd.Timedelta(days=days-1)].copy()
                if horizon_df.empty: print(f"    Внимание: Нет данных для горизонта '{name}'."); continue
                horizon_df['cnt'] = pd.to_numeric(horizon_df['cnt'], errors='coerce'); horizon_df['forecast'] = pd.to_numeric(horizon_df['forecast'], errors='coerce')
                horizon_df.dropna(subset=['cnt', 'forecast'], inplace=True)
                if horizon_df.empty: print(f"    Внимание: Нет валидных числовых данных для горизонта '{name}'."); continue
                try:
                    metrics_per_item = horizon_df.groupby('item_id').apply(lambda x: pd.Series(calculate_metrics(x['cnt'], x['forecast'])))
                    if metrics_per_item.empty or metrics_per_item.isnull().all().all(): print(f"    Внимание: Расчет метрик дал пустой результат для '{name}'."); continue
                    metrics_avg = metrics_per_item.mean().to_dict(); results[name] = metrics_avg
                    print(f"    Метрики ({name}): MAE={metrics_avg.get('MAE',np.nan):.4f}, RMSE={metrics_avg.get('RMSE',np.nan):.4f}, sMAPE={metrics_avg.get('sMAPE',np.nan):.4f}, R2={metrics_avg.get('R2',np.nan):.4f}")
                except Exception as e: print(f"    Ошибка расчета метрик для {name}: {e}")
            else: print(f"  Пропуск горизонта {name}, нужно {days} дн., доступно {available_days}.")
        return results