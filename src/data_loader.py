import pandas as pd
from typing import Optional, Dict

def load_lms_data(filepath: str) -> pd.DataFrame:
    """
    Загружает CSV файл с логами LMS.

    Parameters:
        filepath (str): Путь к CSV файлу.

    Returns:
        pd.DataFrame: DataFrame с логами.
    """
    df = pd.read_csv(filepath)
    print(f"Данные загружены. Строк: {df.shape[0]}, Столбцов: {df.shape[1]}")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Базовая предобработка данных: парсинг дат, фильтрация.
    """
    # Предполагаем, что столбец с датой называется 'timestamp'
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()

    # Оставляем только нужные типы событий для анализа
    event_types_to_keep = ['login', 'assignment_submitted', 'quiz_attempted', 'forum_posted']
    if 'event_type' in df.columns:
        df = df[df['event_type'].isin(event_types_to_keep)].copy()

    print("Предобработка данных завершена.")
    return df