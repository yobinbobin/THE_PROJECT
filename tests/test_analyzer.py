import pandas as pd
import numpy as np
import pytest
import sys
import os

# Добавляем путь к src для импорта
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_lms_data, preprocess_data
from src.analyzer import LearningPathAnalyzer

# --- Фикстуры для тестовых данных ---
@pytest.fixture
def sample_lms_data():
    """Создает пример DataFrame с логами LMS для тестов."""
    data = {
        'user_id': [101, 101, 102, 102, 103, 103, 103],
        'timestamp': ['2023-10-01 09:00:00', '2023-10-01 14:00:00',
                      '2023-10-02 10:00:00', '2023-10-02 16:00:00',
                      '2023-10-03 11:00:00', '2023-10-03 12:00:00', '2023-10-03 15:00:00'],
        'event_type': ['login', 'quiz_attempted',
                       'login', 'assignment_submitted',
                       'login', 'forum_posted', 'assignment_submitted']
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_activity_data():
    """Создает пример данных активности для тестов корреляции."""
    activity_data = pd.DataFrame({
        'login': [5, 3, 8],
        'assignment_submitted': [2, 4, 6],
        'quiz_attempted': [3, 1, 4],
        'forum_posted': [1, 2, 5],
        'total_activity': [11, 10, 23]
    }, index=[101, 102, 103])
    return activity_data

# --- Тесты для data_loader.py ---
def test_load_lms_data(tmp_path):
    """Тест загрузки данных из CSV."""
    # Создаем временный CSV файл
    df = pd.DataFrame({'user_id': [1, 2], 'event_type': ['login', 'quiz']})
    file_path = tmp_path / "test.csv"
    df.to_csv(file_path, index=False)

    # Загружаем данные с помощью нашей функции
    loaded_df = load_lms_data(str(file_path))
    # Проверяем, что данные загрузились корректно
    assert not loaded_df.empty
    assert 'user_id' in loaded_df.columns
    assert len(loaded_df) == 2

def test_preprocess_data(sample_lms_data):
    """Тест предобработки данных."""
    processed_df = preprocess_data(sample_lms_data.copy())
    # Проверяем, что ненужные типы событий отфильтрованы (если были)
    # В нашем примере все типы правильные, поэтому размер не должен измениться
    assert len(processed_df) == len(sample_lms_data)
    # Проверяем наличие новых столбцов
    if 'timestamp' in sample_lms_data.columns:
        assert 'hour' in processed_df.columns
        assert 'day_of_week' in processed_df.columns

# --- Тесты для analyzer.py ---
def test_analyzer_initialization(sample_lms_data):
    """Тест инициализации класса LearningPathAnalyzer."""
    analyzer = LearningPathAnalyzer(sample_lms_data)
    assert analyzer.data is not None
    assert len(analyzer.data) == len(sample_lms_data)

def test_calculate_student_metrics(sample_lms_data):
    """Тест расчета метрик активности студентов."""
    analyzer = LearningPathAnalyzer(sample_lms_data)
    metrics_df = analyzer.calculate_student_metrics()

    # Проверяем, что метрики рассчитаны для уникальных студентов
    expected_users = set(sample_lms_data['user_id'])
    assert set(metrics_df.index) == expected_users
    # Проверяем наличие столбца с общей активностью
    assert 'total_activity' in metrics_df.columns

def test_correlate_activity_with_score(sample_activity_data):
    """Тест расчета корреляции."""
    analyzer = LearningPathAnalyzer(pd.DataFrame())  # Передаем пустой DF, т.к. используем предрассчитанную активность
    analyzer.student_activity = sample_activity_data  # Вручную задаем активность

    # Создаем тестовые оценки
    test_scores = pd.Series([85, 78, 92], index=[101, 102, 103])
    correlation = analyzer.correlate_activity_with_score(test_scores)

    # Проверяем, что корреляция рассчитана для всех типов активности
    assert len(correlation) == 4  # login, assignment_submitted, quiz_attempted, forum_posted
    # Проверяем, что значения корреляции находятся в допустимом диапазоне
    assert all(correlation.between(-1, 1))