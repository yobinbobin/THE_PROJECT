import pandas as pd
import numpy as np
from typing import Tuple, Dict

class LearningPathAnalyzer:
    """
    Анализатор путей обучения на основе логов LMS.
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.student_activity = None

    def calculate_student_metrics(self) -> pd.DataFrame:
        """
        Считает метрики активности для каждого студента.
        """
        if 'user_id' not in self.data.columns or 'event_type' not in self.data.columns:
            raise ValueError("Данные должны содержать 'user_id' и 'event_type'")

        # Группируем данные по студентам и типу активности
        activity_counts = self.data.groupby(['user_id', 'event_type']).size().unstack(fill_value=0)
        # Добавляем общую активность
        activity_counts['total_activity'] = activity_counts.sum(axis=1)

        self.student_activity = activity_counts
        print(f"Рассчитаны метрики для {len(activity_counts)} студентов.")
        return activity_counts

    def correlate_activity_with_score(self, scores: pd.Series) -> pd.DataFrame:
        """
        Считает корреляцию между активностью и оценками.
        scores: Series, где индекс - user_id, значение - оценка.
        """
        if self.student_activity is None:
            self.calculate_student_metrics()

        # Объединяем активность с оценками
        merged_data = self.student_activity.join(scores.rename('final_score'), how='inner')
        # Считаем корреляцию Пирсона
        correlation = merged_data.corr()['final_score'].drop('final_score').sort_values(ascending=False)

        print("Корреляция активности с оценками рассчитана.")
        return correlation