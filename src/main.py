import pandas as pd
import os
import sys

# Добавляем папку src в путь, чтобы импортировать наши модули
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_lms_data, preprocess_data
from src.analyzer import LearningPathAnalyzer
from src.visualizer import plot_activity_correlation, plot_student_activity_distribution

def main():
    """
    Основная функция для запуска анализа пути обучения.
    """
    print("=== Learning Path Analyzer ===")

    # 1. Загрузка данных
    data_path = "data/sample_logs.csv"  # Укажи здесь путь к своим данным
    if not os.path.exists(data_path):
        print(f"Файл данных не найден: {data_path}")
        print("Пожалуйста, создай файл data/sample_logs.csv")
        return

    df = load_lms_data(data_path)
    df = preprocess_data(df)

    # 2. Анализ
    analyzer = LearningPathAnalyzer(df)
    student_metrics = analyzer.calculate_student_metrics()

    # 3. Пример данных для оценок (в реальном проекте загружаются из файла)
    # Создаем "искусственные" оценки для демонстрации
    # Предполагаем, что оценка линейно зависит от общей активности + случайный шум
    np.random.seed(42)  # Для воспроизводимости
    simulated_scores = (student_metrics['total_activity'] * 0.5 + np.random.randn(len(student_metrics)) * 10)
    simulated_scores = simulated_scores.clip(lower=60, upper=100)  # Ограничиваем диапазон
    scores_series = pd.Series(simulated_scores.values, index=student_metrics.index)

    correlation = analyzer.correlate_activity_with_score(scores_series)
    print("\nКорреляция активности с оценками:")
    print(correlation.round(3))

    # 4. Визуализация
    print("\nСоздание графиков...")
    # Сохраняем график корреляции в папку docs
    os.makedirs("docs", exist_ok=True)
    plot_activity_correlation(correlation, save_path="docs/activity_correlation.png")

    # Показываем распределение активности (график откроется в новом окне)
    plot_student_activity_distribution(student_metrics, top_n=8)

    # 5. Вывод рекомендаций (базовый пример)
    most_correlated = correlation.index[0]
    print(f"\n[РЕКОМЕНДАЦИЯ] Активность '{most_correlated}' имеет наибольшую положительную корреляцию с успеваемостью.")
    print("Рекомендуется поощрять данный тип активности среди студентов.")

if __name__ == "__main__":
    main()