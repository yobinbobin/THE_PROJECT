def main():
    """
    Основная функция для запуска анализа пути обучения.
    """
    print("=== Learning Path Analyzer ===")

    # 1. Загрузка данных
    data_path = "data/sample_logs.csv"  # Укажи здесь путь к своим данным
    if not os.path.exists(data_path):
        print(f"Файл данных не найден: {data_path}")
        print("Используем демонстрационные данные...")
        
        # Создаем демонстрационные данные, если файла нет
        df = create_demo_data()
    else:
        df = load_lms_data(data_path)
    
    df = preprocess_data(df)

    # 2. Анализ
    analyzer = LearningPathAnalyzer(df)
    student_metrics = analyzer.calculate_student_metrics()

    # 3. Используем реальные оценки из данных, если есть столбец final_score
    if 'final_score' in df.columns:
        # Берем среднюю оценку для каждого студента
        scores = df.groupby('user_id')['final_score'].mean()
        scores_series = pd.Series(scores.values, index=student_metrics.index)
        print("Используем реальные оценки из данных.")
    else:
        # Создаем "искусственные" оценки для демонстрации
        np.random.seed(42)  # Для воспроизводимости
        simulated_scores = (student_metrics['total_activity'] * 0.5 + np.random.randn(len(student_metrics)) * 10)
        simulated_scores = simulated_scores.clip(lower=60, upper=100)  # Ограничиваем диапазон
        scores_series = pd.Series(simulated_scores.values, index=student_metrics.index)
        print("Используем сгенерированные оценки для демонстрации.")

    correlation = analyzer.correlate_activity_with_score(scores_series)
    print("\nКорреляция активности с оценками:")
    print(correlation.round(3))

    # 4. Визуализация
    print("\nСоздание графиков...")
    # Сохраняем график корреляции в папку docs
    os.makedirs("docs", exist_ok=True)
    plot_activity_correlation(correlation, save_path="docs/activity_correlation.png")

    # 5. Вывод рекомендаций (базовый пример)
    if not correlation.empty:
        most_correlated = correlation.index[0]
        print(f"\n[РЕКОМЕНДАЦИЯ] Активность '{most_correlated}' имеет наибольшую положительную корреляцию с успеваемостью.")
        print("Рекомендуется поощрять данный тип активности среди студентов.")
    
    return df, correlation


def create_demo_data():
    """Создает демонстрационные данные, если файл не найден."""
    import pandas as pd
    import numpy as np
    
    print("Создание демонстрационных данных...")
    
    np.random.seed(42)
    n_students = 50
    n_events = 200
    
    user_ids = np.random.randint(100, 200, n_events)
    event_types = np.random.choice(['login', 'assignment_submitted', 'quiz_attempted', 'forum_posted'], 
                                   n_events, p=[0.4, 0.3, 0.2, 0.1])
    
    dates = pd.date_range(start='2023-09-01', end='2023-10-31', periods=n_events)
    
    df = pd.DataFrame({
        'user_id': user_ids,
        'timestamp': dates,
        'event_type': event_types
    })
    
    print(f"Создано {len(df)} демонстрационных записей для {df['user_id'].nunique()} студентов.")
    return df