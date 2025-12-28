import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_activity_correlation(correlation_series: pd.Series, save_path: str = None):
    """
    Создает столбчатую диаграмму корреляции активности с оценками.
    """
    plt.figure(figsize=(10, 6))
    bars = plt.bar(correlation_series.index, correlation_series.values, color='skyblue')
    plt.title('Корреляция типов активности с итоговой оценкой', fontsize=14)
    plt.xlabel('Тип активности в LMS')
    plt.ylabel('Коэффициент корреляции (Пирсон)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Добавляем значения на столбцы
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"График сохранен: {save_path}")
    else:
        plt.show()

def plot_student_activity_distribution(activity_df: pd.DataFrame, top_n: int = 10):
    """
    Визуализирует распределение активности для топ-N студентов.
    """
    # Сортируем студентов по общей активности
    top_students = activity_df.nlargest(top_n, 'total_activity')

    # Переводим данные в формат для stacked bar chart
    plot_data = top_students.drop(columns=['total_activity']).transpose()

    plt.figure(figsize=(12, 6))
    plot_data.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Set2')
    plt.title(f'Распределение типов активности для топ-{top_n} студентов', fontsize=14)
    plt.xlabel('Тип активности')
    plt.ylabel('Количество событий')
    plt.legend(title='ID студента', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()