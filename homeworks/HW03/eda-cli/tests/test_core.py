from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2

# ТЕСТЫ ДЛЯ ТРЕХ НОВЫХ ЭВРИСТИК

def test_constant_columns_heuristic():
    """Тест для эвристики константных колонок."""
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'constant_col': [10, 10, 10, 10, 10],  # Все значения одинаковые
        'normal_col': [1, 2, 3, 4, 5],
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    assert flags['has_constant_columns'] == True
    assert flags['n_constant_columns'] == 1
    assert 'constant_col' in flags['constant_columns']


def test_no_constant_columns():
    """Тест, когда константных колонок нет."""
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4],
        'col2': ['a', 'b', 'c', 'd'],
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    assert flags['has_constant_columns'] == False
    assert flags['n_constant_columns'] == 0



    


def test_low_cardinality_categoricals():
    """Тест, когда нет высокой кардинальности."""
    df = pd.DataFrame({
        'id': range(10),
        'category': ['A', 'B'] * 5,  # 2 уникальных значения
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    assert flags['has_high_cardinality_categoricals'] == False
    assert flags['n_high_cardinality_columns'] == 0


def test_suspicious_id_duplicates():
    """Тест для эвристики дубликатов ID."""
    df = pd.DataFrame({
        'user_id': [1, 2, 3, 1, 2],  # Дубликаты
        'value': [10, 20, 30, 40, 50],
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    assert flags['has_suspicious_id_duplicates'] == True
    assert len(flags['suspicious_id_columns']) == 1
    assert flags['suspicious_id_columns'][0]['column'] == 'user_id'
    assert flags['suspicious_id_columns'][0]['duplicate_ratio'] > 0


def test_unique_id_no_duplicates():
    """Тест, когда ID уникальны."""
    df = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5],  # Все уникальны
        'data': [10, 20, 30, 40, 50],
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    assert flags['has_suspicious_id_duplicates'] == False
    assert len(flags['suspicious_id_columns']) == 0


def test_id_like_column_not_id():
    """Тест для колонки, похожей на ID, но не являющейся идентификатором."""
    df = pd.DataFrame({
        'guid_id_column': [100, 200, 300, 100, 200],  # Похоже на ID, но дублируется
        'value': [10, 20, 30, 40, 50],
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    assert flags['has_suspicious_id_duplicates'] == True
    assert len(flags['suspicious_id_columns']) == 1


