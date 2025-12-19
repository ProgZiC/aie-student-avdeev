from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Строк: 36
Столбцов: 14

Колонки:
                    name   dtype  non_null  missing  missing_share  unique  is_numeric    min    max        mean         std
                 user_id   int64        36        
0       0.000000      35        True 1001.0 1035.0 1018.194444   10.166667
                 country  object        36        
0       0.000000       4       False    NaN    NaN         NaN         NaN
                    city  object        34        
2       0.055556      16       False    NaN    NaN         NaN         NaN
                  device  object        36        
0       0.000000       3       False    NaN    NaN         NaN         NaN
                 channel  object        36        
0       0.000000       4       False    NaN    NaN         NaN         NaN
       sessions_last_30d   int64        36        
0       0.000000      26        True    0.0   34.0   11.944444    8.608781
avg_session_duration_min float64        34        
2       0.055556      32        True    2.0   15.2    7.247059    3.473382
       pages_per_session float64        36        
0       0.000000      32        True    1.0    7.5    4.100000    1.560586
      purchases_last_30d   int64        36        
0       0.000000       5        True    0.0    4.0    1.138889    1.125110
        revenue_last_30d float64        36        
0       0.000000      23        True    0.0 7000.0 1575.013889 1815.280578
                 churned   int64        36        
0       0.000000       2        True    0.0    1.0    0.333333    0.478091
             signup_year   int64        36        0       0.000000       7        True 2018.0 2024.0 2020.972222    1.521017
                    plan  object        36        0       0.000000       3       False    NaN    NaN         NaN         NaN
       n_support_tickets   int64        36        0       0.000000       6        True    0.0    5.0    1.083333    1.204159
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(
        6, 
        help="Максимум числовых колонок для гистограмм."
    ),
    top_k_categories: int = typer.Option(
        5,
        help="Сколько top-значений выводить для категориальных признаков."
    ),
    min_missing_share: float = typer.Option(
        0.3,
        help="Порог доли пропусков, выше которого колонка считается проблемной."
    ),
) -> None:
    """
    Сгенерировать полный EDA-отчёт:
    - текстовый overview и summary по колонкам (CSV/Markdown);
    - статистика пропусков;
    - корреляционная матрица;
    - top-k категорий по категориальным признакам;
    - картинки: гистограммы, матрица пропусков, heatmap корреляции.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # 1. Обзор
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df)

    # 2. Качество в целом
    quality_flags = compute_quality_flags(summary, missing_df)

    # Находим проблемные колонки по пропускам
    problematic_columns = []
    for col in summary.columns:
        if col.missing_share > min_missing_share:
            problematic_columns.append({
                "name": col.name,
                "missing_share": col.missing_share,
                "missing_count": col.missing,
            })

    # 3. Сохраняем табличные артефакты
    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    # 4. Markdown-отчёт
    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# EDA-отчёт\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Оценка качества: **{quality_flags['quality_score']:.2f}**\n")
        f.write(f"- Макс. доля пропусков по колонке: **{quality_flags['max_missing_share']:.2%}**\n")
        f.write(f"- Слишком мало строк: **{quality_flags['too_few_rows']}**\n")
        f.write(f"- Слишком много колонок: **{quality_flags['too_many_columns']}**\n")
        f.write(f"- Слишком много пропусков: **{quality_flags['too_many_missing']}**\n\n")
# Новые эвристики
        f.write("### Новые эвристики качества\n\n")
        
        f.write(f"1. **Константные колонки**: {quality_flags['has_constant_columns']}\n")
        if quality_flags['has_constant_columns']:
            f.write(f"   - Колонки: {', '.join(quality_flags['constant_columns'])}\n")
            f.write(f"   - Количество: {quality_flags['n_constant_columns']}\n")
        
        f.write(f"\n2. **Высокая кардинальность категориальных признаков** (>50 уникальных значений): "
                f"{quality_flags['has_high_cardinality_categoricals']}\n")
        if quality_flags['has_high_cardinality_categoricals']:
            f.write(f"   - Колонки: {', '.join(quality_flags['high_cardinality_columns'])}\n")
            f.write(f"   - Количество: {quality_flags['n_high_cardinality_columns']}\n")
        
        f.write(f"\n3. **Подозрительные дубликаты ID**: {quality_flags['has_suspicious_id_duplicates']}\n")
        if quality_flags['has_suspicious_id_duplicates']:
            for id_col in quality_flags['suspicious_id_columns']:
                f.write(f"   - {id_col['description']} (дубликатов: {id_col['duplicate_ratio']:.1%})\n")

        # Раздел с проблемными колонками по пропускам
        if problematic_columns:
            f.write(f"\n## Колонки с пропусками > {min_missing_share:.0%}\n\n")
            f.write("| Колонка | Пропуски | Доля пропусков |\n")
            f.write("|---------|----------|----------------|\n")
            for pc in problematic_columns:
                f.write(f"| {pc['name']} | {pc['missing_count']} | {pc['missing_share']:.2%} |\n")
            f.write("\n")

        f.write("## Колонки\n\n")
        f.write("См. файл `summary.csv`.\n\n")

        f.write("## Пропуски\n\n")
        if missing_df.empty:
            f.write("Пропусков нет или датасет пуст.\n\n")
        else:
            f.write("См. файлы `missing.csv` и `missing_matrix.png`.\n\n")

        f.write("## Корреляция числовых признаков\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок для корреляции.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

        f.write("## Категориальные признаки\n\n")
        if not top_cats:
            f.write("Категориальные/строковые признаки не найдены.\n\n")
        else:
            f.write(f"Топ-{top_k_categories} значений по категориальным признакам см. в папке `top_categories/`.\n\n")

        f.write("## Гистограммы числовых колонок\n\n")
        f.write(f"Построено гистограмм для первых {max_hist_columns} числовых колонок.\n")
        f.write("См. файлы `hist_*.png`.\n")
        
    # 5. Картинки
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"- Основной markdown: {md_path}")
    typer.echo("- Табличные файлы: summary.csv, missing.csv, correlation.csv, top_categories/*.csv")
    typer.echo("- Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png")
    typer.echo(f"\nНовые эвристики качества данных:")
    typer.echo(f"- Константные колонки: {quality_flags['has_constant_columns']}")
    typer.echo(f"- Высокая кардинальность категориальных: {quality_flags['has_high_cardinality_categoricals']}")
    typer.echo(f"- Подозрительные дубликаты ID: {quality_flags['has_suspicious_id_duplicates']}")



if __name__ == "__main__":
    app()
