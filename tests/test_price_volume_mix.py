from l4v1.price_volume_mix import PVM
import polars as pl
import pytest


def create_test_datasets() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    df = pl.scan_parquet("data/supermarket_sales.parquet")
    df_1 = df.filter(pl.col("Datetime").dt.month() == 1)
    df_2 = df.filter(pl.col("Datetime").dt.month() == 2)
    return df_1, df_2


def test_pvm_get_table():
    df_1, df_2 = create_test_datasets()

    pvm = PVM(
        df_primary=df_2,
        df_comparison=df_1,
        group_by_columns=["Customer type", "Product line"],
        volume_column_name="Quantity",
        outcome_column_name="Total",
    )

    summary_table = pvm.get_table()

    total_diff = summary_table.get_column("Total_diff").sum()
    effect_sum = (
        summary_table.select(
            pl.sum_horizontal(
                [
                    "volume_effect",
                    "rate_effect",
                    "mix_effect",
                    # "new_effect",
                    # "old_effect",
                ]
            )
        )
        .sum()
        .item()
    )

    assert total_diff == pytest.approx(effect_sum)
