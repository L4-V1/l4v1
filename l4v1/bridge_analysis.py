import polars as pl
import polars.selectors as cs


class Bridge:
    def __init__(
        self,
        df_primary: pl.LazyFrame | pl.DataFrame,
        df_comparison: pl.LazyFrame | pl.DataFrame,
        group_by_columns: str | list[str],
        volume_metric_name: str,
        outcome_metric_name: str,
    ):
        self.df_primary = (
            df_primary.lazy() if isinstance(df_primary, pl.DataFrame) else df_primary
        )
        self.df_comparison = (
            df_comparison.lazy()
            if isinstance(df_comparison, pl.DataFrame)
            else df_comparison
        )
        self.group_by_columns = (
            [group_by_columns]
            if isinstance(group_by_columns, str)
            else group_by_columns
        )
        self.volume_metric_name = volume_metric_name
        self.outcome_metric_name = outcome_metric_name
        self.comparison_suffix = "_cmp"

    def _group_dataframe(self, df: pl.LazyFrame) -> pl.LazyFrame:
        transformed_cols = [
            pl.col(col_name).cast(pl.Utf8).alias(col_name)
            for col_name in self.group_by_columns
        ]
        agg_expressions = [
            pl.col(metric).sum().cast(pl.Float64)
            for metric in [self.volume_metric_name, self.outcome_metric_name]
        ]
        return df.group_by(transformed_cols).agg(agg_expressions)

    def _get_join_key_expression(self) -> pl.Expr:
        group_keys = [
            (
                pl.when(pl.col(key).is_null())
                .then(pl.col(f"{key}{self.comparison_suffix}"))
                .otherwise(key)
            ).str.to_titlecase()
            for key in self.group_by_columns
        ]
        return pl.concat_str(*group_keys, separator=" / ").alias("group_keys")

    def _get_expressions(self) -> tuple[pl.Expr, ...]:
        # Volume
        volume_new = pl.col(self.volume_metric_name)
        volume_comparison = pl.col(f"{self.volume_metric_name}{self.comparison_suffix}")
        volume_diff = (volume_new - volume_comparison).alias(
            f"{self.volume_metric_name}_diff"
        )
        volume_diff_pct = (volume_diff / volume_comparison).alias(
            f"{self.volume_metric_name}_diff_%"
        )

        # Outcome
        outcome_new = pl.col(self.outcome_metric_name)
        outcome_comparison = pl.col(
            f"{self.outcome_metric_name}{self.comparison_suffix}"
        )
        outcome_diff = (outcome_new - outcome_comparison).alias(
            f"{self.outcome_metric_name}_diff"
        )
        outcome_diff_pct = (outcome_diff / outcome_comparison).alias(
            f"{self.outcome_metric_name}_diff_%"
        )

        # Rate
        rate_new = (outcome_new / volume_new).alias("rate")
        rate_comparison = (outcome_comparison / volume_comparison).alias(
            f"rate{self.comparison_suffix}"
        )
        rate_diff = (rate_new - rate_comparison).alias(f"rate_diff")
        rate_diff_pct = (rate_diff / rate_comparison).alias(f"rate_diff_%")
        rate_avg_comparison = outcome_comparison.sum() / volume_comparison.sum()

        # Expressions for the bridge
        rate = (rate_new - rate_comparison) * volume_new
        volume = volume_diff * rate_avg_comparison
        mix = (rate_comparison - rate_avg_comparison) * volume_diff

        def effect_expression(expr: pl.Expr, name: str) -> pl.Expr:
            return (
                pl.when((outcome_comparison.is_null()) | (outcome_new.is_null()))
                .then(pl.lit(0))
                .otherwise(expr)
            ).alias(f"{name}_effect")

        rate_expr = effect_expression(rate, "rate")
        volume_expr = effect_expression(volume, "volume")
        mix_expr = effect_expression(mix, "mix")

        new_expr = (
            pl.when((outcome_comparison.is_null()) & (outcome_new.is_not_null()))
            .then(outcome_new)
            .otherwise(pl.lit(0))
            .alias("new_effect")
        )
        old_expr = (
            pl.when((outcome_new.is_null()) & (outcome_comparison.is_not_null()))
            .then(outcome_comparison * -1)
            .otherwise(pl.lit(0))
            .alias("old_effect")
        )

        return (
            volume_new,
            volume_comparison,
            volume_diff,
            volume_diff_pct,
            rate_new,
            rate_comparison,
            rate_diff,
            rate_diff_pct,
            outcome_new,
            outcome_comparison,
            outcome_diff,
            outcome_diff_pct,
            volume_expr,
            rate_expr,
            mix_expr,
            new_expr,
            old_expr,
        )

    def get_effect_table(self) -> pl.DataFrame:
        df_primary_grouped = self._group_dataframe(self.df_primary)
        df_comparison_grouped = self._group_dataframe(self.df_comparison)

        join_key_expression = self._get_join_key_expression()

        effect_table = (
            df_primary_grouped.join(
                df_comparison_grouped,
                how="outer",
                on=self.group_by_columns,
                suffix=self.comparison_suffix,
            )
            .select(
                join_key_expression,
                *self._get_expressions(),
            )
            .with_columns(cs.numeric().fill_nan(0).fill_null(0))
            .sort(by=f"{self.outcome_metric_name}_diff", descending=True)
        )

        return effect_table.collect()

    def write_effect_table(self, file_path: str) -> None:
        effect_df = self.get_effect_table()
        effect_df.write_excel(
            workbook=file_path,
            table_style="Table Style Light 1",
            conditional_formats={
            ("volume_effect", "rate_effect", "mix_effect", "new_effect", "old_effect"): {
                "type": "3_color_scale",
                "min_color": "#ff0000",
                "mid_color": "#ffffff",
                "max_color": "#73e656",
                }
            },
            column_formats={
                cs.ends_with("%"): "0.0%",
                cs.ends_with("_effect"): "[Black]#,##0;[Black]-#,##0",
                (pl.col("group_keys") | cs.ends_with("%")): {"right": 2},
            },
            freeze_panes=(1,0),
            column_widths={
                "group_keys":250,
                "volume_effect": 100,
                "rate_effect": 100,
                "mix_effect": 100,
                "new_effect": 100,
                "old_effect": 100,
            }
        )