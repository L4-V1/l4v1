import polars as pl
import polars.selectors as cs
import plotly.graph_objects as go
from typing import Callable


class PVM:
    """
    Provides tools to calculate and visualize the impacts of price, volume, and mix changes between two datasets.

    This class takes two Polars data- or lazyframes having the same schema, one for the current period and one
    for a comparison period, along with metadata about the datasets to perform grouping, calculations, and
    visualizations on the price, volume, and mix effects.

    Parameters:
        df_primary (pl.LazyFrame | pl.DataFrame): The primary dataset for the current period.
        df_comparison (pl.LazyFrame | pl.DataFrame): The comparison dataset for the period.
        group_by_columns (str | List[str]): Column name(s) to use for grouping.
        volume_column_name (str): The name of the column representing volume data, e.g quantities.
        outcome_column_name (str): The name of the column representing outcome data, e.g. revenue or cost.

    Methods:
        get_table(): Computes a table summarizing the changes per group as well as the effects of price, volume, and mix changes.
        write_xlsx_table(file_path): Writes the impact table to an Excel file with heatmap applied for the price, volume, mix effects.
        waterfall_plot(): Generates a waterfall plot visualizing the effects from the computed table.
    """

    def __init__(
        self,
        df_primary: pl.LazyFrame | pl.DataFrame,
        df_comparison: pl.LazyFrame | pl.DataFrame,
        group_by_columns: str | list[str],
        volume_column_name: str,
        outcome_column_name: str,
    ):
        self._df_primary = (
            df_primary.lazy() if isinstance(df_primary, pl.DataFrame) else df_primary
        )
        self._df_comparison = (
            df_comparison.lazy()
            if isinstance(df_comparison, pl.DataFrame)
            else df_comparison
        )
        self._group_by_columns = (
            [group_by_columns]
            if isinstance(group_by_columns, str)
            else group_by_columns
        )
        self._volume_column_name = volume_column_name
        self._outcome_column_name = outcome_column_name
        self._rate_metric_name = "Rate" if volume_column_name[0].isupper() else "rate"
        self._comparison_suffix = "_cmp"

    def _group_dataframe(self, df: pl.LazyFrame) -> pl.LazyFrame:
        transformed_cols = [
            pl.col(col_name).cast(pl.Utf8).alias(col_name)
            for col_name in self._group_by_columns
        ]
        agg_expressions = [
            pl.col(metric).sum().cast(pl.Float64)
            for metric in [self._volume_column_name, self._outcome_column_name]
        ]
        return df.group_by(transformed_cols).agg(agg_expressions)

    def _get_join_key_expression(self) -> pl.Expr:
        group_keys = [
            (
                pl.when(pl.col(key).is_null())
                .then(pl.col(f"{key}{self._comparison_suffix}"))
                .otherwise(key)
            ).str.to_titlecase()
            for key in self._group_by_columns
        ]
        return pl.concat_str(*group_keys, separator=r" \ ").alias("group_keys")

    def _get_expressions(self) -> tuple[pl.Expr, ...]:
        # Volume
        volume_new = pl.col(self._volume_column_name)
        volume_comparison = pl.col(
            f"{self._volume_column_name}{self._comparison_suffix}"
        )
        volume_diff = (volume_new.fill_null(0) - volume_comparison.fill_null(0)).alias(
            f"{self._volume_column_name}_diff"
        )
        volume_diff_pct = (volume_diff / volume_comparison).alias(
            f"{self._volume_column_name}_diff_%"
        )

        # Outcome
        outcome_new = pl.col(self._outcome_column_name)
        outcome_comparison = pl.col(
            f"{self._outcome_column_name}{self._comparison_suffix}"
        )
        outcome_diff = (
            outcome_new.fill_null(0) - outcome_comparison.fill_null(0)
        ).alias(f"{self._outcome_column_name}_diff")
        outcome_diff_pct = (outcome_diff / outcome_comparison).alias(
            f"{self._outcome_column_name}_diff_%"
        )

        # Rate
        rate_new = (outcome_new / volume_new).alias(f"{self._rate_metric_name}")
        rate_comparison = (outcome_comparison / volume_comparison).alias(
            f"{self._rate_metric_name}{self._comparison_suffix}"
        )
        rate_diff = (rate_new.fill_null(0) - rate_comparison.fill_null(0)).alias(
            f"{self._rate_metric_name}_diff"
        )
        rate_diff_pct = (rate_diff / rate_comparison).alias(
            f"{self._rate_metric_name}_diff_%"
        )
        rate_avg_comparison = outcome_comparison.sum() / volume_comparison.sum()

        # Expressions for the bridge
        rate_effect = ((rate_new - rate_comparison) * volume_new).alias("rate_effect")
        volume_effect = (volume_diff * rate_avg_comparison).alias("volume_effect")
        mix_effect = (
            pl.when(rate_comparison.is_null() | rate_comparison.is_nan())
            .then((rate_new - rate_avg_comparison) * volume_diff)
            .otherwise((rate_comparison - rate_avg_comparison) * volume_diff)
        ).alias("mix_effect")

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
            volume_effect,
            rate_effect,
            mix_effect,
        )

    def get_table(self) -> pl.DataFrame:
        """
        Generates a summary table for changes together with price, volume, and mix effects.

        Returns:
            pl.DataFrame: A dataframe containing grouped and calculated results showing the summary of changes as well as impacts of price, volume, and mix changes.
        """
        df_primary_grouped = self._group_dataframe(self._df_primary)
        df_comparison_grouped = self._group_dataframe(self._df_comparison)

        join_key_expression = self._get_join_key_expression()

        effect_table = (
            df_primary_grouped.join(
                df_comparison_grouped,
                how="outer",
                on=self._group_by_columns,
                suffix=self._comparison_suffix,
            )
            .select(
                join_key_expression,
                *self._get_expressions(),
            )
            .with_columns(cs.numeric().fill_nan(0).fill_null(0))
            .sort(by=f"{self._outcome_column_name}_diff", descending=True)
        )

        return effect_table.collect()

    def write_xlsx_table(self, file_path: str) -> None:
        """
        Writes the impact analysis results to an Excel file with heatmap applied for the price, volume, and mix effect columns.

        This method takes the table generated by get_table(), applies conditional formatting, and writes it to the specified Excel file path.

        Parameters:
            file_path (str): The path to the Excel file where the table will be saved.
        """
        effect_df = self.get_table()
        effect_df.write_excel(
            workbook=file_path,
            table_style="Table Style Light 1",
            conditional_formats={
                (
                    "volume_effect",
                    "rate_effect",
                    "mix_effect",
                    # "new_effect",
                    # "old_effect",
                ): {
                    "type": "3_color_scale",
                    "min_color": "#ff0000",
                    "mid_color": "#ffffff",
                    "max_color": "#73e656",
                }
            },
            column_formats={
                cs.matches("group_keys"): {"right": 2},
                cs.ends_with(
                    self._volume_column_name,
                    self._outcome_column_name,
                    self._rate_metric_name,
                ): {
                    "num_format": "#,##0",
                    "font_color": "black",
                },
                cs.ends_with(f"{self._comparison_suffix}"): {
                    "num_format": "#,##0",
                    "font_color": "gray",
                },
                cs.ends_with("_diff"): {
                    "num_format": "#,##0",
                    "font_color": "black",
                },
                cs.ends_with("%"): {"num_format": "0.0%", "right": 2},
                cs.ends_with("_effect"): {
                    "num_format": "#,##0",
                    "font_color": "black",
                },
            },
            freeze_panes=(1, 0),
            column_widths={
                "group_keys": 250,
                "volume_effect": 100,
                "rate_effect": 100,
                "mix_effect": 100,
                # "new_effect": 100,
                # "old_effect": 100,
            },
        )

    def _create_data_label(
        self, value: float, previous_value: float, format_func: Callable
    ) -> str:
        formatted_value = format_func(value)
        if previous_value is not None:
            growth = value - previous_value
            sign = "+" if growth >= 0 else ""
            formatted_growth = f"{sign}{format_func(growth)}"
            return f"{formatted_value} ({formatted_growth})"
        return formatted_value

    def _prep_data_for_waterfall_plot(
        self,
        impact_table: pl.DataFrame,
        format_data_labels: Callable,
        primary_total_label: str,
        comparison_total_label: str,
        skip_zero_change: bool,
    ) -> tuple[list, list, list, list]:
        if format_data_labels is None:
            format_data_labels = lambda value: f"{value:,.0f}"
        primary_total_label = primary_total_label or self._outcome_column_name
        comparison_total_label = (
            comparison_total_label or f"COMPARISON {self._outcome_column_name}"
        )

        x_labels, y_values, data_labels, measure_list = [], [], [], []
        outcome_comparison = impact_table.get_column(
            f"{self._outcome_column_name}{self._comparison_suffix}"
        ).sum()

        x_labels.append(f"<b>{comparison_total_label}</b>")
        y_values.append(outcome_comparison)
        data_labels.append(f"<b>{format_data_labels(outcome_comparison)}</b>")
        measure_list.append("absolute")

        cumulative_sum = outcome_comparison
        previous_value = outcome_comparison

        impact_types = ["volume", "rate", "mix"]

        for impact_type in impact_types:
            for key in (
                impact_table.get_column("group_keys").unique().sort(descending=True)
            ):
                impact_value = (
                    impact_table.filter(pl.col("group_keys") == key)
                    .get_column(f"{impact_type}_effect")
                    .sum()
                )
                if not (skip_zero_change and impact_value == 0):
                    x_labels.append(f"{key} ({impact_type[0]}.)".lower())
                    y_values.append(impact_value)
                    data_labels.append(format_data_labels(impact_value))
                    measure_list.append("relative")
                    cumulative_sum += impact_value

            x_labels.append(f"<b>{impact_type.capitalize()} Impact Subtotal</b>")
            y_values.append(cumulative_sum)
            data_labels.append(
                self._create_data_label(
                    cumulative_sum, previous_value, format_data_labels
                )
            )
            measure_list.append("absolute")
            previous_value = cumulative_sum

        outcome_new = impact_table.get_column(self._outcome_column_name).sum()
        x_labels.append(f"<b>{primary_total_label}</b>")
        y_values.append(outcome_new)
        data_labels.append(
            f"<b>{self._create_data_label(outcome_new, outcome_comparison, format_data_labels)}</b>"
        )
        measure_list.append("total")

        return x_labels, y_values, data_labels, measure_list

    def waterfall_plot(
        self,
        primary_total_label: str = None,
        comparison_total_label: str = None,
        format_data_labels: str = "{:,.0f}",
        skip_zero_change: bool = True,
        title: str = None,
        color_increase: str = "#00AF00",
        color_decrease: str = "#FF0000",
        color_total: str = "#F1F1F1",
        text_font_size: int = 8,
        plot_height: int = None,
        plot_width: int = 750,
        plotly_template: str = "plotly_white",
        plotly_trace_settings: dict[str, any] = None,
        plotly_layout_settings: dict[str, any] = None,
    ) -> go.Figure:
        """
        Creates a waterfall plot visualizing the price, volume, and mix impacts.

        This visualization helps in understanding how different factors have contributed to the overall change.

        Parameters:
            primary_total_label (str, optional): The label for the primary dataset's total in the plot.
            comparison_total_label (str, optional): The label for the comparison dataset's total in the plot.
            format_data_labels (str, optional): The format string for numbers in the plot.
            skip_zero_change (bool, optional): Whether to skip zero changes in the plot visualization.
            title (str, optional): Title of the plot.
            color_increase (str, optional): Color for positive changes.
            color_decrease (str, optional): Color for negative changes.
            color_total (str, optional): Color for the total columns.
            text_font_size (int, optional): Font size of the text in the plot.
            plot_height (int, optional): The height of the plot.
            plot_width (int, optional): The width of the plot.
            plotly_template (str, optional): The Plotly template for styling the plot.
            plotly_trace_settings (dict[str, any], optional): Additional settings for Plotly traces.
            plotly_layout_settings (dict[str, any], optional): Additional settings for the Plotly layout.

        Returns:
            go.Figure: A Plotly Figure object representing the waterfall plot of the impacts.
        """
        # Convert format string to a formatting function
        formatter = lambda x: format_data_labels.format(x)

        # Prepare data for plotting
        x_labels, y_values, data_labels, measure_list = (
            self._prep_data_for_waterfall_plot(
                self.get_table(),
                formatter,
                primary_total_label,
                comparison_total_label,
                skip_zero_change,
            )
        )

        # Create the plot
        fig = go.Figure(
            go.Waterfall(
                orientation="h",
                measure=measure_list,
                x=y_values,
                y=x_labels,
                text=data_labels,
                textposition="auto",
                textfont=dict(size=text_font_size),
                increasing=dict(marker=dict(color=color_increase)),
                decreasing=dict(marker=dict(color=color_decrease)),
                totals=dict(
                    marker=dict(color=color_total, line=dict(color="black", width=1))
                ),
            )
        )

        # Update layout with basic settings
        layout_params = {
            "title": title,
            "height": plot_height if plot_height else len(x_labels) * 25 + 100,
            "width": plot_width,
            "template": plotly_template,
        }

        # Apply advanced settings if provided
        if plotly_trace_settings:
            fig.update_traces(plotly_trace_settings)
        if plotly_layout_settings:
            fig.update_layout(**plotly_layout_settings)
        else:
            fig.update_layout(**layout_params)

        return fig
