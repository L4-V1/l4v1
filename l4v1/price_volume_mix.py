import polars as pl
import polars.selectors as cs
import plotly.graph_objects as go
from typing import Callable


def _is_invalid(expr: pl.Expr) -> pl.Expr:
    return expr.is_null() | expr.is_nan() | expr.is_infinite()


def _sanitize(expr: pl.Expr) -> pl.Expr:
    return pl.when(_is_invalid(expr)).then(0.0).otherwise(expr).fill_null(0.0)


class PVM:
    """Decomposes the change in an outcome metric into volume, rate, and mix effects.

    Compares two datasets (primary vs. comparison period) grouped by one or more
    dimensions, and attributes the outcome difference to:

    - **Volume effect** -- change driven by total volume shifting at the average rate.
    - **Rate effect** -- change driven by the rate (outcome per unit of volume) shifting.
    - **Mix effect** -- change driven by the composition of volume across groups.

    Args:
        df_primary: Primary (current) period dataset.
        df_comparison: Comparison (prior) period dataset.
        group_by_columns: Column(s) defining the decomposition groups.
        volume_column_name: Column representing volume (e.g. quantity, revenue).
        outcome_column_name: Column representing the outcome (e.g. revenue, profit).
    """

    _CMP_SUFFIX = "_cmp"

    def __init__(
        self,
        df_primary: pl.LazyFrame | pl.DataFrame,
        df_comparison: pl.LazyFrame | pl.DataFrame,
        group_by_columns: str | list[str],
        volume_column_name: str,
        outcome_column_name: str,
    ):
        self._df_primary = df_primary.lazy() if isinstance(df_primary, pl.DataFrame) else df_primary
        self._df_comparison = df_comparison.lazy() if isinstance(df_comparison, pl.DataFrame) else df_comparison
        self._group_cols = [group_by_columns] if isinstance(group_by_columns, str) else group_by_columns
        self._volume_col = volume_column_name
        self._outcome_col = outcome_column_name
        self._rate_col = "Rate" if volume_column_name[0].isupper() else "rate"

    # -- internal helpers -----------------------------------------------------

    def _group(self, df: pl.LazyFrame) -> pl.LazyFrame:
        keys = [pl.col(c).cast(pl.Utf8) for c in self._group_cols]
        aggs = [pl.col(c).sum().cast(pl.Float64) for c in (self._volume_col, self._outcome_col)]
        return df.group_by(keys).agg(aggs)

    def _join_key_expr(self) -> pl.Expr:
        parts = [
            pl.when(pl.col(k).is_null())
            .then(pl.col(f"{k}{self._CMP_SUFFIX}"))
            .otherwise(k)
            .str.to_titlecase()
            for k in self._group_cols
        ]
        return pl.concat_str(*parts, separator=r" \ ").alias("group_keys")

    def _metric_exprs(self) -> tuple[pl.Expr, ...]:
        """Build per-row display columns and volume/mix effect expressions.

        Rate effect is NOT included -- it is derived as a residual in
        ``get_table()`` to guarantee the identity:
            outcome_diff ≡ volume_effect + rate_effect + mix_effect
        """
        sfx = self._CMP_SUFFIX

        v = pl.col(self._volume_col)
        v0 = pl.col(f"{self._volume_col}{sfx}")
        o = pl.col(self._outcome_col)
        o0 = pl.col(f"{self._outcome_col}{sfx}")

        v_diff = (v.fill_null(0) - v0.fill_null(0)).alias(f"{self._volume_col}_diff")
        v_diff_pct = (v_diff / v0).alias(f"{self._volume_col}_diff_%")

        o_diff = (o.fill_null(0) - o0.fill_null(0)).alias(f"{self._outcome_col}_diff")
        o_diff_pct = (o_diff / o0).alias(f"{self._outcome_col}_diff_%")

        r = (o / v).alias(self._rate_col)
        r0 = (o0 / v0).alias(f"{self._rate_col}{sfx}")
        r_diff = (r.fill_null(0) - r0.fill_null(0)).alias(f"{self._rate_col}_diff")
        r_diff_pct = (r_diff / r0).alias(f"{self._rate_col}_diff_%")

        r_avg0 = pl.when(v0.sum() == 0).then(0.0).otherwise(o0.sum() / v0.sum())

        volume_effect = (v_diff * r_avg0).alias("volume_effect")
        mix_effect = (
            pl.when(_is_invalid(r0) & _is_invalid(r))
            .then(0.0)
            .when(_is_invalid(r0))
            .then((r - r_avg0) * v_diff)
            .otherwise((r0 - r_avg0) * v_diff)
        ).alias("mix_effect")

        return (
            v, v0, v_diff, v_diff_pct,
            r, r0, r_diff, r_diff_pct,
            o, o0, o_diff, o_diff_pct,
            volume_effect, mix_effect,
        )

    def _format_label(
        self, value: float, previous: float | None, fmt: Callable[[float], str],
    ) -> str:
        label = fmt(value)
        if previous is not None:
            delta = value - previous
            label = f"{label} ({'+' if delta >= 0 else ''}{fmt(delta)})"
        return label

    def _waterfall_data(
        self,
        table: pl.DataFrame,
        fmt: Callable[[float], str],
        primary_label: str,
        comparison_label: str,
        skip_zero: bool,
    ) -> tuple[list, list, list, list]:
        fmt = fmt or (lambda v: f"{v:,.0f}")
        primary_label = primary_label or self._outcome_col
        comparison_label = comparison_label or f"COMPARISON {self._outcome_col}"

        x, y, text, measure = [], [], [], []

        cmp_total = table.get_column(f"{self._outcome_col}{self._CMP_SUFFIX}").sum()
        x.append(f"<b>{comparison_label}</b>")
        y.append(cmp_total)
        text.append(f"<b>{fmt(cmp_total)}</b>")
        measure.append("absolute")

        running, prev = cmp_total, cmp_total
        keys = table.get_column("group_keys").unique().sort(descending=True)

        for impact in ("volume", "rate", "mix"):
            for key in keys:
                val = (
                    table.filter(pl.col("group_keys") == key)
                    .get_column(f"{impact}_effect")
                    .sum()
                )
                if skip_zero and val == 0:
                    continue
                x.append(f"{key} ({impact[0]}.)".lower())
                y.append(val)
                text.append(fmt(val))
                measure.append("relative")
                running += val

            x.append(f"<b>{impact.capitalize()} Impact Subtotal</b>")
            y.append(running)
            text.append(self._format_label(running, prev, fmt))
            measure.append("absolute")
            prev = running

        new_total = table.get_column(self._outcome_col).sum()
        x.append(f"<b>{primary_label}</b>")
        y.append(new_total)
        text.append(f"<b>{self._format_label(new_total, cmp_total, fmt)}</b>")
        measure.append("total")

        return x, y, text, measure

    # -- public API -----------------------------------------------------------

    def get_table(self) -> pl.DataFrame:
        """Compute the PVM decomposition table.

        Returns a DataFrame with one row per group containing volume, rate, and
        outcome metrics with diffs, plus the three effect columns whose sum
        equals ``outcome_diff`` by construction.
        """
        diff_col = f"{self._outcome_col}_diff"

        return (
            self._group(self._df_primary)
            .join(
                self._group(self._df_comparison),
                how="full",
                on=self._group_cols,
                suffix=self._CMP_SUFFIX,
            )
            .select(self._join_key_expr(), *self._metric_exprs())
            .with_columns(
                _sanitize(pl.col("volume_effect")).alias("volume_effect"),
                _sanitize(pl.col("mix_effect")).alias("mix_effect"),
                _sanitize(pl.col(diff_col)).alias(diff_col),
            )
            .with_columns(
                (pl.col(diff_col) - pl.col("volume_effect") - pl.col("mix_effect"))
                .alias("rate_effect")
            )
            .with_columns(
                pl.when(cs.float().is_nan() | cs.float().is_infinite())
                .then(0.0)
                .otherwise(cs.float())
                .name.keep()
                .fill_null(0.0)
            )
            .sort(diff_col, descending=True)
            .select(
                pl.all().exclude("volume_effect", "rate_effect", "mix_effect"),
                "volume_effect", "rate_effect", "mix_effect",
            )
            .collect()
        )

    def write_xlsx_table(self, file_path: str) -> None:
        """Write the decomposition table to Excel with conditional formatting."""
        self.get_table().write_excel(
            workbook=file_path,
            table_style="Table Style Light 1",
            conditional_formats={
                ("volume_effect", "rate_effect", "mix_effect"): {
                    "type": "3_color_scale",
                    "min_color": "#ff0000",
                    "mid_color": "#ffffff",
                    "max_color": "#73e656",
                }
            },
            column_formats={
                cs.matches("group_keys"): {"right": 2},
                cs.ends_with(self._volume_col, self._outcome_col, self._rate_col): {
                    "num_format": "#,##0",
                    "font_color": "black",
                },
                cs.ends_with(self._CMP_SUFFIX): {
                    "num_format": "#,##0",
                    "font_color": "gray",
                },
                cs.ends_with("_diff"): {"num_format": "#,##0", "font_color": "black"},
                cs.ends_with("%"): {"num_format": "0.0%", "right": 2},
                cs.ends_with("_effect"): {"num_format": "#,##0", "font_color": "black"},
            },
            freeze_panes=(1, 0),
            column_widths={
                "group_keys": 250,
                "volume_effect": 120,
                "rate_effect": 120,
                "mix_effect": 120,
            },
        )

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
        """Create a horizontal waterfall chart of the PVM decomposition.

        Args:
            primary_total_label: Label for the primary-period total bar.
            comparison_total_label: Label for the comparison-period total bar.
            format_data_labels: Python format string for numeric labels.
            skip_zero_change: Omit groups whose effect is zero.
            title: Chart title.
            color_increase / color_decrease / color_total: Bar colours.
            text_font_size: Font size for bar labels.
            plot_height / plot_width: Dimensions in px (height auto-scales if *None*).
            plotly_template: Plotly theme name.
            plotly_trace_settings: Extra kwargs forwarded to ``fig.update_traces()``.
            plotly_layout_settings: Extra kwargs forwarded to ``fig.update_layout()``.
        """
        x, y, text, measure = self._waterfall_data(
            self.get_table(),
            format_data_labels.format,
            primary_total_label,
            comparison_total_label,
            skip_zero_change,
        )

        fig = go.Figure(
            go.Waterfall(
                orientation="h",
                measure=measure,
                x=y,
                y=x,
                text=text,
                textposition="auto",
                textfont=dict(size=text_font_size),
                increasing=dict(marker=dict(color=color_increase)),
                decreasing=dict(marker=dict(color=color_decrease)),
                totals=dict(
                    marker=dict(color=color_total, line=dict(color="black", width=1))
                ),
            )
        )

        if plotly_trace_settings:
            fig.update_traces(plotly_trace_settings)

        layout = plotly_layout_settings or {
            "title": title,
            "height": plot_height or len(x) * 25 + 100,
            "width": plot_width,
            "template": plotly_template,
        }
        fig.update_layout(**layout)

        return fig
