from l4v1.price_volume_mix import PVM
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Helpers for synthetic test data
# ---------------------------------------------------------------------------


def _make_pvm(primary: dict, comparison: dict) -> PVM:
    df_primary = pl.DataFrame(primary).lazy()
    df_comparison = pl.DataFrame(comparison).lazy()
    return PVM(
        df_primary=df_primary,
        df_comparison=df_comparison,
        group_by_columns="group",
        volume_column_name="volume",
        outcome_column_name="outcome",
    )


def _assert_effects_sum_to_diff(table: pl.DataFrame) -> None:
    total_diff = table.get_column("outcome_diff").sum()
    effect_sum = (
        table.select(
            pl.sum_horizontal(["volume_effect", "rate_effect", "mix_effect", "remainder_effect"])
        )
        .sum()
        .item()
    )
    assert total_diff == pytest.approx(effect_sum, abs=1e-9)


# ---------------------------------------------------------------------------
# Core decomposition tests
# ---------------------------------------------------------------------------


class TestBasicDecomposition:
    def test_identity_holds_simple(self):
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [100, 200], "outcome": [500, 800]},
            {"group": ["A", "B"], "volume": [80, 180], "outcome": [400, 720]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)

    def test_pure_volume_change(self):
        """Same rate across groups, only volume changes => rate and mix effects ~0."""
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [200, 400], "outcome": [1000, 2000]},
            {"group": ["A", "B"], "volume": [100, 200], "outcome": [500, 1000]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)
        assert table.get_column("rate_effect").sum() == pytest.approx(0, abs=1e-9)
        assert table.get_column("mix_effect").sum() == pytest.approx(0, abs=1e-9)

    def test_pure_rate_change(self):
        """Same volume, same mix, rate changes => volume and mix effects ~0."""
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [100, 200], "outcome": [600, 1200]},
            {"group": ["A", "B"], "volume": [100, 200], "outcome": [500, 1000]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)
        assert table.get_column("volume_effect").sum() == pytest.approx(0, abs=1e-9)
        assert table.get_column("mix_effect").sum() == pytest.approx(0, abs=1e-9)

    def test_pure_mix_change(self):
        """Total volume unchanged but composition shifts between groups with different rates."""
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [150, 150], "outcome": [750, 300]},
            {"group": ["A", "B"], "volume": [100, 200], "outcome": [500, 400]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)
        assert table.get_column("volume_effect").sum() == pytest.approx(0, abs=1e-9)

    def test_single_group(self):
        """With one group only, mix effect should be zero."""
        pvm = _make_pvm(
            {"group": ["A"], "volume": [150], "outcome": [900]},
            {"group": ["A"], "volume": [100], "outcome": [500]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)
        assert table.get_column("mix_effect").sum() == pytest.approx(0, abs=1e-9)


# ---------------------------------------------------------------------------
# Edge cases: groups present in only one period
# ---------------------------------------------------------------------------


class TestMissingGroups:
    def test_group_only_in_primary(self):
        """A new group appears in primary but not in comparison."""
        pvm = _make_pvm(
            {"group": ["A", "B", "C"], "volume": [100, 200, 50], "outcome": [500, 800, 300]},
            {"group": ["A", "B"], "volume": [80, 180], "outcome": [400, 720]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)
        assert table.height == 3

    def test_group_only_in_comparison(self):
        """A group disappears in the primary period."""
        pvm = _make_pvm(
            {"group": ["A"], "volume": [100], "outcome": [500]},
            {"group": ["A", "B"], "volume": [80, 180], "outcome": [400, 720]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)
        assert table.height == 2

    def test_completely_disjoint_groups(self):
        """No overlap at all between primary and comparison."""
        pvm = _make_pvm(
            {"group": ["X", "Y"], "volume": [100, 200], "outcome": [500, 800]},
            {"group": ["A", "B"], "volume": [80, 180], "outcome": [400, 720]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)
        assert table.height == 4


# ---------------------------------------------------------------------------
# Edge cases: zero division scenarios
# ---------------------------------------------------------------------------


class TestZeroDivision:
    def test_zero_volume_in_comparison(self):
        """Rate = outcome/volume would divide by zero in comparison period."""
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [100, 200], "outcome": [500, 800]},
            {"group": ["A", "B"], "volume": [0, 180], "outcome": [0, 720]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)
        assert not table.select(pl.col("volume_effect").is_nan().any()).item()
        assert not table.select(pl.col("rate_effect").is_nan().any()).item()
        assert not table.select(pl.col("mix_effect").is_nan().any()).item()

    def test_zero_volume_in_primary(self):
        """Rate = outcome/volume would divide by zero in primary period."""
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [0, 200], "outcome": [0, 800]},
            {"group": ["A", "B"], "volume": [80, 180], "outcome": [400, 720]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)
        assert not table.select(pl.col("volume_effect").is_nan().any()).item()
        assert not table.select(pl.col("rate_effect").is_nan().any()).item()
        assert not table.select(pl.col("mix_effect").is_nan().any()).item()

    def test_zero_volume_both_periods(self):
        """Both periods have zero volume for a group."""
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [0, 200], "outcome": [0, 800]},
            {"group": ["A", "B"], "volume": [0, 180], "outcome": [0, 720]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)
        assert not table.select(pl.col("volume_effect").is_nan().any()).item()
        assert not table.select(pl.col("rate_effect").is_nan().any()).item()
        assert not table.select(pl.col("mix_effect").is_nan().any()).item()

    def test_all_zero_volume_in_comparison(self):
        """Entire comparison period has zero volume (r_avg0 denominator = 0)."""
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [100, 200], "outcome": [500, 800]},
            {"group": ["A", "B"], "volume": [0, 0], "outcome": [0, 0]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)
        assert not table.select(pl.col("volume_effect").is_nan().any()).item()
        assert not table.select(pl.col("rate_effect").is_nan().any()).item()
        assert not table.select(pl.col("mix_effect").is_nan().any()).item()

    def test_all_zero_everywhere(self):
        """Both periods are entirely zero."""
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [0, 0], "outcome": [0, 0]},
            {"group": ["A", "B"], "volume": [0, 0], "outcome": [0, 0]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)
        assert table.get_column("volume_effect").sum() == 0
        assert table.get_column("rate_effect").sum() == 0
        assert table.get_column("mix_effect").sum() == 0


# ---------------------------------------------------------------------------
# Edge cases: zero volume with non-zero outcome
# ---------------------------------------------------------------------------


class TestZeroVolumeNonZeroOutcome:
    """When volume=0 but outcome≠0, the rate becomes undefined (outcome/0 = inf).
    The sum of effects must still equal outcome_diff for every row."""

    def test_zero_volume_nonzero_outcome_both_periods(self):
        """Both periods have a group with zero volume but non-zero outcome."""
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [0, 200], "outcome": [500, 800]},
            {"group": ["A", "B"], "volume": [0, 180], "outcome": [300, 720]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)

    def test_zero_volume_nonzero_outcome_primary_only(self):
        """Primary period has a group with zero volume but non-zero outcome."""
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [0, 200], "outcome": [500, 800]},
            {"group": ["A", "B"], "volume": [100, 180], "outcome": [400, 720]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)

    def test_zero_volume_nonzero_outcome_comparison_only(self):
        """Comparison period has a group with zero volume but non-zero outcome."""
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [100, 200], "outcome": [500, 800]},
            {"group": ["A", "B"], "volume": [0, 180], "outcome": [300, 720]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)

    def test_multiple_zero_volume_groups_with_outcome(self):
        """Multiple groups have zero volume but non-zero outcome in both periods."""
        pvm = _make_pvm(
            {"group": ["A", "B", "C"], "volume": [0, 0, 200], "outcome": [500, 300, 800]},
            {"group": ["A", "B", "C"], "volume": [0, 0, 180], "outcome": [400, 250, 720]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)

    def test_all_groups_zero_volume_nonzero_outcome(self):
        """Every group has zero volume but non-zero outcome."""
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [0, 0], "outcome": [500, 800]},
            {"group": ["A", "B"], "volume": [0, 0], "outcome": [300, 600]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)


# ---------------------------------------------------------------------------
# Edge cases: non-zero volume with zero outcome
# ---------------------------------------------------------------------------


class TestNonZeroVolumeZeroOutcome:
    """When volume>0 but outcome=0, the rate = outcome/volume = 0. Mathematically
    valid but a degenerate case worth verifying."""

    def test_zero_outcome_both_periods(self):
        """A group has volume in both periods but zero outcome in both."""
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [100, 200], "outcome": [0, 800]},
            {"group": ["A", "B"], "volume": [80, 180], "outcome": [0, 720]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)

    def test_zero_outcome_primary_only(self):
        """Outcome drops to zero in primary while volume remains."""
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [100, 200], "outcome": [0, 800]},
            {"group": ["A", "B"], "volume": [80, 180], "outcome": [400, 720]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)

    def test_zero_outcome_comparison_only(self):
        """Outcome was zero in comparison, becomes non-zero in primary."""
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [100, 200], "outcome": [500, 800]},
            {"group": ["A", "B"], "volume": [80, 180], "outcome": [0, 720]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)

    def test_all_groups_zero_outcome(self):
        """All groups have volume but zero outcome in both periods."""
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [100, 200], "outcome": [0, 0]},
            {"group": ["A", "B"], "volume": [80, 180], "outcome": [0, 0]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)
        assert table.get_column("volume_effect").sum() == 0
        assert table.get_column("rate_effect").sum() == 0
        assert table.get_column("mix_effect").sum() == 0

    def test_mixed_zero_volume_and_zero_outcome(self):
        """One group has volume but no outcome, another has outcome but no volume."""
        pvm = _make_pvm(
            {"group": ["A", "B", "C"], "volume": [100, 0, 200], "outcome": [0, 500, 800]},
            {"group": ["A", "B", "C"], "volume": [80, 0, 180], "outcome": [0, 300, 720]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)

    def test_outcome_disappears_volume_remains(self):
        """Group had non-zero outcome in comparison, drops to zero in primary while volume persists."""
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [120, 200], "outcome": [0, 800]},
            {"group": ["A", "B"], "volume": [100, 180], "outcome": [600, 720]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)


# ---------------------------------------------------------------------------
# Edge cases: null and special values
# ---------------------------------------------------------------------------


class TestNullsAndSpecialValues:
    def test_null_volume_in_primary(self):
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [None, 200], "outcome": [None, 800]},
            {"group": ["A", "B"], "volume": [80, 180], "outcome": [400, 720]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)
        assert not table.select(pl.col("volume_effect").is_null().any()).item()
        assert not table.select(pl.col("rate_effect").is_null().any()).item()
        assert not table.select(pl.col("mix_effect").is_null().any()).item()

    def test_null_volume_in_comparison(self):
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [100, 200], "outcome": [500, 800]},
            {"group": ["A", "B"], "volume": [None, 180], "outcome": [None, 720]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)
        assert not table.select(pl.col("volume_effect").is_null().any()).item()

    def test_negative_volumes(self):
        """Negative volumes should still decompose correctly."""
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [-50, 200], "outcome": [-250, 800]},
            {"group": ["A", "B"], "volume": [80, 180], "outcome": [400, 720]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)

    def test_negative_outcome(self):
        """Negative outcomes should decompose correctly."""
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [100, 200], "outcome": [-100, 800]},
            {"group": ["A", "B"], "volume": [80, 180], "outcome": [400, -200]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)


# ---------------------------------------------------------------------------
# Edge cases: large number of groups / duplicates within input
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_duplicate_rows_get_aggregated(self):
        """Multiple rows with the same group key should be summed before decomposition."""
        pvm = _make_pvm(
            {"group": ["A", "A", "B"], "volume": [50, 50, 200], "outcome": [250, 250, 800]},
            {"group": ["A", "B", "B"], "volume": [80, 90, 90], "outcome": [400, 360, 360]},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)
        assert table.height == 2

    def test_many_groups(self):
        """Stress test with many groups, identity should still hold."""
        n = 50
        groups = [f"G{i}" for i in range(n)]
        pvm = _make_pvm(
            {"group": groups, "volume": list(range(10, 10 + n)), "outcome": list(range(100, 100 + n))},
            {"group": groups, "volume": list(range(5, 5 + n)), "outcome": list(range(80, 80 + n))},
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)
        assert table.height == n


# ---------------------------------------------------------------------------
# Edge cases: DataFrame vs LazyFrame inputs
# ---------------------------------------------------------------------------


class TestInputTypes:
    def test_dataframe_inputs(self):
        """PVM should accept eager DataFrames as well as LazyFrames."""
        df_primary = pl.DataFrame(
            {"group": ["A", "B"], "volume": [100, 200], "outcome": [500, 800]}
        )
        df_comparison = pl.DataFrame(
            {"group": ["A", "B"], "volume": [80, 180], "outcome": [400, 720]}
        )
        pvm = PVM(
            df_primary=df_primary,
            df_comparison=df_comparison,
            group_by_columns="group",
            volume_column_name="volume",
            outcome_column_name="outcome",
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)

    def test_lazyframe_inputs(self):
        """Explicit LazyFrame inputs."""
        df_primary = pl.DataFrame(
            {"group": ["A", "B"], "volume": [100, 200], "outcome": [500, 800]}
        ).lazy()
        df_comparison = pl.DataFrame(
            {"group": ["A", "B"], "volume": [80, 180], "outcome": [400, 720]}
        ).lazy()
        pvm = PVM(
            df_primary=df_primary,
            df_comparison=df_comparison,
            group_by_columns="group",
            volume_column_name="volume",
            outcome_column_name="outcome",
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)


# ---------------------------------------------------------------------------
# Multi-column group_by
# ---------------------------------------------------------------------------


class TestMultiColumnGroupBy:
    def test_two_group_columns(self):
        pvm = PVM(
            df_primary=pl.DataFrame({
                "region": ["North", "North", "South"],
                "product": ["X", "Y", "X"],
                "volume": [100, 200, 150],
                "outcome": [500, 900, 600],
            }).lazy(),
            df_comparison=pl.DataFrame({
                "region": ["North", "North", "South"],
                "product": ["X", "Y", "X"],
                "volume": [80, 180, 120],
                "outcome": [400, 720, 480],
            }).lazy(),
            group_by_columns=["region", "product"],
            volume_column_name="volume",
            outcome_column_name="outcome",
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)
        assert table.height == 3

    def test_multi_column_partial_overlap(self):
        """Some group combinations only exist in one period."""
        pvm = PVM(
            df_primary=pl.DataFrame({
                "region": ["North", "North", "South"],
                "product": ["X", "Y", "Z"],
                "volume": [100, 200, 150],
                "outcome": [500, 900, 600],
            }).lazy(),
            df_comparison=pl.DataFrame({
                "region": ["North", "South", "South"],
                "product": ["X", "X", "Y"],
                "volume": [80, 120, 60],
                "outcome": [400, 480, 180],
            }).lazy(),
            group_by_columns=["region", "product"],
            volume_column_name="volume",
            outcome_column_name="outcome",
        )
        table = pvm.get_table()
        _assert_effects_sum_to_diff(table)


# ---------------------------------------------------------------------------
# Output structure validation
# ---------------------------------------------------------------------------


class TestOutputStructure:
    def test_no_nan_in_output(self):
        """Final table should never contain NaN values."""
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [100, 0], "outcome": [500, 0]},
            {"group": ["A", "B"], "volume": [0, 180], "outcome": [0, 720]},
        )
        table = pvm.get_table()
        for col in table.columns:
            if table[col].dtype.is_float():
                assert not table.select(pl.col(col).is_nan().any()).item(), (
                    f"Column {col} contains NaN"
                )

    def test_no_nulls_in_output(self):
        """Final table should never contain null values in numeric columns."""
        pvm = _make_pvm(
            {"group": ["A", "B", "C"], "volume": [100, 200, 50], "outcome": [500, 800, 300]},
            {"group": ["B", "C", "D"], "volume": [180, 60, 90], "outcome": [720, 240, 450]},
        )
        table = pvm.get_table()
        for col in table.columns:
            if table[col].dtype.is_float():
                assert not table.select(pl.col(col).is_null().any()).item(), (
                    f"Column {col} contains null"
                )

    def test_no_inf_in_output(self):
        """Final table should never contain infinite values."""
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [100, 200], "outcome": [500, 800]},
            {"group": ["A", "B"], "volume": [0, 0], "outcome": [0, 0]},
        )
        table = pvm.get_table()
        for col in table.columns:
            if table[col].dtype.is_float():
                assert not table.select(pl.col(col).is_infinite().any()).item(), (
                    f"Column {col} contains infinity"
                )

    def test_expected_columns_present(self):
        pvm = _make_pvm(
            {"group": ["A", "B"], "volume": [100, 200], "outcome": [500, 800]},
            {"group": ["A", "B"], "volume": [80, 180], "outcome": [400, 720]},
        )
        table = pvm.get_table()
        expected = {"volume_effect", "rate_effect", "mix_effect", "remainder_effect", "outcome_diff", "group_keys"}
        assert expected.issubset(set(table.columns))
