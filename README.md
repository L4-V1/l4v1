# l4v1
l4v1 is a Python library designed to simplify data analytics tasks through data manipulation and visualization techniques. Built on top of Polars and Plotly, it offers a straightforward API for quickly creating detailed summaries. This project is a work in progress, and more functionality will be added in the future.

## Installation
You can install the `l4v1` package directly from PyPI:

```bash
pip install l4v1
```

## Usage
Calculate price, volume, and mix effects conveniently, and visualize them in an Excel heatmap or a Plotly waterfall chart.

Start by importing Polars to load the data:

```python
import polars as pl

# Load your datasets
sales_week_1 = pl.read_csv("data/sales_week1.csv")
sales_week_2 = pl.read_csv("data/sales_week2.csv")
```

Once you have the data, import the PVM module:
```python
from l4v1.price_volume_mix import PVM

# Initialize the class with your data and desired dimensions
pvm = PVM(
    df_primary=sales_week_2, # Data to analyse
    df_comparison=sales_week_1, # Data to compare against
    group_by_columns=["Product line", "Customer type"] # Dimension(s) to use
    volume_column_name="Quantity", # Column name containing volume (e.g. quantity)
    outcome_column_name="Total", # Column name containing outcome (e.g. revenue or cost)
)
```
Once the class is initialized, you can decide whether to create an Excel table, a Plotly waterfall chart, or continue working with the data in a Polars DataFrame.

To create a waterfall chart:
```python
pvm.waterfall_plot(
    primary_total_label="Week 2 Sales", # Optional label
    comparison_total_label="Week 1 Sales", # Optional label
    title="Sales Week 2 vs 1", # Optional title,
    color_total = "#F1F1F1", # Optional color for totals
    # etc.
)
```
![Waterfall Plot Example](docs/readme/example_waterfall.png)

In Excel, it is easier to visualize if there are many dimensions used:
```python
pvm.write_xlsx_table("your/path/file_name.xlsx") # Must end to .xlsx file extensions
```
![Heatmap Example](docs/readme/example_excel_heatmap.png)

For large datasets, it might be most convenient to continue exploring directly in Polars. In that case, simply call get_table:
```python
pvm.get_table()
```
