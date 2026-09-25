# Brewline Coffee Co. — Sales Dashboard Demo

A synthetic sales dataset for a fictional 8-store coffee chain, plus an interactive
[Streamlit](https://streamlit.io) + Plotly dashboard that summarizes it.

## Quick start

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows PowerShell
py -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app.py          # opens http://localhost:8501
```

The dataset is already committed at `data/sales.parquet`. To regenerate it (or make a
different variant):

```bash
python generate_data.py                  # same data (seed 42, year 2025)
python generate_data.py --seed 7 --csv   # new random draw, also writes data/sales.csv
```

## The dataset

~102k line items / ~60k transactions across calendar year 2025. Stores are open 9am–5pm
and closed on New Year's Day, Thanksgiving and Christmas.

| Column | Description |
|---|---|
| `transaction_id` | Groups the line items that belong to one sale |
| `timestamp` | Time of the sale (09:00–16:59) |
| `store_id`, `store_name`, `city`, `region`, `store_type` | 8 stores in 3 regions; `urban` or `suburban` |
| `product_id`, `product_name`, `category` | 24 products in 6 categories (Coffee, Tea, Seasonal, Bakery, Food, Retail) |
| `quantity`, `unit_price`, `unit_cost` | Line quantity and per-unit price and cost |
| `revenue`, `cost`, `profit` | `quantity × price`, `quantity × cost`, and the difference |
| `customer_type` | Rewards Member / Returning / New |
| `payment_method` | Credit/Debit Card, Mobile App, Mobile Wallet, Cash, Gift Card |

Patterns built into the data, so the dashboard has real stories to show:

- **Time of day:** a morning rush at 9am and a lunch peak at noon; urban stores peak on weekday mornings, while suburban stores are busiest on weekends.
- **Seasonality:** iced drinks and salads in summer; chai and hot chocolate in winter; limited-time drinks (Lavender Oat Latte in spring, Pumpkin Spice in fall, Peppermint Mocha during the holidays); retail gifts in December; a January slump; and about 8% growth over the year.
- **Dayparts:** breakfast sandwiches and pastries sell in the morning, and paninis and salads at lunch.
- **New store:** Fondren opens on March 10 and ramps up over about 10 weeks. Its hotter climate means more iced drinks.
- **Customers:** rewards members buy bigger baskets and mostly pay in the app. Cash is more common at suburban stores, mobile wallets at urban stores, and gift-card use jumps in January.

Margins count product cost (COGS) only, so they are gross margins (~74%) with no labour or rent.

## The dashboard

- **Filters** (date range, region, store, category) sit in one row and apply to every chart.
- **KPI tiles:** revenue, gross profit, margin, transactions, average ticket and items sold. When the selected range leaves room for it, each tile shows the change vs. the previous period of the same length.
- **Overview:** daily revenue with a 7-day average, revenue by category and by region, and the monthly category mix.
- **Time patterns:** a weekday × hour heatmap, transactions per hour (weekday vs. weekend), and revenue by weekday.
- **Products:** revenue by product, a product scorecard, and weekly units for the seasonal menu.
- **Stores:** revenue by store, a store scorecard (including rewards-member share), and a store × month heatmap.
- **Customers & payments:** revenue and average ticket by customer type, and payment mix by customer type and by store type.
- **Raw data:** the filtered rows, with a CSV download.

Every chart has a **Show data** expander with the underlying table.

## Files

```
generate_data.py      synthetic data generator (numpy/pandas, seeded)
app.py                Streamlit dashboard
data/sales.parquet    generated dataset (~1.8 MB)
.streamlit/config.toml  light theme matching the chart palette
```
