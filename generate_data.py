"""Generate a synthetic year of point-of-sale data for a fictional coffee chain.

One row per line item (a transaction can hold several items). Built-in patterns:
  * stores open 09:00-17:00, closed New Year's Day, Thanksgiving and Christmas
  * morning rush and lunch peak; urban stores busier on weekdays, suburban on weekends
  * mild traffic seasonality, a slight growth trend across the year, daily noise
  * seasonal menu items (iced drinks in summer, pumpkin spice in fall, etc.)
  * food follows meal times (breakfast sandwiches in the morning, paninis at lunch)
  * a store that opens partway through the year (Sacramento, March 2025)
  * customer type and payment method with realistic correlations

Usage:
    python generate_data.py                 # writes data/sales.parquet
    python generate_data.py --csv           # also writes data/sales.csv
    python generate_data.py --seed 7 --year 2024
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OPEN_HOUR, CLOSE_HOUR = 9, 17  # 9am-5pm; last sale before 17:00
HOURS = np.arange(OPEN_HOUR, CLOSE_HOUR)

# --------------------------------------------------------------------------- stores
# traffic: relative daily transaction volume; kind drives the weekday/weekend mix.
STORES = pd.DataFrame(
    [
        ("S01", "Seattle Downtown", "Seattle", "Seattle Metro", "urban", 1.40, "2020-01-01"),
        ("S02", "Capitol Hill", "Seattle", "Seattle Metro", "urban", 1.10, "2020-01-01"),
        ("S03", "Bellevue Square", "Bellevue", "Seattle Metro", "suburban", 1.00, "2020-01-01"),
        ("S04", "Pearl District", "Portland", "Portland Metro", "urban", 1.00, "2020-01-01"),
        ("S05", "Lake Oswego", "Lake Oswego", "Portland Metro", "suburban", 0.75, "2020-01-01"),
        ("S06", "SF Financial District", "San Francisco", "Northern California", "urban", 1.30, "2020-01-01"),
        ("S07", "Rockridge", "Oakland", "Northern California", "suburban", 0.85, "2020-01-01"),
        ("S08", "Sacramento Midtown", "Sacramento", "Northern California", "suburban", 0.90, "{year}-03-10"),
    ],
    columns=["store_id", "store_name", "city", "region", "store_type", "traffic", "open_date"],
)

# -------------------------------------------------------------------------- products
# season: None = all year, "summer", "winter", or an explicit (start, end) month-day window.
# daypart: None, "morning" or "lunch" - shifts demand within the day.
PRODUCTS = pd.DataFrame(
    [
        # id, name, category, price, unit_cost, popularity, season, daypart
        ("P01", "Drip Coffee", "Coffee", 2.95, 0.45, 10.0, None, "morning"),
        ("P02", "Americano", "Coffee", 3.45, 0.55, 6.0, None, None),
        ("P03", "Latte", "Coffee", 4.95, 1.10, 10.0, None, None),
        ("P04", "Cappuccino", "Coffee", 4.75, 1.00, 5.0, None, "morning"),
        ("P05", "Mocha", "Coffee", 5.25, 1.30, 4.0, None, None),
        ("P06", "Cold Brew", "Coffee", 4.45, 0.80, 5.0, "summer", None),
        ("P07", "Iced Latte", "Coffee", 5.25, 1.20, 5.0, "summer", None),
        ("P08", "Chai Latte", "Tea", 4.95, 1.05, 4.0, "winter", None),
        ("P09", "Green Tea", "Tea", 3.25, 0.50, 2.5, None, None),
        ("P10", "Iced Tea", "Tea", 3.45, 0.45, 3.0, "summer", None),
        ("P11", "Pumpkin Spice Latte", "Seasonal", 5.95, 1.40, 9.0, ("08-26", "11-30"), None),
        ("P12", "Peppermint Mocha", "Seasonal", 5.95, 1.45, 9.0, ("11-06", "12-31"), None),
        ("P13", "Lavender Oat Latte", "Seasonal", 5.75, 1.35, 6.0, ("03-15", "05-31"), None),
        ("P14", "Hot Chocolate", "Seasonal", 3.95, 0.85, 3.0, "winter", None),
        ("P15", "Butter Croissant", "Bakery", 3.75, 1.10, 5.0, None, "morning"),
        ("P16", "Blueberry Muffin", "Bakery", 3.50, 0.95, 4.0, None, "morning"),
        ("P17", "Chocolate Chip Cookie", "Bakery", 2.75, 0.60, 4.0, None, None),
        ("P18", "Cinnamon Roll", "Bakery", 4.25, 1.20, 3.0, None, "morning"),
        ("P19", "Breakfast Sandwich", "Food", 6.95, 2.40, 4.0, None, "morning"),
        ("P20", "Avocado Toast", "Food", 8.50, 3.10, 2.5, None, None),
        ("P21", "Turkey Pesto Panini", "Food", 9.95, 3.80, 3.5, None, "lunch"),
        ("P22", "Greek Salad", "Food", 10.50, 4.10, 2.5, "summer", "lunch"),
        ("P23", "Whole Bean Bag (12oz)", "Retail", 16.95, 7.50, 0.8, None, None),
        ("P24", "Travel Mug", "Retail", 24.00, 9.00, 0.4, None, None),
    ],
    columns=["product_id", "product_name", "category", "unit_price", "unit_cost", "popularity", "season", "daypart"],
)

CUSTOMER_TYPES = np.array(["Rewards Member", "Returning", "New"])
PAYMENT_METHODS = np.array(["Credit/Debit Card", "Mobile App", "Mobile Wallet", "Cash", "Gift Card"])


# ------------------------------------------------------------------ demand factors
def season_factor(season, day: pd.Timestamp) -> float:
    """Multiplier on a product's popularity for a given day."""
    if season is None:
        return 1.0
    doy = day.dayofyear
    # smooth annual cycle: +1 at mid-July, -1 at mid-January
    summer = -np.cos(2 * np.pi * (doy - 15) / 365.25)
    if season == "summer":
        return float(np.clip(1.0 + 0.9 * summer, 0.1, None))
    if season == "winter":
        return float(np.clip(1.0 - 0.8 * summer, 0.1, None))
    start, end = season
    md = day.strftime("%m-%d")
    return 1.0 if start <= md <= end else 0.0


def daypart_factor(daypart, hour: int) -> float:
    if daypart == "morning":
        return {9: 2.2, 10: 1.8, 11: 1.0}.get(hour, 0.45)
    if daypart == "lunch":
        return {11: 1.6, 12: 2.6, 13: 2.0}.get(hour, 0.35)
    return 1.0


def hourly_weights(store_type: str, weekend: bool) -> np.ndarray:
    """Relative transaction volume for each open hour (9..16)."""
    if weekend:
        w = np.array([0.9, 1.35, 1.4, 1.3, 1.1, 0.9, 0.75, 0.6])
    elif store_type == "urban":
        w = np.array([1.9, 1.35, 0.95, 1.4, 1.1, 0.7, 0.65, 0.5])
    else:
        w = np.array([1.5, 1.3, 1.0, 1.25, 1.05, 0.8, 0.75, 0.6])
    return w / w.sum()


def day_traffic(store, day: pd.Timestamp, year_start: pd.Timestamp) -> float:
    weekend = day.dayofweek >= 5
    if store.store_type == "urban":
        dow = 0.6 if weekend else 1.1
    else:
        dow = 1.3 if weekend else 0.9
    dow *= 1.08 if day.dayofweek == 4 else 1.0  # Friday bump
    doy = day.dayofyear
    seasonal = 1.0 + 0.06 * np.sin(2 * np.pi * (doy - 80) / 365.25)  # soft spring/summer high
    if day.month == 1:
        seasonal *= 0.9  # January slump
    if day.month == 12 and day.day < 24:
        seasonal *= 1.12  # holiday shopping
    trend = 1.0 + 0.08 * (day - year_start).days / 365  # ~8% growth over the year
    # new store ramps up over its first ~10 weeks
    days_open = (day - store.open_date).days
    ramp = min(1.0, 0.45 + 0.55 * days_open / 70) if days_open < 70 else 1.0
    return store.traffic * dow * seasonal * trend * ramp


def closed_days(year: int) -> set[pd.Timestamp]:
    nov = pd.date_range(f"{year}-11-01", f"{year}-11-30")
    thanksgiving = nov[nov.dayofweek == 3][3]
    return {pd.Timestamp(f"{year}-01-01"), thanksgiving, pd.Timestamp(f"{year}-12-25")}


# ------------------------------------------------------------------------ generator
def generate(year: int = 2025, seed: int = 42, base_transactions: float = 20.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stores = STORES.assign(open_date=pd.to_datetime(STORES.open_date.map(lambda s: s.format(year=year))))
    year_start = pd.Timestamp(f"{year}-01-01")
    days = pd.date_range(year_start, f"{year}-12-31")
    closed = closed_days(year)

    n_products = len(PRODUCTS)
    retail_idx = PRODUCTS.index[PRODUCTS.category == "Retail"].to_numpy()
    iced_idx = PRODUCTS.index[PRODUCTS.product_name.str.contains("Iced|Cold", regex=True)].to_numpy()

    chunks = []
    txn_counter = 0
    for day in days:
        if day in closed:
            continue
        weekend = day.dayofweek >= 5
        # product weights for each open hour on this day: (hours, products)
        base = np.array([p * season_factor(s, day) for p, s in zip(PRODUCTS.popularity, PRODUCTS.season)])
        hour_w = np.array([[base[j] * daypart_factor(PRODUCTS.daypart[j], h) for j in range(n_products)] for h in HOURS])
        if day.month == 12 and day.day < 24:
            hour_w[:, retail_idx] *= 3.0  # gift season
        noise = rng.lognormal(0, 0.12)  # weather / local events, shared across stores that day

        for store in stores.itertuples(index=False):
            if day < store.open_date:
                continue
            lam = base_transactions * day_traffic(store, day, year_start) * noise * rng.lognormal(0, 0.08)
            n_txn = rng.poisson(lam)
            if n_txn == 0:
                continue

            # timestamps
            hours = rng.choice(HOURS, size=n_txn, p=hourly_weights(store.store_type, weekend))
            seconds = rng.integers(0, 3600, size=n_txn)
            ts = day + pd.to_timedelta(hours * 3600 + seconds, unit="s")

            # customers & payment
            cust = rng.choice(CUSTOMER_TYPES, size=n_txn, p=[0.42, 0.38, 0.20])
            pay_p = np.where(
                (cust == "Rewards Member")[:, None],
                [0.25, 0.55, 0.12, 0.04, 0.04],
                [0.50, 0.06, 0.29, 0.07, 0.05] if store.store_type == "urban" else [0.54, 0.06, 0.13, 0.20, 0.05],
            )
            if day.month == 1:  # holiday gift cards get redeemed
                pay_p[:, 4] *= 3
            pay_cdf = np.cumsum(pay_p / pay_p.sum(axis=1, keepdims=True), axis=1)
            pay = PAYMENT_METHODS[(pay_cdf < rng.random((n_txn, 1))).sum(axis=1).clip(max=len(PAYMENT_METHODS) - 1)]

            # basket size: rewards members buy a bit more
            n_items = np.where(
                cust == "Rewards Member",
                rng.choice([1, 2, 3, 4], size=n_txn, p=[0.40, 0.38, 0.17, 0.05]),
                rng.choice([1, 2, 3, 4], size=n_txn, p=[0.58, 0.30, 0.10, 0.02]),
            )

            # line items
            line_txn = np.repeat(np.arange(n_txn), n_items)
            line_hour_idx = hours[line_txn] - OPEN_HOUR
            w = hour_w[line_hour_idx].copy()
            if store.city == "Sacramento":
                w[:, iced_idx] *= 1.6  # hotter climate
            cdf = np.cumsum(w, axis=1)
            u = rng.random(len(line_txn)) * cdf[:, -1]
            prod = (cdf < u[:, None]).sum(axis=1)
            qty = np.where(
                np.isin(prod, retail_idx),
                1,
                rng.choice([1, 2, 3], size=len(prod), p=[0.86, 0.11, 0.03]),
            )

            chunks.append(
                pd.DataFrame(
                    {
                        "transaction_id": txn_counter + line_txn,
                        "timestamp": ts[line_txn],
                        "store_id": store.store_id,
                        "product_idx": prod,
                        "quantity": qty,
                        "customer_type": cust[line_txn],
                        "payment_method": pay[line_txn],
                    }
                )
            )
            txn_counter += n_txn

    df = pd.concat(chunks, ignore_index=True)
    prod_cols = PRODUCTS[["product_id", "product_name", "category", "unit_price", "unit_cost"]]
    df = df.join(prod_cols, on="product_idx").drop(columns="product_idx")
    df = df.merge(stores[["store_id", "store_name", "city", "region", "store_type"]], on="store_id")
    df["revenue"] = (df.quantity * df.unit_price).round(2)
    df["cost"] = (df.quantity * df.unit_cost).round(2)
    df["profit"] = (df.revenue - df.cost).round(2)
    df["transaction_id"] = "T" + df.transaction_id.astype(str).str.zfill(7)

    for col in ["store_id", "store_name", "city", "region", "store_type", "product_id",
                "product_name", "category", "customer_type", "payment_method"]:
        df[col] = df[col].astype("category")

    cols = ["transaction_id", "timestamp", "store_id", "store_name", "city", "region", "store_type",
            "product_id", "product_name", "category", "quantity", "unit_price", "unit_cost",
            "revenue", "cost", "profit", "customer_type", "payment_method"]
    return df[cols].sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/sales.parquet"))
    parser.add_argument("--csv", action="store_true", help="also write a CSV next to the parquet file")
    args = parser.parse_args()

    df = generate(year=args.year, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    if args.csv:
        df.to_csv(args.out.with_suffix(".csv"), index=False)

    print(f"Wrote {len(df):,} line items / {df.transaction_id.nunique():,} transactions to {args.out}")
    print(f"Revenue ${df.revenue.sum():,.0f} | profit ${df.profit.sum():,.0f} "
          f"({df.profit.sum() / df.revenue.sum():.1%} margin)")


if __name__ == "__main__":
    main()
