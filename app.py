"""Streamlit dashboard for the synthetic Local Grind Coffee Co. sales dataset.

Run:  streamlit run app.py
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

DATA_PATH = Path(__file__).parent / "data" / "sales.parquet"

# ------------------------------------------------------------------ design tokens
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
# categorical slots, always assigned in this order (validated for colour-vision deficiency)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# one-hue green ramp built around slot 6 (#008300); OKLab lightness falls monotonically 0.93 -> 0.30
SEQ_GREEN = ["#d6f0d3", "#aedfa8", "#7fc977", "#4cad44", "#1f9119", "#007400", "#005400", "#003a00"]

CATEGORY_ORDER = ["Coffee", "Food", "Bakery", "Seasonal", "Tea", "Retail"]
REGION_ORDER = ["Memphis Metro", "Nashville Metro", "Mississippi"]
CUSTOMER_ORDER = ["Rewards Member", "Returning", "New"]
PAYMENT_ORDER = ["Credit/Debit Card", "Mobile App", "Mobile Wallet", "Cash", "Gift Card"]
DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# colour follows the entity, never its rank - filtering never repaints survivors
CATEGORY_COLORS = dict(zip(CATEGORY_ORDER, SLOTS))
REGION_COLORS = dict(zip(REGION_ORDER, SLOTS))
CUSTOMER_COLORS = dict(zip(CUSTOMER_ORDER, SLOTS))
PAYMENT_COLORS = dict(zip(PAYMENT_ORDER, SLOTS))

pio.templates["brew"] = go.layout.Template(
    layout=dict(
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK_2, size=13),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=SLOTS,
        margin=dict(l=8, r=8, t=36, b=8),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
        xaxis=dict(gridcolor=GRID, linecolor=BASELINE, zeroline=False, tickfont=dict(color=MUTED), title_font=dict(color=MUTED)),
        yaxis=dict(gridcolor=GRID, linecolor=BASELINE, zeroline=False, tickfont=dict(color=MUTED), title_font=dict(color=MUTED)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title_text="", font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=INK)),
        bargap=0.25,
        bargroupgap=0.08,
    ),
    data=dict(bar=[go.Bar(marker=dict(cornerradius=4, line=dict(width=0)))]),
)
pio.templates.default = "brew"

st.set_page_config(page_title="Local Grind Coffee Co. — Sales Dashboard", page_icon="☕", layout="wide")

st.markdown(
    """
    <style>
    @media (max-width: 1000px) {
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
            flex: 1 1 280px;
            min-width: min(100%, 280px);
        }
    }
    div.st-key-kpi-grid div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        flex: 1 1 calc(33.333% - 16px);
        min-width: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading sales data…")
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        from generate_data import generate

        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate().to_parquet(DATA_PATH, index=False)
    df = pd.read_parquet(DATA_PATH)
    df["date"] = df.timestamp.dt.normalize()
    df["hour"] = df.timestamp.dt.hour
    df["dow"] = pd.Categorical(df.timestamp.dt.strftime("%a"), categories=DOW_ORDER, ordered=True)
    df["month"] = df.timestamp.dt.to_period("M").dt.to_timestamp()
    return df


def summarize(frame: pd.DataFrame) -> dict[str, float]:
    revenue = frame.revenue.sum()
    txns = frame.transaction_id.nunique()
    return {
        "revenue": revenue,
        "profit": frame.profit.sum(),
        "margin": frame.profit.sum() / revenue if revenue else 0.0,
        "transactions": txns,
        "avg_ticket": revenue / txns if txns else 0.0,
        "items": frame.quantity.sum(),
    }


def group_kpis(frame: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """Revenue, profit, transactions, items and derived ratios for any grouping."""
    out = frame.groupby(by, observed=True).agg(
        revenue=("revenue", "sum"),
        profit=("profit", "sum"),
        transactions=("transaction_id", "nunique"),
        items=("quantity", "sum"),
    )
    out["margin"] = out.profit / out.revenue
    out["avg_ticket"] = out.revenue / out.transactions
    return out.reset_index()


# ------------------------------------------------------------------------ helpers
def show(fig: go.Figure, table: pd.DataFrame | None = None, height: int = 360, **fmt) -> None:
    """Render a chart plus a table-view twin, so no value is only reachable by hover."""
    fig.update_layout(height=height, legend_title_text="")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    if table is not None:
        with st.expander("Show data"):
            st.dataframe(table, hide_index=True, width="stretch", column_config=fmt or None)


def money_col(label: str) -> st.column_config.NumberColumn:
    return st.column_config.NumberColumn(label, format="dollar")


def pct_col(label: str) -> st.column_config.NumberColumn:
    return st.column_config.NumberColumn(label, format="percent")


def hbar(frame: pd.DataFrame, y: str, x: str, title: str, color: str | None = None,
         color_map: dict | None = None, fmt: str = "$,.0f") -> go.Figure:
    frame = frame.sort_values(x)
    fig = px.bar(frame, x=x, y=y, orientation="h", title=title, color=color,
                 color_discrete_map=color_map, color_discrete_sequence=[SLOTS[0]])
    fig.update_traces(hovertemplate=f"%{{y}}<br>%{{x:{fmt}}}<extra></extra>",
                      texttemplate=f"%{{x:{fmt}}}", textposition="outside", cliponaxis=False,
                      textfont=dict(color=INK_2))
    fig.update_layout(xaxis_title=None, yaxis_title=None, showlegend=color is not None, legend_title_text="",
                      xaxis_range=[0, frame[x].max() * 1.15])
    fig.update_yaxes(categoryorder="array", categoryarray=frame[y].tolist(), showgrid=False)
    fig.update_xaxes(tickformat=fmt.replace(",.0f", ".2s") if fmt.startswith("$") else fmt)
    return fig


# ------------------------------------------------------------------------ layout
df = load_data()

st.title("☕ Local Grind Coffee Co. — Sales Dashboard")
st.caption(
    f"Synthetic point-of-sale data · {df.store_id.nunique()} stores · "
    f"{df.timestamp.min():%b %d, %Y} – {df.timestamp.max():%b %d, %Y} · open 9am–5pm"
)

# one filter row above everything it scopes
min_d, max_d = df.date.min().date(), df.date.max().date()
f1, f2, f3, f4 = st.columns([1.4, 1.2, 1.8, 1.6])
date_range = f1.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
regions = f2.multiselect("Region", REGION_ORDER, placeholder="All regions")
store_options = sorted(df[df.region.isin(regions)].store_name.unique() if regions else df.store_name.unique())
stores = f3.multiselect("Store", store_options, placeholder="All stores")
categories = f4.multiselect("Category", CATEGORY_ORDER, placeholder="All categories")

start, end = (date_range if len(date_range) == 2 else (date_range[0], date_range[0]))
start, end = pd.Timestamp(start), pd.Timestamp(end)


def apply_filters(frame: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    mask = frame.date.between(lo, hi)
    if regions:
        mask &= frame.region.isin(regions)
    if stores:
        mask &= frame.store_name.isin(stores)
    if categories:
        mask &= frame.category.isin(categories)
    return frame[mask]


fdf = apply_filters(df, start, end)
if fdf.empty:
    st.warning("No sales match these filters.")
    st.stop()

# ---------------------------------------------------------------------- KPI tiles
cur = summarize(fdf)
span = end - start
prev_start, prev_end = start - span - timedelta(days=1), start - timedelta(days=1)
prev = summarize(apply_filters(df, prev_start, prev_end)) if prev_start >= df.date.min() else None


def delta(key: str, pct_points: bool = False) -> str | None:
    if not prev or not prev[key]:
        return None
    if pct_points:
        return f"{(cur[key] - prev[key]) * 100:+.1f} pts"
    return f"{cur[key] / prev[key] - 1:+.1%}"


with st.container(key="kpi-grid"):
    k = st.columns(3)
    k[0].metric("Revenue", f"${cur['revenue']:,.0f}", delta("revenue"), border=True)
    k[1].metric("Gross profit", f"${cur['profit']:,.0f}", delta("profit"), border=True)
    k[2].metric("Gross margin", f"{cur['margin']:.1%}", delta("margin", pct_points=True), border=True)
    k = st.columns(3)
    k[0].metric("Transactions", f"{cur['transactions']:,}", delta("transactions"), border=True)
    k[1].metric("Avg ticket", f"${cur['avg_ticket']:.2f}", delta("avg_ticket"), border=True)
    k[2].metric("Items sold", f"{cur['items']:,}", delta("items"), border=True)
st.caption(
    f"Change vs. the previous {span.days + 1} days ({prev_start:%b %d} – {prev_end:%b %d})."
    if prev else "Pick a shorter date range to compare against the previous period."
)

tab_overview, tab_time, tab_products, tab_stores, tab_customers, tab_data = st.tabs(
    ["Overview", "Time patterns", "Products", "Stores", "Customers & payments", "Raw data"]
)

# ----------------------------------------------------------------------- overview
with tab_overview:
    daily = fdf.groupby("date").revenue.sum().rename("revenue").to_frame()
    daily = daily.reindex(pd.date_range(start, end, name="date"), fill_value=0)
    daily["avg_7d"] = daily.revenue.rolling(7, min_periods=1).mean()
    daily = daily.reset_index()

    fig = go.Figure()
    fig.add_scatter(x=daily.date, y=daily.revenue, name="Daily revenue", mode="lines",
                    line=dict(color=BASELINE, width=1), hovertemplate="%{y:$,.0f}")
    fig.add_scatter(x=daily.date, y=daily.avg_7d, name="7-day average", mode="lines",
                    line=dict(color=SLOTS[0], width=2), hovertemplate="%{y:$,.0f}")
    fig.update_layout(title="Revenue over time", hovermode="x unified", yaxis_tickformat="$.2s")
    show(fig, daily, date=st.column_config.DateColumn("Date"), revenue=money_col("Revenue"),
         avg_7d=money_col("7-day average"))

    c1, c2 = st.columns(2)
    with c1:
        by_cat = group_kpis(fdf, "category")
        show(hbar(by_cat, "category", "revenue", "Revenue by category"), by_cat, height=320,
             revenue=money_col("Revenue"), profit=money_col("Profit"), margin=pct_col("Margin"),
             avg_ticket=None)
    with c2:
        by_region = group_kpis(fdf, "region")
        show(hbar(by_region, "region", "revenue", "Revenue by region"), by_region, height=320,
             revenue=money_col("Revenue"), profit=money_col("Profit"), margin=pct_col("Margin"),
             avg_ticket=money_col("Avg ticket"))

    mix = group_kpis(fdf, ["month", "category"])
    mix["share"] = mix.revenue / mix.groupby("month").revenue.transform("sum")
    fig = px.bar(mix, x="month", y="share", color="category", title="Monthly category mix (share of revenue)",
                 category_orders={"category": CATEGORY_ORDER}, color_discrete_map=CATEGORY_COLORS,
                 custom_data=["category", "revenue"])
    fig.update_traces(marker_cornerradius=0, marker_line=dict(color=SURFACE, width=1),
                      hovertemplate="%{customdata[0]}<br>%{x|%b %Y}<br>%{y:.1%} · %{customdata[1]:$,.0f}<extra></extra>")
    fig.update_layout(xaxis_title=None, yaxis_title=None, yaxis_tickformat=".0%", bargap=0.15)
    fig.update_xaxes(dtick="M1", tickformat="%b")
    show(fig, mix[["month", "category", "revenue", "share"]],
         month=st.column_config.DateColumn("Month", format="MMM YYYY"),
         revenue=money_col("Revenue"), share=pct_col("Share"))

# ---------------------------------------------------------------- time patterns
with tab_time:
    n_days = fdf.groupby("dow", observed=False).date.nunique().clip(lower=1)
    heat = fdf.groupby(["dow", "hour"], observed=False).revenue.sum().unstack("hour").fillna(0)
    heat = heat.div(n_days, axis=0).T  # rows = hours, columns = weekdays
    heat.index = [f"{h % 12 or 12}{'am' if h < 12 else 'pm'}" for h in heat.index]
    fig = px.imshow(heat, x=heat.columns.astype(str), y=heat.index, color_continuous_scale=SEQ_GREEN,
                    aspect="auto", title="Average revenue per day, by weekday and hour")
    fig.update_traces(xgap=2, ygap=2, hovertemplate="%{x} %{y}<br>%{z:$,.0f} per day<extra></extra>")
    fig.update_layout(coloraxis_colorbar=dict(title=None, tickformat="$,.0f", thickness=12),
                      xaxis_title=None, yaxis_title=None)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    heat_table = heat.round(2).rename_axis("Hour").reset_index()
    heat_table.columns = [str(c) for c in heat_table.columns]
    show(fig, heat_table, height=420)

    c1, c2 = st.columns(2)
    with c1:
        fdf_kind = fdf.assign(day_type=fdf.timestamp.dt.dayofweek.map(lambda d: "Weekend" if d >= 5 else "Weekday"))
        days_per_kind = fdf_kind.groupby("day_type").date.nunique()
        hourly = fdf_kind.groupby(["day_type", "hour"]).transaction_id.nunique().rename("transactions").reset_index()
        hourly["per_day"] = hourly.transactions / hourly.day_type.map(days_per_kind)
        fig = px.line(hourly, x="hour", y="per_day", color="day_type", markers=True,
                      title="Transactions per hour (avg day)", color_discrete_sequence=SLOTS[:2],
                      category_orders={"day_type": ["Weekday", "Weekend"]})
        fig.update_traces(line_width=2, marker_size=8, hovertemplate="%{y:.1f} transactions")
        fig.update_layout(xaxis_title=None, yaxis_title=None, hovermode="x unified")
        fig.update_xaxes(tickvals=list(range(9, 17)), ticktext=[f"{h % 12 or 12}{'am' if h < 12 else 'pm'}" for h in range(9, 17)])
        show(fig, hourly, per_day=st.column_config.NumberColumn("Per day", format="%.1f"))
    with c2:
        by_dow = group_kpis(fdf, "dow")
        by_dow["per_day"] = by_dow.revenue / by_dow.dow.map(n_days).astype(float)
        fig = px.bar(by_dow, x="dow", y="per_day", title="Average daily revenue by weekday",
                     color_discrete_sequence=[SLOTS[0]])
        fig.update_traces(hovertemplate="%{x}<br>%{y:$,.0f} per day<extra></extra>")
        fig.update_layout(xaxis_title=None, yaxis_title=None, yaxis_tickformat="$,.0f")
        show(fig, by_dow[["dow", "revenue", "per_day", "transactions", "avg_ticket"]],
             revenue=money_col("Revenue"), per_day=money_col("Per day"), avg_ticket=money_col("Avg ticket"))

    st.caption("Urban stores peak in the weekday morning rush; suburban stores are busiest on weekends — "
               "filter by store to compare.")

# ----------------------------------------------------------------------- products
with tab_products:
    by_prod = group_kpis(fdf, ["product_name", "category"]).sort_values("revenue", ascending=False)
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = hbar(by_prod, "product_name", "revenue", "Revenue by product", color="category",
                   color_map=CATEGORY_COLORS)
        fig.update_layout(barmode="relative", legend_traceorder="normal")
        fig.for_each_trace(lambda t: t.update(legendrank=CATEGORY_ORDER.index(t.name)))
        show(fig, height=max(420, 22 * len(by_prod)))
    with c2:
        st.markdown("**Product scorecard**")
        st.dataframe(
            by_prod[["product_name", "category", "revenue", "items", "margin"]],
            hide_index=True, width="stretch", height=max(420, 22 * len(by_prod)) - 30,
            column_config={
                "product_name": "Product",
                "category": "Category",
                "revenue": st.column_config.ProgressColumn("Revenue", format="dollar", min_value=0,
                                                           max_value=float(by_prod.revenue.max())),
                "items": st.column_config.NumberColumn("Units", format="localized"),
                "margin": pct_col("Margin"),
            },
        )

    seasonal = fdf[fdf.category == "Seasonal"]
    if not seasonal.empty:
        weekly = (seasonal.set_index("timestamp").groupby("product_name", observed=True)
                  .quantity.resample("W-MON").sum().rename("units").reset_index())
        order = ["Lavender Oat Latte", "Pumpkin Spice Latte", "Peppermint Mocha", "Hot Chocolate"]
        fig = px.line(weekly, x="timestamp", y="units", color="product_name", title="Seasonal menu — weekly units sold",
                      category_orders={"product_name": order}, color_discrete_sequence=SLOTS)
        fig.update_traces(line_width=2, hovertemplate="%{y:,} units")
        fig.update_layout(xaxis_title=None, yaxis_title=None, hovermode="x unified")
        show(fig, weekly, timestamp=st.column_config.DateColumn("Week of"))

# ------------------------------------------------------------------------- stores
with tab_stores:
    by_store = group_kpis(fdf, ["store_name", "region", "store_type"])
    rewards = (fdf.drop_duplicates("transaction_id").assign(rewards=lambda d: d.customer_type == "Rewards Member")
               .groupby("store_name", observed=True).rewards.mean())
    by_store["rewards_share"] = by_store.store_name.map(rewards).astype(float)
    by_store = by_store.sort_values("revenue", ascending=False)

    c1, c2 = st.columns([2, 3])
    with c1:
        fig = hbar(by_store, "store_name", "revenue", "Revenue by store", color="region", color_map=REGION_COLORS)
        fig.for_each_trace(lambda t: t.update(legendrank=REGION_ORDER.index(t.name)))
        fig.update_layout(barmode="relative")
        show(fig, height=380)
    with c2:
        st.markdown("**Store scorecard**")
        st.dataframe(
            by_store[["store_name", "region", "store_type", "revenue", "transactions", "avg_ticket", "margin", "rewards_share"]],
            hide_index=True, width="stretch", height=350,
            column_config={
                "store_name": "Store", "region": "Region", "store_type": "Type",
                "revenue": money_col("Revenue"), "transactions": st.column_config.NumberColumn("Transactions", format="localized"),
                "avg_ticket": money_col("Avg ticket"), "margin": pct_col("Margin"), "rewards_share": pct_col("Rewards %"),
            },
        )

    store_month = fdf.pivot_table(index="store_name", columns="month", values="revenue", aggfunc="sum",
                                  observed=True).reindex(by_store.store_name)
    fig = px.imshow(store_month, x=[m.strftime("%b") for m in store_month.columns], y=store_month.index,
                    color_continuous_scale=SEQ_BLUE, aspect="auto", title="Monthly revenue by store")
    fig.update_traces(xgap=2, ygap=2, hovertemplate="%{y} · %{x}<br>%{z:$,.0f}<extra></extra>")
    fig.update_layout(coloraxis_colorbar=dict(title=None, tickformat="$.2s", thickness=12),
                      xaxis_title=None, yaxis_title=None)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    sm_table = store_month.round(0).reset_index()
    sm_table.columns = ["Store"] + [m.strftime("%b %Y") for m in store_month.columns]
    show(fig, sm_table, height=380)
    st.caption("Fondren opened on March 10 and ramps up over its first ~10 weeks.")

# ---------------------------------------------------------------------- customers
with tab_customers:
    txns = fdf.groupby("transaction_id", observed=True).agg(
        revenue=("revenue", "sum"), items=("quantity", "sum"),
        customer_type=("customer_type", "first"), payment_method=("payment_method", "first"),
        store_type=("store_type", "first"),
    )
    by_cust = txns.groupby("customer_type", observed=True).agg(
        revenue=("revenue", "sum"), transactions=("revenue", "size"),
        avg_ticket=("revenue", "mean"), items_per_txn=("items", "mean")).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        show(hbar(by_cust, "customer_type", "revenue", "Revenue by customer type"), by_cust, height=260,
             revenue=money_col("Revenue"), avg_ticket=money_col("Avg ticket"),
             items_per_txn=st.column_config.NumberColumn("Items / txn", format="%.2f"))
    with c2:
        show(hbar(by_cust, "customer_type", "avg_ticket", "Average ticket by customer type", fmt="$,.2f"),
             height=260)

    pay_mix = txns.groupby(["customer_type", "payment_method"], observed=True).size().rename("transactions").reset_index()
    pay_mix["share"] = pay_mix.transactions / pay_mix.groupby("customer_type").transactions.transform("sum")
    fig = px.bar(pay_mix, y="customer_type", x="share", color="payment_method", orientation="h",
                 title="Payment method mix by customer type", color_discrete_map=PAYMENT_COLORS,
                 category_orders={"payment_method": PAYMENT_ORDER, "customer_type": CUSTOMER_ORDER[::-1]},
                 custom_data=["payment_method", "transactions"])
    fig.update_traces(marker_cornerradius=0, marker_line=dict(color=SURFACE, width=2),
                      hovertemplate="%{y} · %{customdata[0]}<br>%{x:.1%} (%{customdata[1]:,} txns)<extra></extra>")
    fig.update_layout(xaxis_title=None, yaxis_title=None, xaxis_tickformat=".0%", bargap=0.35)
    show(fig, pay_mix, height=300, share=pct_col("Share"))

    pay_store = txns.groupby(["store_type", "payment_method"], observed=True).size().rename("transactions").reset_index()
    pay_store["share"] = pay_store.transactions / pay_store.groupby("store_type").transactions.transform("sum")
    fig = px.bar(pay_store, x="payment_method", y="share", color="store_type", barmode="group",
                 title="Payment method share: urban vs. suburban stores", color_discrete_sequence=SLOTS[:2],
                 category_orders={"payment_method": PAYMENT_ORDER, "store_type": ["urban", "suburban"]})
    fig.update_traces(hovertemplate="%{x}<br>%{y:.1%}<extra></extra>")
    fig.update_layout(xaxis_title=None, yaxis_title=None, yaxis_tickformat=".0%")
    show(fig, pay_store, height=320, share=pct_col("Share"))

# ------------------------------------------------------------------------ raw data
with tab_data:
    st.markdown(f"**{len(fdf):,} line items** match the current filters.")
    cols = ["transaction_id", "timestamp", "store_name", "region", "product_name", "category", "quantity",
            "unit_price", "revenue", "profit", "customer_type", "payment_method"]
    st.dataframe(fdf[cols].head(5000), hide_index=True, width="stretch", height=480,
                 column_config={"unit_price": money_col("Unit price"), "revenue": money_col("Revenue"),
                                "profit": money_col("Profit")})
    if len(fdf) > 5000:
        st.caption("Showing the first 5,000 rows; the download contains all of them.")
    st.download_button("Download filtered data (CSV)", fdf[cols].to_csv(index=False).encode(),
                       file_name="local_grind_sales_filtered.csv", mime="text/csv")
