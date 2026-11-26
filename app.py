import html
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "epl_players_scored.csv"
WEIGHTS_PATH = BASE_DIR / "model_weights.json"


st.set_page_config(
    page_title="EPL Wage & Transfer DSS",
    page_icon="⚽",
    layout="wide",
)

CUSTOM_CSS = """
<style>
  html, body, .stApp, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
  .stApp p, .stApp label, .stApp input, .stApp button, .stApp textarea,
  [data-testid="stSidebar"], [data-testid="stMarkdownContainer"],
  [data-testid="stTabs"] button, .dss-card, .dss-badge {
    font-family: "Times New Roman", Times, serif !important;
  }
  .dss-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    margin-bottom: 8px;
  }
  @media (max-width: 900px) {
    .dss-grid {
      grid-template-columns: 1fr;
    }
  }
  .dss-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.35);
    border-radius: 0;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 88px;
    box-sizing: border-box;
  }
  .dss-card--ok { border-left: 3px solid #22c55e; }
  .dss-card--warn { border-left: 3px solid #f59e0b; }
  .dss-card--danger { border-left: 3px solid #ef4444; }
  .dss-card--info { border-left: 3px solid #3b82f6; }
  .dss-card--neutral { border-left: 3px solid rgba(128, 128, 128, 0.55); }

  .dss-label {
    font-size: 0.74rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    opacity: 0.72;
    margin-bottom: 4px;
  }
  .dss-value {
    font-size: 1.35rem;
    font-weight: 700;
    line-height: 1.15;
    font-variant-numeric: tabular-nums;
    margin-bottom: 4px;
  }
  .dss-sub {
    font-size: 0.80rem;
    opacity: 0.82;
    line-height: 1.3;
    font-variant-numeric: tabular-nums;
  }
  .dss-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 0;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: #ffffff;
    width: fit-content;
    margin-bottom: 4px;
  }
  .dss-badge--ok { background: #16a34a; }
  .dss-badge--warn { background: #d97706; }
  .dss-badge--danger { background: #dc2626; }
  .dss-badge--info { background: #2563eb; }
  .dss-pos { color: #22c55e; font-weight: 600; }
  .dss-neg { color: #ef4444; font-weight: 600; }
  .dss-muted { opacity: 0.75; }

  /* Sharp corners for Streamlit bordered containers too */
  [data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 0 !important;
  }
</style>
"""


def render_card(label: str, value_html: str, sub_html: str, tone: str = "neutral") -> str:
    return (
        f'<div class="dss-card dss-card--{tone}">'
        f'  <div class="dss-label">{html.escape(label)}</div>'
        f'  <div>{value_html}</div>'
        f'  <div class="dss-sub">{sub_html}</div>'
        f'</div>'
    )


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_model_info() -> dict:
    with open(WEIGHTS_PATH, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def apply_rules(df: pd.DataFrame, overpaid_ratio: float, underpaid_ratio: float) -> pd.DataFrame:
    out = df.copy()
    out["wage_diff_eur"] = out["wage_eur"] - out["fair_wage_eur"]
    out["wage_ratio"] = (out["wage_eur"] / out["fair_wage_eur"]).round(3)

    def _label(r: float) -> str:
        if r >= overpaid_ratio:
            return "Overpaid"
        if r <= underpaid_ratio:
            return "Underpaid"
        return "Fair"

    out["wage_status"] = out["wage_ratio"].apply(_label)
    return out


def render_fixed_scatter(squad_df: pd.DataFrame, show_source: bool = False) -> None:
    if squad_df.empty:
        st.info("No players in squad to display.")
        return

    max_val = float(max(squad_df["wage_eur"].max(), squad_df["fair_wage_eur"].max(), 10000))
    axis_max = max_val * 1.08
    diag_df = pd.DataFrame({"x": [0, axis_max], "y": [0, axis_max]})

    diag_line = (
        alt.Chart(diag_df)
        .mark_line(strokeDash=[5, 5], color="gray")
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[0, axis_max])),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[0, axis_max])),
        )
    )

    tooltips = [
        "short_name", "primary_pos", "age", "overall", "potential",
        alt.Tooltip("wage_eur:Q", format=",.0f", title="Actual (€)"),
        alt.Tooltip("fair_wage_eur:Q", format=",.0f", title="ML Fair (€)"),
        alt.Tooltip("wage_diff_eur:Q", format="+,.0f", title="Diff (€)"),
        "wage_status",
    ]
    if show_source and "source" in squad_df.columns:
        tooltips.append("source")

    points = (
        alt.Chart(squad_df)
        .mark_circle(size=90)
        .encode(
            x=alt.X("fair_wage_eur:Q", title="ML Fair Wage (€/week)", scale=alt.Scale(domain=[0, axis_max])),
            y=alt.Y("wage_eur:Q", title="Actual Wage (€/week)", scale=alt.Scale(domain=[0, axis_max])),
            color=alt.Color(
                "wage_status:N",
                scale=alt.Scale(
                    domain=["Overpaid", "Fair", "Underpaid"],
                    range=["#e74c3c", "#95a5a6", "#22c55e"],
                ),
                title="Status",
            ),
            tooltip=tooltips,
        )
    )
    chart = (diag_line + points).properties(height=420).configure(font="Times New Roman")
    st.altair_chart(chart, use_container_width=True)


def evaluate_budget_status(
    wage_bill: int,
    wage_cap: int,
    warn_cap_pct: int,
    overpaid_df: pd.DataFrame,
    has_changes: bool = False,
    ovr_delta: float = 0.0,
) -> tuple[str, str, str]:
    cap_pct = (wage_bill / wage_cap) * 100 if wage_cap > 0 else 0.0
    headroom = wage_cap - wage_bill

    if cap_pct > 100:
        if not overpaid_df.empty:
            top = overpaid_df.iloc[0]
            sub = (
                f"Suggested cut: <b>{html.escape(str(top['short_name']))}</b> "
                f"(€{top['wage_eur']:,.0f} vs fair €{top['fair_wage_eur']:,.0f})."
            )
        else:
            sub = "Reduce wages or raise the club budget cap."
        return "danger", "OVER BUDGET", sub

    if has_changes and ovr_delta >= 0 and cap_pct <= warn_cap_pct:
        return (
            "ok",
            "RECOMMENDED",
            f"Under cap ({cap_pct:.1f}%) and squad OVR {ovr_delta:+.2f}. Ready to approve.",
        )

    if cap_pct >= warn_cap_pct:
        return (
            "warn",
            "NEAR CAP",
            f"Only €{headroom:,.0f}/wk remaining. Review {len(overpaid_df)} overpaid players.",
        )

    return (
        "info",
        "HEALTHY",
        f"Safe margin ({cap_pct:.1f}% used). €{headroom:,.0f}/wk available for transfers.",
    )


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    df = load_data()
    model_info = load_model_info()

    st.title("EPL Football Wage & Transfer DSS")
    st.caption("Decision Support System: Base Club Financial Audit + Separate What-If Simulator + ML Fair Wage")

    # ---------------- SIDEBAR: CLUB & GLOBAL RULES ----------------
    st.sidebar.header("Club & Budget Settings")
    clubs = sorted(df["club_name"].unique().tolist())
    default_idx = clubs.index("Manchester United") if "Manchester United" in clubs else 0
    selected_club = st.sidebar.selectbox("Select Club", clubs, index=default_idx)

    base_raw = df[df["club_name"] == selected_club].copy()
    market_pool = df[df["club_name"] != selected_club].copy()

    base_wage = int(base_raw["wage_eur"].sum())
    default_cap = int(round(base_wage * 1.05 / 50000) * 50000)
    min_cap = max(200000, int(round(base_wage * 0.6 / 50000) * 50000))
    max_cap = int(round(base_wage * 1.4 / 50000) * 50000)

    wage_cap = st.sidebar.slider(
        "Club's Weekly Wage Budget (€)",
        min_value=min_cap,
        max_value=max_cap,
        value=default_cap,
        step=50000,
        format="€%d",
    )

    with st.sidebar.expander("Decision Rules Thresholds"):
        overpaid_ratio = st.slider("Overpaid threshold (Actual / Fair ≥)", 1.10, 1.50, 1.25, 0.05)
        underpaid_ratio = st.slider("Underpaid threshold (Actual / Fair ≤)", 0.50, 0.90, 0.80, 0.05)
        warn_cap_pct = st.slider("Near-cap warning (% of budget)", 80, 98, 90, 1)

    # ---------------- BASELINE CALCULATIONS (UNMODIFIED) ----------------
    base_squad = apply_rules(base_raw, overpaid_ratio, underpaid_ratio)
    base_cap_pct = (base_wage / wage_cap) * 100 if wage_cap > 0 else 0.0
    base_headroom = wage_cap - base_wage
    base_over_amount = base_wage - wage_cap
    base_ovr = float(base_squad["overall"].mean())
    base_age = float(base_squad["age"].mean())
    base_overpaid = base_squad[base_squad["wage_status"] == "Overpaid"].sort_values("wage_diff_eur", ascending=False)
    base_overpay_eur = int(base_overpaid["wage_diff_eur"].sum()) if not base_overpaid.empty else 0

    base_tone, base_status_label, base_status_sub = evaluate_budget_status(
        base_wage, wage_cap, warn_cap_pct, base_overpaid
    )

    # ---------------- MAIN TABS ----------------
    tab_base, tab_whatif, tab_ml = st.tabs([
        "1. Base Club Overview",
        "2. What-If Simulator",
        "3. ML Model Metrics",
    ])

    # ==================== TAB 1: BASE CLUB OVERVIEW ====================
    with tab_base:
        st.subheader(f"{selected_club} — Base Financial & Squad Overview")

        # Row 1: Wage Paid | Wage Usage | Remaining / Over Budget
        usage_tone = "danger" if base_cap_pct > 100 else ("warn" if base_cap_pct >= warn_cap_pct else "ok")
        b_paid = render_card(
            "Wage Paid per Week (EUR)",
            f'<div class="dss-value">€{base_wage:,.0f} / wk</div>',
            f"Club budget cap: <b>€{wage_cap:,.0f}/wk</b> &nbsp;|&nbsp; Squad: <b>{len(base_squad)} players</b>",
            tone="neutral",
        )
        b_usage = render_card(
            "Wage Usage per Week",
            f'<div class="dss-value">{base_cap_pct:.1f}%</div>',
            f"€{base_wage:,.0f} of €{wage_cap:,.0f} weekly cap",
            tone=usage_tone,
        )
        if base_over_amount > 0:
            b_rem = render_card(
                "Over Budget",
                f'<div class="dss-value dss-neg">+€{base_over_amount:,.0f} / wk</div>',
                f"Exceeds cap by <b>+{base_cap_pct - 100:.1f}%</b>",
                tone="danger",
            )
        else:
            b_rem = render_card(
                "Remaining Budget",
                f'<div class="dss-value dss-pos">€{base_headroom:,.0f} / wk</div>',
                f"<b>{100 - base_cap_pct:.1f}%</b> budget headroom available",
                tone="warn" if base_cap_pct >= warn_cap_pct else "ok",
            )
        st.markdown(f'<div class="dss-grid">{b_paid}{b_usage}{b_rem}</div>', unsafe_allow_html=True)

        # Row 2: Avg Rating | Overpaid Players | Base Status
        b_ovr = render_card(
            "Avg Squad Rating (OVR)",
            f'<div class="dss-value">{base_ovr:.2f}</div>',
            f"Avg age: <b>{base_age:.1f} yrs</b> &nbsp;|&nbsp; Top OVR: <b>{int(base_squad['overall'].max())}</b>",
            tone="info",
        )
        b_overpaid = render_card(
            "Overpaid Players (ML)",
            f'<div class="dss-value">{len(base_overpaid)} / {len(base_squad)} players</div>',
            f"Est. weekly overpay: <b>+€{base_overpay_eur:,.0f}/wk</b> (≥ {overpaid_ratio:.2f}× fair)",
            tone="danger" if len(base_overpaid) >= 8 else ("warn" if len(base_overpaid) > 0 else "ok"),
        )
        b_status = render_card(
            "Base Status (Decision Engine)",
            f'<span class="dss-badge dss-badge--{base_tone}">{base_status_label}</span>',
            base_status_sub,
            tone=base_tone,
        )
        st.markdown(f'<div class="dss-grid">{b_ovr}{b_overpaid}{b_status}</div>', unsafe_allow_html=True)

        st.markdown("#### Actual Wage vs. ML Fair Wage (Base Squad)")
        render_fixed_scatter(base_squad, show_source=False)

        base_cols = [
            "short_name", "primary_pos", "age", "overall", "potential",
            "wage_eur", "fair_wage_eur", "wage_diff_eur", "wage_ratio", "wage_status",
        ]
        st.dataframe(
            base_squad[base_cols].sort_values("wage_diff_eur", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    # ==================== TAB 2: WHAT-IF SIMULATOR ====================
    with tab_whatif:
        st.subheader(f"{selected_club} — What-If Transfer & Wage Simulator")

        with st.container(border=True):
            wc1, wc2, wc3 = st.columns([1.3, 1.3, 1.0])
            with wc1:
                sell_filter = st.selectbox(
                    "Filter Squad to Sell",
                    ["ALL", "Overpaid Only", "GK", "DEF", "MID", "FWD"],
                )
                if sell_filter == "ALL":
                    filtered_sell = base_squad
                elif sell_filter == "Overpaid Only":
                    filtered_sell = base_squad[base_squad["wage_status"] == "Overpaid"]
                else:
                    filtered_sell = base_squad[base_squad["pos_group"] == sell_filter]
                sell_options = filtered_sell.sort_values("wage_eur", ascending=False)["short_name"].tolist()
                sold_names = st.multiselect("Sell Players (Remove from Squad)", sell_options)
            with wc2:
                pos_filter = st.selectbox("Filter Transfer Market by Line", ["ALL", "GK", "DEF", "MID", "FWD"])
                filtered_market = market_pool if pos_filter == "ALL" else market_pool[market_pool["pos_group"] == pos_filter]
                buy_options = filtered_market.sort_values("overall", ascending=False)["short_name"].tolist()
                bought_names = st.multiselect("Buy Players (Add to Squad)", buy_options)
            with wc3:
                wage_adj_pct = st.slider("Squad Wage Adjustment (%)", min_value=-20, max_value=20, value=0, step=1)

        kept_squad = base_raw[~base_raw["short_name"].isin(sold_names)].copy()
        kept_squad["source"] = "Current"

        bought_squad = market_pool[market_pool["short_name"].isin(bought_names)].copy()
        bought_squad["source"] = "New Signing"

        sim_squad = pd.concat([kept_squad, bought_squad], ignore_index=True)
        if wage_adj_pct != 0:
            sim_squad["wage_eur"] = (sim_squad["wage_eur"] * (1 + wage_adj_pct / 100.0)).round().astype(int)
        sim_squad = apply_rules(sim_squad, overpaid_ratio, underpaid_ratio)

        new_wage = int(sim_squad["wage_eur"].sum())
        wage_delta = new_wage - base_wage
        new_cap_pct = (new_wage / wage_cap) * 100 if wage_cap > 0 else 0.0
        new_headroom = wage_cap - new_wage
        new_over_amount = new_wage - wage_cap

        new_ovr = float(sim_squad["overall"].mean()) if len(sim_squad) else 0.0
        ovr_delta = new_ovr - base_ovr
        sim_overpaid = sim_squad[sim_squad["wage_status"] == "Overpaid"].sort_values("wage_diff_eur", ascending=False)
        sim_overpay_eur = int(sim_overpaid["wage_diff_eur"].sum()) if not sim_overpaid.empty else 0
        overpaid_delta = len(sim_overpaid) - len(base_overpaid)

        has_changes = bool(sold_names or bought_names or wage_adj_pct != 0)
        sim_tone, sim_status_label, sim_status_sub = evaluate_budget_status(
            new_wage, wage_cap, warn_cap_pct, sim_overpaid, has_changes=has_changes, ovr_delta=ovr_delta
        )

        # Row 1: What-If Wage Usage / Current Wage | Over/Remaining | Status
        sim_usage_tone = "danger" if new_cap_pct > 100 else ("warn" if new_cap_pct >= warn_cap_pct else "ok")
        w_delta_cls = "dss-neg" if wage_delta > 0 else ("dss-pos" if wage_delta < 0 else "dss-muted")
        w_usage = render_card(
            "What-If Wage Usage / Paid",
            f'<div class="dss-value">{new_cap_pct:.1f}% &nbsp;·&nbsp; €{new_wage:,.0f}</div>',
            f'Base: €{base_wage:,.0f}/wk &nbsp;|&nbsp; Change: <span class="{w_delta_cls}">{wage_delta:+,.0f} €/wk</span>',
            tone=sim_usage_tone,
        )
        if new_over_amount > 0:
            w_over = render_card(
                "Over Budget",
                f'<div class="dss-value dss-neg">+€{new_over_amount:,.0f} / wk</div>',
                f"Exceeds cap by <b>+{new_cap_pct - 100:.1f}%</b> (needs €{new_over_amount:,.0f} cut)",
                tone="danger",
            )
        else:
            w_over = render_card(
                "Remaining Budget",
                f'<div class="dss-value dss-pos">€{new_headroom:,.0f} / wk</div>',
                f"Within cap (<b>{100 - new_cap_pct:.1f}%</b> headroom remaining)",
                tone="warn" if new_cap_pct >= warn_cap_pct else "ok",
            )
        w_status = render_card(
            "What-If Status",
            f'<span class="dss-badge dss-badge--{sim_tone}">{sim_status_label}</span>',
            sim_status_sub,
            tone=sim_tone,
        )
        st.markdown(f'<div class="dss-grid">{w_usage}{w_over}{w_status}</div>', unsafe_allow_html=True)

        # Row 2: New OVR | Overpaid Change | Transfer Summary
        ovr_cls = "dss-pos" if ovr_delta > 0 else ("dss-neg" if ovr_delta < 0 else "dss-muted")
        w_ovr = render_card(
            "What-If Avg Rating (OVR)",
            f'<div class="dss-value">{new_ovr:.2f}</div>',
            f'Base: {base_ovr:.2f} &nbsp;|&nbsp; Delta: <span class="{ovr_cls}">{ovr_delta:+.2f} OVR</span>',
            tone="ok" if ovr_delta > 0 else ("warn" if ovr_delta < 0 else "neutral"),
        )
        op_cls = "dss-pos" if overpaid_delta < 0 else ("dss-neg" if overpaid_delta > 0 else "dss-muted")
        w_op = render_card(
            "Overpaid Players (After What-If)",
            f'<div class="dss-value">{len(sim_overpaid)} / {len(sim_squad)} players</div>',
            f'Overpay: <b>+€{sim_overpay_eur:,.0f}/wk</b> &nbsp;|&nbsp; vs Base: <span class="{op_cls}">{overpaid_delta:+d}</span>',
            tone="danger" if len(sim_overpaid) >= 8 else ("warn" if len(sim_overpaid) > 0 else "ok"),
        )
        changes_txt = f"Sold: <b>{len(sold_names)}</b> &nbsp;|&nbsp; Bought: <b>{len(bought_names)}</b>"
        if wage_adj_pct != 0:
            changes_txt += f" &nbsp;|&nbsp; Adj: <b>{wage_adj_pct:+d}%</b>"
        w_sum = render_card(
            "What-If Squad Summary",
            f'<div class="dss-value">{len(sim_squad)} players</div>',
            changes_txt,
            tone="info" if has_changes else "neutral",
        )
        st.markdown(f'<div class="dss-grid">{w_ovr}{w_op}{w_sum}</div>', unsafe_allow_html=True)

        st.markdown("#### Simulated Squad Table")
        sim_cols = [
            "short_name", "source", "primary_pos", "age", "overall", "potential",
            "wage_eur", "fair_wage_eur", "wage_diff_eur", "wage_ratio", "wage_status",
        ]
        st.dataframe(
            sim_squad[sim_cols].sort_values("wage_diff_eur", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    # ==================== TAB 3: ML MODEL METRICS ====================
    with tab_ml:
        st.subheader("Ridge Regression Model (Log-Wage Target)")
        m = model_info["metrics"]
        m_cards = (
            render_card(
                "R² (Log-Wage)",
                f'<div class="dss-value">{m["r2_log"]:.4f}</div>',
                f"Explains {m['r2_log'] * 100:.1f}% of log-wage variance",
                tone="ok",
            )
            + render_card(
                "R² (Actual EUR)",
                f'<div class="dss-value">{m["r2_eur"]:.4f}</div>',
                f"Explains {m['r2_eur'] * 100:.1f}% of weekly EUR wages",
                tone="info",
            )
            + render_card(
                "Test MAE",
                f'<div class="dss-value">€{m["mae_eur"]:,.0f} / wk</div>',
                f"Evaluated on {m['test_samples']} held-out EPL players",
                tone="neutral",
            )
        )
        st.markdown(f'<div class="dss-grid">{m_cards}</div>', unsafe_allow_html=True)

        coef_df = pd.DataFrame({
            "Feature": model_info["features"],
            "Standardized Weight": model_info["coefficients"],
        }).sort_values("Standardized Weight", ascending=False)

        bar = (
            alt.Chart(coef_df)
            .mark_bar()
            .encode(
                x=alt.X("Standardized Weight:Q"),
                y=alt.Y("Feature:N", sort="-x"),
                color=alt.condition(
                    alt.datum["Standardized Weight"] > 0,
                    alt.value("#2ecc71"),
                    alt.value("#e74c3c"),
                ),
            )
            .configure(font="Times New Roman")
        )
        st.altair_chart(bar, use_container_width=True)
        st.caption(f"Trained on 80% / Evaluated on {m['test_samples']} EPL players (20% test set), alpha={model_info['alpha']}.")


if __name__ == "__main__":
    main()
