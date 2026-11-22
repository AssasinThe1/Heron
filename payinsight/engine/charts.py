"""Plotly figures, built only from computed results.

Palette: the dataviz reference instance (validated adjacent-pair order).
  slot 1 blue   = the current period / the main series
  slot 2 orange = the baseline period / the second series
  slot 3 aqua   = a third series
Outage windows use a light critical-red band and are always labelled in text,
never by colour alone. One y-axis per chart; transparent backgrounds so the
Streamlit light/dark theme shows through.
"""

import pandas as pd
import plotly.graph_objects as go

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, "#008300", "#6250d6", "#e34948"]
CRITICAL = "rgba(227,73,72,0.14)"
MUTED = "#8a8984"
GRID = "rgba(138,137,132,0.18)"


def _layout(fig, title, y_title=None, x_title=None, height=340, percent=False):
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15)),
        height=height, margin=dict(l=8, r=8, t=48, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12), hoverlabel=dict(font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        bargap=0.25,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, title=x_title, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=True, zerolinecolor=GRID, title=y_title,
                     tickfont=dict(color=MUTED), ticksuffix="%" if percent else None)
    return fig


def mix_rate_waterfall(R0, mix, rate, R1, label0, label1):
    """Baseline rate -> mix effect -> rate effect -> current rate (all in %)."""
    fig = go.Figure(go.Waterfall(
        x=[label0, "Mix effect", "Rate effect", label1],
        measure=["absolute", "relative", "relative", "total"],
        y=[R0 * 100, mix * 100, rate * 100, R1 * 100],
        text=[f"{R0 * 100:.1f}%", f"{mix * 100:+.2f} pp", f"{rate * 100:+.2f} pp", f"{R1 * 100:.1f}%"],
        textposition="outside", connector=dict(line=dict(color=GRID, width=1)),
        decreasing=dict(marker=dict(color=ORANGE)), increasing=dict(marker=dict(color=AQUA)),
        totals=dict(marker=dict(color=BLUE)),
        hovertemplate="%{x}: %{text}<extra></extra>",
    ))
    lo = min(R0, R1) * 100 - 3
    hi = max(R0, R1) * 100 + 1.5
    fig.update_yaxes(range=[lo, hi])
    return _layout(fig, "Where the change came from: mix vs rate", "Success rate", percent=True)


def share_and_rate(seg, dim_label):
    """Two side-by-side small multiples would be ideal; here: grouped bars of
    share (current vs baseline). The rate change is in the hover and the table."""
    seg = seg.sort_values("w1", ascending=False)
    names = [str(i) for i in seg.index]
    fig = go.Figure()
    fig.add_bar(name="Baseline", x=names, y=seg.w0 * 100, marker_color=ORANGE,
                customdata=list(zip(seg.r0 * 100, seg.n0)),
                hovertemplate="%{x}<br>Baseline share %{y:.1f}%<br>Success rate %{customdata[0]:.1f}%"
                              "<br>Attempts %{customdata[1]:,}<extra></extra>")
    fig.add_bar(name="Current", x=names, y=seg.w1 * 100, marker_color=BLUE,
                customdata=list(zip(seg.r1 * 100, seg.n1)),
                hovertemplate="%{x}<br>Current share %{y:.1f}%<br>Success rate %{customdata[0]:.1f}%"
                              "<br>Attempts %{customdata[1]:,}<extra></extra>")
    fig.update_traces(marker_line_width=0)
    return _layout(fig, f"Share of attempts by {dim_label}", "Share of attempts", percent=True)


def rate_by_segment(seg, dim_label):
    """Success rate per segment with 95% intervals, baseline vs current."""
    seg = seg.sort_values("w1", ascending=False)
    names = [str(i) for i in seg.index]
    fig = go.Figure()
    for nm, r, lo, hi, color in (("Baseline", seg.r0, seg.ci0_lo, seg.ci0_hi, ORANGE),
                                 ("Current", seg.r1, seg.ci1_lo, seg.ci1_hi, BLUE)):
        fig.add_scatter(name=nm, x=names, y=r * 100, mode="markers", marker=dict(size=10, color=color),
                        error_y=dict(type="data", symmetric=False, array=(hi - r) * 100,
                                     arrayminus=(r - lo) * 100, color=color, thickness=1.5, width=4),
                        hovertemplate=f"%{{x}}<br>{nm} %{{y:.1f}}%<extra></extra>")
    return _layout(fig, f"Success rate by {dim_label} (95% ranges)", "Success rate", percent=True)


def contributions(ranked, n=8):
    top = ranked.head(n).iloc[::-1]
    labels = [f"{d}: {s}" for d, s in zip(top.dim, top.segment)]
    fig = go.Figure()
    fig.add_bar(name="Mix", y=labels, x=top.mix * 100, orientation="h", marker_color=BLUE,
                hovertemplate="%{y}<br>Mix %{x:+.2f} pp<extra></extra>")
    fig.add_bar(name="Rate", y=labels, x=top.rate * 100, orientation="h", marker_color=ORANGE,
                hovertemplate="%{y}<br>Rate %{x:+.2f} pp<extra></extra>")
    fig.update_layout(barmode="relative")
    return _layout(fig, "Segments that moved the total most", None, "Contribution (pp)",
                   height=60 + 34 * len(top))


def hourly(df, title, r0=None, events=None):
    """Hourly success rate line with logged outage windows shaded and labelled."""
    fig = go.Figure()
    if events is not None:
        for ev in events.itertuples():
            fig.add_vrect(x0=ev.started_at, x1=ev.ended_at, fillcolor=CRITICAL, line_width=0,
                          annotation_text="logged outage", annotation_position="top left",
                          annotation_font_size=10, annotation_font_color=MUTED)
    fig.add_scatter(x=df.hour, y=df.rate * 100, mode="lines", name="Success rate",
                    line=dict(color=BLUE, width=2), customdata=df.attempts,
                    hovertemplate="%{x|%d %b %H:00}<br>%{y:.1f}% of %{customdata:,} attempts<extra></extra>")
    if r0 is not None:
        fig.add_hline(y=r0 * 100, line=dict(color=ORANGE, width=1.5, dash="dot"),
                      annotation_text=f"baseline {r0 * 100:.1f}%", annotation_position="bottom right",
                      annotation_font_size=10, annotation_font_color=MUTED)
    fig.update_layout(showlegend=False)
    return _layout(fig, title, "Success rate", percent=True)


def weekly_trend(df, i, cmp):
    """Weekly success rate, the selected week marked, its 4-week baseline as a dotted segment."""
    fig = go.Figure()
    half = pd.Timedelta(days=3.5)            # points sit on each week's Monday; shade whole weeks
    if cmp is not None:
        first_base = df.week.iloc[i - 4]
        fig.add_vrect(x0=first_base - half, x1=df.week.iloc[i] - half, fillcolor=GRID,
                      line_width=0, annotation_text="4 weeks before", annotation_position="top left",
                      annotation_font_size=10, annotation_font_color=MUTED)
        fig.add_shape(type="line", x0=first_base, x1=df.week.iloc[i],
                      y0=cmp["base_rate"], y1=cmp["base_rate"],
                      line=dict(color=ORANGE, width=1.5, dash="dot"))
        fig.add_annotation(x=first_base, y=cmp["base_rate"], text=f"average {cmp['base_rate']:.1f}%",
                           showarrow=False, yshift=-10, xanchor="left",
                           font=dict(size=10, color=MUTED))
    fig.add_scatter(x=df.week, y=df.rate, mode="lines+markers", name="Success rate",
                    line=dict(color=BLUE, width=2), marker=dict(size=8, color=BLUE),
                    customdata=list(zip(df.week_end.dt.strftime("%d %b"), df.attempts)),
                    hovertemplate="Week %{x|%d %b} – %{customdata[0]}<br>%{y:.1f}% of "
                                  "%{customdata[1]:,} attempts<extra></extra>")
    fig.add_scatter(x=[df.week.iloc[i]], y=[df.rate.iloc[i]], mode="markers+text",
                    marker=dict(size=13, color=BLUE, line=dict(color="white", width=2)),
                    text=[f"{df.rate.iloc[i]:.1f}%"], textposition="bottom center",
                    hoverinfo="skip")
    fig.update_layout(showlegend=False)
    fig = _layout(fig, "Success rate by week (Mon–Sun, IST)", "Success rate", percent=True)
    lo, hi = df.rate.min(), df.rate.max()
    fig.update_yaxes(range=[lo - 0.35 * (hi - lo), hi + 0.15 * (hi - lo)])
    return fig


def metric_chart(df, dims, bucket, unit, title):
    """Generic chart for a metric query: line over time, bars by segment."""
    pct = unit == "percent"
    fmt = ".1f" if pct else (",.0f" if unit == "count" else ",.2f")
    fig = go.Figure()
    if bucket and dims:
        for i, (key, g) in enumerate(df.groupby(dims[0], sort=False)):
            if i >= 7:
                break
            fig.add_scatter(x=g[bucket], y=g.value, mode="lines+markers", name=str(key),
                            line=dict(color=SERIES[i], width=2), marker=dict(size=8),
                            hovertemplate=f"{key}<br>%{{x}}<br>%{{y:{fmt}}}<extra></extra>")
    elif bucket:
        fig.add_scatter(x=df[bucket], y=df.value, mode="lines+markers", name=title,
                        line=dict(color=BLUE, width=2), marker=dict(size=8),
                        hovertemplate=f"%{{x}}<br>%{{y:{fmt}}}<extra></extra>")
        fig.update_layout(showlegend=False)
    elif dims:
        labels = df[dims].astype(str).agg(" · ".join, axis=1)
        fig.add_bar(x=labels, y=df.value, marker_color=BLUE, marker_line_width=0,
                    hovertemplate=f"%{{x}}<br>%{{y:{fmt}}}<extra></extra>")
        fig.update_layout(showlegend=False)
    else:
        return None
    return _layout(fig, title, None, percent=pct)


def recon_waterfall(rc):
    labels = ["Transactions (IST month)"] + [s[0] for s in rc.steps]
    vals = [rc.txn_value] + [s[1] for s in rc.steps]
    measure = ["absolute"] + ["relative"] * len(rc.steps)
    if abs(rc.unexplained) >= 1:
        labels.append("Unexplained")
        vals.append(rc.unexplained)
        measure.append("relative")
    labels.append("Settlements (before fees)")
    vals.append(rc.settled_before_fees)
    measure.append("total")
    cr = [v / 1e7 for v in vals]
    fig = go.Figure(go.Waterfall(
        x=labels, y=cr, measure=measure,
        text=[f"₹{v:,.2f} Cr" if m != "relative" else f"{v:+,.2f} Cr" for v, m in zip(cr, measure)],
        textposition="outside", connector=dict(line=dict(color=GRID, width=1)),
        decreasing=dict(marker=dict(color=ORANGE)), increasing=dict(marker=dict(color=AQUA)),
        totals=dict(marker=dict(color=BLUE)), hovertemplate="%{x}<br>%{text}<extra></extra>"))
    lo = min(rc.txn_value, rc.settled_before_fees) / 1e7
    fig.update_yaxes(range=[lo * 0.985, max(rc.txn_value, rc.settled_before_fees) / 1e7 * 1.006])
    return _layout(fig, "From transactions to settlements, step by step", "₹ crore", height=380)
