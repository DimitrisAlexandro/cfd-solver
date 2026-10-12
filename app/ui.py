"""AeroLab visual language: an engineering data sheet.

- palette: white sheet, cool grey panels, one engineering blue for actions and
  the first data series; muted comparison series; status colours only for status
- type: IBM Plex Sans (theme, .streamlit/config.toml), tabular figures for numbers
- plots: NACA-report style (boxed frame, inward ticks, fine grid, zero line)
"""
import streamlit as st

INK = "#17202B"
GRAPHITE = "#56606C"
RULE = "#CDD4DC"
GRID = "#E6EAEE"
PANEL = "#EEF1F4"
BLUE = "#1D4E89"
SERIES = ["#1D4E89", "#C2571A", "#2B7A4B", "#7A4FA0", "#8A7B12", "#2F8F9D", "#9C3D54"]
STATUS = {"running": ("blue", "running"), "finished": ("green", "finished"),
          "converged": ("green", "converged"), "failed": ("red", "failed"),
          "diverged": ("red", "diverged"), "cancelled": ("gray", "cancelled"),
          "unsteady": ("orange", "unsteady")}

CSS = f"""
<style>
/* numbers line up in tables, metrics and inputs */
html, body, [class*="st-"], input, textarea {{ font-variant-numeric: tabular-nums; }}

/* sheet width and density */
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1480px; }}
[data-testid="stVerticalBlock"] {{ gap: 0.7rem; }}

/* headings: quiet, set by weight not size */
h1 {{ font-size: 1.55rem !important; font-weight: 600 !important; letter-spacing: -0.01em;
      padding: 0 !important; margin: 0 !important; }}
h2 {{ font-size: 1.15rem !important; font-weight: 600 !important; padding-top: 0.6rem !important; }}
h3 {{ font-size: 1.0rem !important; font-weight: 600 !important; padding-top: 0.4rem !important; }}
h4, h5 {{ font-size: 0.92rem !important; font-weight: 600 !important; color: {GRAPHITE}; }}

/* page header */
.al-head {{ border-bottom: 1px solid {RULE}; padding-bottom: 0.7rem; margin-bottom: 0.4rem; }}
.al-head .al-purpose {{ color: {GRAPHITE}; font-size: 0.93rem; margin-top: 0.25rem; max-width: 78ch; }}

/* setup panel headings inside bordered containers */
.al-step {{ font-weight: 600; font-size: 0.9rem; color: {INK}; margin: 0.5rem 0 0.15rem 0;
            padding-top: 0.45rem; border-top: 1px solid {GRID}; }}
.al-step.first {{ border-top: none; padding-top: 0; margin-top: 0; }}

/* widget labels */
[data-testid="stWidgetLabel"] p {{ font-size: 0.84rem; color: {GRAPHITE}; }}

/* metrics: readouts, not posters */
[data-testid="stMetric"] {{ border-left: 2px solid {BLUE}; padding: 0.15rem 0 0.15rem 0.7rem; }}
[data-testid="stMetricLabel"] p {{ font-size: 0.8rem; color: {GRAPHITE}; }}
[data-testid="stMetricValue"] {{ font-size: 1.35rem; font-weight: 500; color: {INK}; }}
[data-testid="stMetricDelta"] {{ font-size: 0.8rem; }}

/* tabs as a plain ruled strip */
.stTabs [data-baseweb="tab-list"] {{ gap: 1.4rem; border-bottom: 1px solid {RULE}; }}
.stTabs [data-baseweb="tab"] {{ padding: 0.35rem 0; font-size: 0.9rem; }}

/* buttons: rectangular, firm */
.stButton button, .stDownloadButton button, .stFormSubmitButton button {{
    border-radius: 3px; font-weight: 500; }}

/* sidebar navigation */
[data-testid="stSidebarNav"] [data-testid="stNavSectionHeader"] {{
    color: {GRAPHITE}; font-weight: 600; font-size: 0.8rem; margin-top: 0.6rem; }}
[data-testid="stSidebarNavLink"] span {{ font-size: 0.9rem; }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{
    background: #FFFFFF; box-shadow: inset 2px 0 0 {BLUE}; border-radius: 2px; }}
/* code and logs */
code, pre {{ font-size: 0.8rem !important; }}

/* captions */
[data-testid="stCaptionContainer"] {{ color: {GRAPHITE}; }}

@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
"""

def apply():
    """global CSS + plot template; call once per run (entry point)"""
    st.markdown(CSS, unsafe_allow_html=True)
    _plotly_template()
    _matplotlib_style()


def brand(assets):
    st.logo(str(assets / "logo.svg"), icon_image=str(assets / "mark.png"), size="large")


def header(title, purpose=None):
    """page title + one-line purpose under a rule"""
    p = f'<div class="al-purpose">{purpose}</div>' if purpose else ""
    st.markdown(f'<div class="al-head"><h1>{title}</h1>{p}</div>', unsafe_allow_html=True)


def step(text, first=False):
    """small heading inside a setup panel (a hairline separates the groups)"""
    st.markdown(f'<div class="al-step{" first" if first else ""}">{text}</div>', unsafe_allow_html=True)


def status_badge(status):
    key = next((k for k in STATUS if str(status).startswith(k)), None)
    color, _ = STATUS.get(key, ("gray", status))
    st.badge(str(status), color=color)


def status_md(status):
    """inline markdown badge (for titles of expanders, table cells)"""
    key = next((k for k in STATUS if str(status).startswith(k)), None)
    color = STATUS.get(key, ("gray", ""))[0]
    return f":{color}-badge[{status}]"


TRUST = {"green": ("green", "validated range"), "amber": ("orange", "use with care"),
         "red": ("red", "low confidence"), "info": ("gray", "note")}


def trust_md(level, text=None):
    """inline confidence badge (green / amber / red)"""
    color, label = TRUST.get(level, ("gray", str(level)))
    return f":{color}-badge[{text or label}]"


def trust_notes(notes, title="Confidence", expanded=False, container=None):
    """[(level, text)] -> one badge line (the worst level) + the reasons in an expander"""
    import trust as T
    if not notes:
        return
    lev = T.worst(*(l for l, _ in notes))
    c = container or st
    with c.expander(f"{title}: {TRUST[lev][1]}", expanded=expanded, icon=":material/verified:"
                    if lev == "green" else ":material/warning:"):
        for l, t in notes:
            st.markdown(f"{trust_md(l, TRUST[l][1])} {t}")


def plot(fig, key=None, height=None, container=None):
    """plotly figure with the AeroLab template (Streamlit's own theme off)"""
    _plotly_template()
    fig.update_layout(template="aerolab", plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
    lay = fig.layout
    if lay.showlegend is not False and sum(1 for d in fig.data if d.showlegend is not False) > 1:
        fig.update_layout(margin=dict(b=86))          # room for the legend under the x axis
    if "yaxis2" in lay and lay.yaxis2.overlaying:
        fig.update_layout(margin=dict(r=64), yaxis2=dict(zeroline=False, showline=True,
                                                         linecolor=INK, tickfont=dict(size=11)))
    if height:
        fig.update_layout(height=height)
    (container or st).plotly_chart(fig, width="stretch", theme=None, key=key)


def cp_axis(values, title="Cp"):
    """reversed Cp axis that ignores the singular points of sharp edges and
    cusps (panel method: Cp of -80 at a slat cusp would flatten everything)"""
    import numpy as np
    v = np.concatenate([np.asarray(x, float).ravel() for x in values])
    v = v[np.isfinite(v)]
    lo, hi = np.percentile(v, [1.0, 100.0]) if len(v) else (-1.0, 1.0)
    pad = 0.06 * (hi - lo)
    return dict(range=[hi + pad, lo - pad], title=title)


# ----------------------------------------------------------------------
def _plotly_template():
    import plotly.graph_objects as go
    import plotly.io as pio
    axis = dict(showline=True, linecolor=INK, linewidth=1, mirror="ticks", ticks="inside",
                ticklen=5, tickcolor=INK, showgrid=True, gridcolor=GRID, gridwidth=1,
                zeroline=True, zerolinecolor="#9AA4AF", zerolinewidth=1,
                title=dict(font=dict(size=12, color=INK), standoff=8),
                tickfont=dict(size=11, color=INK), minor=dict(ticks="inside", ticklen=3,
                                                              tickcolor=INK, showgrid=False))
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family="IBM Plex Sans, Helvetica, Arial, sans-serif", size=12, color=INK),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", colorway=SERIES,
        xaxis=axis, yaxis=axis,
        margin=dict(l=56, r=18, t=34, b=48),
        title=dict(font=dict(size=13, color=INK), x=0.0, xanchor="left", y=0.98),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0.0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=RULE, font=dict(color=INK, size=11)),
        hovermode="closest")
    t.data.scatter = [go.Scatter(marker=dict(size=5, line=dict(width=0)), line=dict(width=1.6))]
    pio.templates["aerolab"] = t
    pio.templates.default = "aerolab"


def _matplotlib_style():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
        "ytick.right": True, "axes.grid": False, "figure.dpi": 110, "savefig.dpi": 110,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES), "image.cmap": "turbo"})
