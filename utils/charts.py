"""
utils/charts.py
-----------------
All Plotly figure construction lives here, themed to match the app's
dark/light mode and sentiment color palette (config.COLORS), so every
page's charts look consistent instead of each page rolling its own.
"""

import math
import random

import plotly.graph_objects as go
import plotly.express as px

from config import COLORS

SENTIMENT_COLOR_MAP = {
    "positive": COLORS["positive"],
    "negative": COLORS["negative"],
    "neutral": COLORS["neutral"],
}


def _themed_layout(fig, dark: bool, height=380):
    """Applies consistent transparent-background theming to any figure."""
    text_color = COLORS["dark_text"] if dark else COLORS["light_text"]
    grid_color = COLORS["dark_border"] if dark else COLORS["light_border"]
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=text_color, family="Inter, sans-serif"),
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=grid_color, zerolinecolor=grid_color)
    fig.update_yaxes(gridcolor=grid_color, zerolinecolor=grid_color)
    return fig


def sentiment_pie_chart(class_counts: dict, dark: bool = True):
    labels = list(class_counts.keys())
    values = list(class_counts.values())
    colors = [SENTIMENT_COLOR_MAP.get(l, "#999999") for l in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=colors, line=dict(color=COLORS["dark_bg"] if dark else "#fff", width=2)),
        textinfo="label+percent",
    )])
    return _themed_layout(fig, dark, height=360)


def sentiment_bar_chart(class_counts: dict, dark: bool = True):
    labels = list(class_counts.keys())
    values = list(class_counts.values())
    colors = [SENTIMENT_COLOR_MAP.get(l, "#999999") for l in labels]

    fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors)])
    return _themed_layout(fig, dark, height=340)


def confusion_matrix_heatmap(cm, labels, dark: bool = True):
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, COLORS["dark_surface"] if dark else "#fff"], [1, COLORS["brand_orange"]]],
        text=cm, texttemplate="%{text}",
        showscale=False,
    ))
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    fig.update_yaxes(autorange="reversed")
    return _themed_layout(fig, dark, height=380)


def training_curves(history: dict, dark: bool = True):
    epochs = list(range(1, len(history["accuracy"]) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history["accuracy"], name="Train Accuracy",
                              line=dict(color=COLORS["brand_orange"])))
    fig.add_trace(go.Scatter(x=epochs, y=history["val_accuracy"], name="Validation Accuracy",
                              line=dict(color=COLORS["positive"], dash="dot")))
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Accuracy")
    return _themed_layout(fig, dark, height=340)


def loss_curves(history: dict, dark: bool = True):
    epochs = list(range(1, len(history["loss"]) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history["loss"], name="Train Loss",
                              line=dict(color=COLORS["negative"])))
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], name="Validation Loss",
                              line=dict(color=COLORS["neutral"], dash="dot")))
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss")
    return _themed_layout(fig, dark, height=340)


def roc_chart(roc_data: dict, dark: bool = True):
    fig = go.Figure()
    for label, data in roc_data.items():
        color = SENTIMENT_COLOR_MAP.get(label, "#999999")
        fig.add_trace(go.Scatter(
            x=data["fpr"], y=data["tpr"], mode="lines",
            name=f"{label} (AUC={data['auc']:.2f})",
            line=dict(color=color),
        ))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                              name="Random", line=dict(color="gray", dash="dash")))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return _themed_layout(fig, dark, height=380)


def word_frequency_chart(word_counts: list, color: str, dark: bool = True):
    """word_counts: list of [word, count] pairs."""
    words = [w for w, _ in word_counts][::-1]
    counts = [c for _, c in word_counts][::-1]

    fig = go.Figure(data=[go.Bar(x=counts, y=words, orientation="h", marker_color=color)])
    return _themed_layout(fig, dark, height=380)


def length_histogram(bins: list, hist: list, dark: bool = True, color="#FF4500"):
    fig = go.Figure(data=[go.Bar(x=bins[:-1], y=hist, marker_color=color)])
    return _themed_layout(fig, dark, height=300)


def word_cloud_chart(word_counts: list, dark: bool = True, max_words: int = 40):
    """
    A Plotly-based word cloud with collision detection to minimize
    overlapping words while remaining interactive and theme-aware.
    """
    items = word_counts[:max_words]
    if not items:
        return None

    counts = [c for _, c in items]
    max_c, min_c = max(counts), min(counts)

    def _scale(c):
        if max_c == min_c:
            return 24
        return 14 + (c - min_c) / (max_c - min_c) * 26

    palette = [
        COLORS["brand_orange"], COLORS["positive"], COLORS["negative"],
        COLORS["neutral"], "#8B5CF6", "#0EA5E9", "#EC4899",
    ]

    rnd = random.Random(42)
    xs, ys, sizes, colors_, texts, hover = [], [], [], [], [], []
    placed = []

    for i, (word, count) in enumerate(items):
        size = _scale(count)
        width = max(len(word) * size * 0.14, size)
        height = size * 0.8
        angle = rnd.random() * 2 * math.pi
        radius = 0.0

        for _ in range(2000):
            angle += 0.35
            radius += 0.9
            x = radius * math.cos(angle) * 1.6
            y = radius * math.sin(angle) * 0.9

            collision = False
            for px, py, pw, ph in placed:
                if abs(x - px) < (width + pw) / 2 and abs(y - py) < (height + ph) / 2:
                    collision = True
                    break

            if not collision:
                placed.append((x, y, width, height))
                xs.append(x); ys.append(y); sizes.append(size)
                colors_.append(palette[i % len(palette)])
                texts.append(word)
                hover.append(f"{word}: {count} occurrence(s)")
                break

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="text",
        text=texts,
        textfont=dict(size=sizes, color=colors_, family="Inter, sans-serif"),
        hovertext=hover, hoverinfo="text",
    ))
    fig.update_xaxes(visible=False, showgrid=False, zeroline=False)
    fig.update_yaxes(visible=False, showgrid=False, zeroline=False)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=560,
        margin=dict(l=30, r=30, t=30, b=30),
        showlegend=False,
    )
    return fig

def probability_bar_chart(prob_df, dark: bool = True):
    colors = [SENTIMENT_COLOR_MAP.get(s.lower(), "#999999") for s in prob_df["Sentiment"]]
    fig = go.Figure(data=[go.Bar(
        x=prob_df["Sentiment"], y=prob_df["Probability"], marker_color=colors,
        text=[f"{v:.1f}%" for v in prob_df["Probability"]], textposition="outside",
    )])
    fig.update_layout(yaxis_title="Probability (%)", yaxis_range=[0, 105])
    return _themed_layout(fig, dark, height=320)


def confidence_gauge(confidence: float, sentiment: str, dark: bool = True):
    color = SENTIMENT_COLOR_MAP.get(sentiment, COLORS["brand_orange"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "gray"},
            "bar": {"color": color},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 1,
            "bordercolor": COLORS["dark_border"] if dark else COLORS["light_border"],
        },
    ))
    return _themed_layout(fig, dark, height=260)
