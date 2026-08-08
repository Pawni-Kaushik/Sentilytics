"""
utils/charts.py
-----------------
This file contains all Plotly chart functions used in the project.

Why?
Instead of writing chart code on every Streamlit page,
all graphs are created here so the design and theme remain consistent.
"""

import math          # Used for spiral positioning in the word cloud
import random        # Used to generate random positions for words

import plotly.graph_objects as go   # Low-level Plotly charts
import plotly.express as px         # High-level Plotly charts (kept for future use)

from config import COLORS           # Imports project color palette

# Maps each sentiment to its predefined color
SENTIMENT_COLOR_MAP = {
    "positive": COLORS["positive"],
    "negative": COLORS["negative"],
    "neutral": COLORS["neutral"],
}


def _themed_layout(fig, dark: bool, height=380):
    """
    Applies the same theme to every graph.

    Purpose:
    - Keeps all charts visually consistent.
    - Supports both dark mode and light mode.
    """

    # Select text and grid colors depending on theme
    text_color = COLORS["dark_text"] if dark else COLORS["light_text"]
    grid_color = COLORS["dark_border"] if dark else COLORS["light_border"]

    fig.update_layout(

        # Transparent background allows Streamlit container background to show
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        # Apply common font settings
        font=dict(color=text_color, family="Inter, sans-serif"),

        # Same height for consistency
        height=height,

        # Small margins so charts use maximum space
        margin=dict(l=10, r=10, t=40, b=10),

        # Legend: transparent background AND an explicit font color.
        # Relying on the global `font` above to cascade down to the
        # legend/axes isn't reliable across Plotly versions -- it was
        # leaving legend and axis text almost invisible in light mode
        # (rendering with a much lower-contrast default instead), so
        # every sub-component gets the color explicitly here.
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=text_color)),
    )

    # Make grid/zero lines match the theme, AND set tick + axis-title
    # font color explicitly (same reason as the legend above).
    fig.update_xaxes(
        gridcolor=grid_color, zerolinecolor=grid_color,
        tickfont=dict(color=text_color), title_font=dict(color=text_color),
    )
    fig.update_yaxes(
        gridcolor=grid_color, zerolinecolor=grid_color,
        tickfont=dict(color=text_color), title_font=dict(color=text_color),
    )

    # Trace-level text (e.g. a pie chart's "label + percent" slice
    # text) has its own textfont, separate from the layout font --
    # set explicitly so it doesn't fall back to a low-contrast default.
    fig.update_traces(selector=dict(type="pie"), textfont=dict(color=text_color))

    return fig


def sentiment_pie_chart(class_counts: dict, dark: bool = True):

    # Extract sentiment names
    labels = list(class_counts.keys())

    # Extract number of reviews for each sentiment
    values = list(class_counts.values())

    # Assign appropriate color to every sentiment
    colors = [SENTIMENT_COLOR_MAP.get(l, "#999999") for l in labels]

    fig = go.Figure(data=[go.Pie(

        # Labels shown on pie chart
        labels=labels,

        # Values decide slice size
        values=values,

        # Creates a donut chart
        hole=0.55,

        marker=dict(
            colors=colors,

            # White/Dark outline improves visibility
            line=dict(
                color=COLORS["dark_bg"] if dark else "#fff",
                width=2
            )
        ),

        # Show both label and percentage
        textinfo="label+percent",
    )])

    return _themed_layout(fig, dark, height=360)


def sentiment_bar_chart(class_counts: dict, dark: bool = True):

    labels = list(class_counts.keys())
    values = list(class_counts.values())

    # Match each sentiment with its color
    colors = [SENTIMENT_COLOR_MAP.get(l, "#999999") for l in labels]

    fig = go.Figure(
        data=[go.Bar(
            x=labels,
            y=values,
            marker_color=colors
        )]
    )

    return _themed_layout(fig, dark, height=340)


def confusion_matrix_heatmap(cm, labels, dark: bool = True):

    # Heatmap visualizes classification performance
    fig = go.Figure(data=go.Heatmap(

        # Matrix values
        z=cm,

        # Predicted labels
        x=labels,

        # Actual labels
        y=labels,

        # Dark background to orange color scale
        colorscale=[
            [0, COLORS["dark_surface"] if dark else "#fff"],
            [1, COLORS["brand_orange"]]
        ],

        # Display numbers inside every cell
        text=cm,
        texttemplate="%{text}",

        # Hide color bar
        showscale=False,
    ))

    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )

    # Reverse Y-axis so matrix appears correctly
    fig.update_yaxes(autorange="reversed")

    return _themed_layout(fig, dark, height=380)


def training_curves(history: dict, dark: bool = True):

    # Create epoch numbers automatically
    epochs = list(range(1, len(history["accuracy"]) + 1))

    fig = go.Figure()

    # Training accuracy line
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history["accuracy"],
        name="Train Accuracy",
        line=dict(color=COLORS["brand_orange"])
    ))

    # Validation accuracy line
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history["val_accuracy"],
        name="Validation Accuracy",
        line=dict(
            color=COLORS["positive"],
            dash="dot"
        )
    ))

    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Accuracy"
    )

    return _themed_layout(fig, dark, height=340)


def loss_curves(history: dict, dark: bool = True):

    epochs = list(range(1, len(history["loss"]) + 1))

    fig = go.Figure()

    # Training loss
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history["loss"],
        name="Train Loss",
        line=dict(color=COLORS["negative"])
    ))

    # Validation loss
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history["val_loss"],
        name="Validation Loss",
        line=dict(
            color=COLORS["neutral"],
            dash="dot"
        )
    ))

    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Loss"
    )

    return _themed_layout(fig, dark, height=340)


def roc_chart(roc_data: dict, dark: bool = True):

    fig = go.Figure()

    # Plot ROC curve for every sentiment class
    for label, data in roc_data.items():

        color = SENTIMENT_COLOR_MAP.get(label, "#999999")

        fig.add_trace(go.Scatter(

            # False Positive Rate
            x=data["fpr"],

            # True Positive Rate
            y=data["tpr"],

            mode="lines",

            # Display AUC value in legend
            name=f"{label} (AUC={data['auc']:.2f})",

            line=dict(color=color),
        ))

    # Diagonal line represents random guessing
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(color="gray", dash="dash")
    ))

    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    return _themed_layout(fig, dark, height=380)


def word_cloud_chart(word_counts, dark: bool = True):
    """
    Builds a scatter-based word cloud: each word is placed on a spiral
    around the center, with font size scaled to how often it appears.

    word_counts: list of (word, count) tuples, most common first
                 (e.g. from collections.Counter.most_common()).
    """

    if not word_counts:
        return _themed_layout(go.Figure(), dark, height=380)

    max_count = word_counts[0][1]
    min_count = word_counts[-1][1]
    count_range = max(max_count - min_count, 1)

    # Alternate between brand colors so the cloud isn't monotone
    palette = [
        COLORS["brand_orange"],
        COLORS["positive"],
        COLORS["neutral"],
        COLORS["negative"],
    ]

    xs, ys, sizes, colors, texts = [], [], [], [], []

    # Places each word along an outward-growing spiral (Archimedean
    # spiral) so bigger/more-frequent words tend to land near the
    # center, with a little random jitter so it doesn't look too rigid.
    angle_step = 0.6
    radius_step = 2.2

    for i, (word, count) in enumerate(word_counts):
        angle = i * angle_step
        radius = i * radius_step
        jitter_x = random.uniform(-1.5, 1.5)
        jitter_y = random.uniform(-1.5, 1.5)

        xs.append(radius * math.cos(angle) + jitter_x)
        ys.append(radius * math.sin(angle) + jitter_y)

        # Scale font size between ~14 and ~48 based on frequency
        normalized = (count - min_count) / count_range
        sizes.append(14 + normalized * 34)

        colors.append(palette[i % len(palette)])
        texts.append(word)

    fig = go.Figure(data=[go.Scatter(
        x=xs,
        y=ys,
        mode="text",
        text=texts,
        textfont=dict(size=sizes, color=colors),
        hoverinfo="text",
    )])

    fig.update_xaxes(visible=False, showgrid=False, zeroline=False)
    fig.update_yaxes(visible=False, showgrid=False, zeroline=False)

    return _themed_layout(fig, dark, height=380)


def confidence_gauge(confidence: float, sentiment: str, dark: bool = True):
    """
    Circular gauge showing model confidence (0-100) for the predicted
    sentiment, colored to match that sentiment.
    """

    color = SENTIMENT_COLOR_MAP.get(sentiment, COLORS["brand_orange"])
    track_color = COLORS["dark_border"] if dark else COLORS["light_border"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number=dict(suffix="%", font=dict(color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=track_color),
            bar=dict(color=color),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=1,
            bordercolor=track_color,
        ),
    ))

    return _themed_layout(fig, dark, height=260)


def probability_bar_chart(probabilities, dark: bool = True):
    """
    Horizontal bar chart of per-class probabilities for a single
    prediction. `probabilities` is a DataFrame with columns
    "Sentiment" and "Probability" (0-100), as returned by
    predict_sentiment().
    """

    labels = list(probabilities["Sentiment"])
    values = list(probabilities["Probability"])
    colors = [SENTIMENT_COLOR_MAP.get(l, "#999999") for l in labels]

    fig = go.Figure(data=[go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    )])

    fig.update_xaxes(title="Probability (%)", range=[0, 100])

    return _themed_layout(fig, dark, height=260)


def word_frequency_chart(words, color: str = None, dark: bool = True):
    """
    Horizontal bar chart of the most common words for one sentiment
    class. `words` is a list of (word, count) pairs, most common first
    (e.g. from top_words_by_class[label] in the precomputed metrics).
    """

    bar_color = color or COLORS["brand_orange"]

    # Reverse so the most common word ends up at the top of the chart
    labels = [w for w, _ in words][::-1]
    values = [c for _, c in words][::-1]

    fig = go.Figure(data=[go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=bar_color,
    )])

    fig.update_xaxes(title="Count")

    return _themed_layout(fig, dark, height=max(340, 22 * len(words)))


def length_histogram(bins, hist, dark: bool = True, color: str = None):
    """
    Bar chart for a precomputed histogram (e.g. character or word-count
    length distribution). `bins` are the bin edges (len = len(hist)+1,
    as returned by numpy.histogram); `hist` is the count per bin.
    """

    bar_color = color or COLORS["brand_orange"]

    # Use bin centers as x so each bar sits between its edges
    centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(hist))]

    fig = go.Figure(data=[go.Bar(
        x=centers,
        y=hist,
        marker_color=bar_color,
    )])

    fig.update_layout(bargap=0.05)
    fig.update_xaxes(title="Length")
    fig.update_yaxes(title="Count")

    return _themed_layout(fig, dark, height=300)