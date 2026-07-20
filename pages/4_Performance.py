"""
pages/4_Performance.py
------------------------
Full evaluation transparency: accuracy, precision/recall/F1,
confusion matrix, training/validation curves, and ROC curves.

All numbers here come from utils/metrics.py, which reads a precomputed
JSON built by evaluating the ALREADY-TRAINED model (models/sentiment_model.keras)
against a reproducible test split of the training dataset -- nothing
on this page retrains or re-evaluates live.
"""

import streamlit as st

from utils.helpers import init_session_state, load_css, render_navbar, render_footer
from utils.metrics import (
    get_headline_numbers, get_classification_report_df,
    get_confusion_matrix, get_training_history, get_roc_data,
)
from utils.charts import confusion_matrix_heatmap, training_curves, loss_curves, roc_chart

init_session_state()
load_css()
render_navbar(active="Performance")

dark = st.session_state.dark_mode
numbers = get_headline_numbers()

st.markdown("## ■ Model Performance")
st.markdown(
    "<p style='color:var(--text-muted)'>Evaluated on a held-out 20% test split, "
    "never seen during training.</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------
report_df = get_classification_report_df()
overall_precision = report_df["Precision"].mean()
overall_recall = report_df["Recall"].mean()
overall_f1 = report_df["F1-Score"].mean()

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""<div class="stat-tile"><div class="stat-value">{numbers['test_accuracy']*100:.1f}%</div>
                <div class="stat-label">Accuracy</div></div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class="stat-tile"><div class="stat-value">{overall_precision:.2f}</div>
                <div class="stat-label">Avg Precision</div></div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""<div class="stat-tile"><div class="stat-value">{overall_recall:.2f}</div>
                <div class="stat-label">Avg Recall</div></div>""", unsafe_allow_html=True)
with m4:
    st.markdown(f"""<div class="stat-tile"><div class="stat-value">{overall_f1:.2f}</div>
                <div class="stat-label">Avg F1-Score</div></div>""", unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------
# Classification report table
# ---------------------------------------------------------------
st.markdown("#### Classification Report")
st.dataframe(report_df, width="stretch", hide_index=True)

st.divider()

# ---------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------
cm, labels = get_confusion_matrix()
cm_col, roc_col = st.columns(2)
with cm_col:
    st.markdown("#### Confusion Matrix")
    st.plotly_chart(confusion_matrix_heatmap(cm, labels, dark=dark), width="stretch")
with roc_col:
    st.markdown("#### ROC Curve (One-vs-Rest)")
    roc_data = get_roc_data()
    st.plotly_chart(roc_chart(roc_data, dark=dark), width="stretch")

st.divider()

# ---------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------
st.markdown("#### Training History")
history = get_training_history()
curve_col1, curve_col2 = st.columns(2)
with curve_col1:
    st.markdown("**Accuracy**")
    st.plotly_chart(training_curves(history, dark=dark), width="stretch")
with curve_col2:
    st.markdown("**Loss**")
    st.plotly_chart(loss_curves(history, dark=dark), width="stretch")

st.caption(
    "Training stopped early once validation loss stopped improving "
    "(EarlyStopping with patience=5), restoring the best-performing weights."
)

render_footer()
