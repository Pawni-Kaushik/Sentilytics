"""
utils/keras_compat.py
----------------------
One place that decides how Keras is imported, so the rest of the app
never touches `tf.keras` directly.

Why: on some machines (e.g. when both `tensorflow` and `tensorflow-cpu`
are installed, or keras is mismatched) `import tensorflow` succeeds but
`tf.keras` is missing, giving:
    AttributeError: module 'tensorflow' has no attribute 'keras'
Standalone `keras` (Keras 3) can load the same .keras model files, so
we try that first and only fall back to tf.keras if it isn't available.
"""

try:
    import keras
    load_model = keras.models.load_model
    pad_sequences = keras.utils.pad_sequences
except Exception:
    import tensorflow as tf
    load_model = tf.keras.models.load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # noqa: F401
