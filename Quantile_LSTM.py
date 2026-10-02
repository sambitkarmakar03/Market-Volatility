import tensorflow as tf
from tensorflow.keras import layers, models
import keras


@tf.keras.utils.register_keras_serializable(package="Custom")
class PinballLoss(tf.keras.losses.Loss):
    

    def __init__(self, quantiles=(0.10, 0.50, 0.90), name="pinball_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.quantiles_list = list(quantiles)
        self.quantiles = tf.constant(self.quantiles_list, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        if len(y_true.shape) == 1:
            y_true = tf.expand_dims(y_true, -1)

        error = y_true - y_pred
        losses = tf.maximum(self.quantiles * error, (self.quantiles - 1.0) * error)
        return tf.reduce_mean(tf.reduce_sum(losses, axis=-1))

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"quantiles": self.quantiles_list})
        return base_config


def build_quantile_lstm(
    lookback_window: int = 30,
    num_features: int = 4,
    quantiles=(0.10, 0.50, 0.90),
    enforce_monotonic: bool = True,
):
    """
    Constructs a Quantile LSTM model for multi-quantile volatility forecasting.

    enforce_monotonic: if True, guarantees q10 <= q50 <= q90 by construction
    (predict the median, then two non-negative offsets added/subtracted via
    softplus, instead of three independent linear outputs that can cross).
    Quantile crossing is a well-known failure mode of naive multi-head
    quantile regression and is worth calling out explicitly if you disable it.
    """
    inputs = layers.Input(shape=(lookback_window, num_features), name="sequence_input")

    x = layers.LSTM(64, return_sequences=True, name="lstm_1")(inputs)
    x = layers.Dropout(0.2, name="dropout_1")(x)

    x = layers.LSTM(32, return_sequences=False, name="lstm_2")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)

    x = layers.Dense(16, activation="relu", name="dense_hidden")(x)

    if enforce_monotonic and len(quantiles) == 3:
        # Predict median directly, plus two non-negative gaps (softplus >= 0).
        median = layers.Dense(1, activation="linear", name="q_median")(x)
        upper_gap = layers.Dense(1, activation="softplus", name="q_upper_gap")(x)
        lower_gap = layers.Dense(1, activation="softplus", name="q_lower_gap")(x)

        q_low = layers.Subtract(name="q_low")([median, lower_gap])
        q_high = layers.Add(name="q_high")([median, upper_gap])

        outputs = layers.Concatenate(name="quantile_outputs")([q_low, median, q_high])
    else:
        outputs = layers.Dense(len(quantiles), activation="linear", name="quantile_outputs")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Quantile_LSTM")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=PinballLoss(quantiles=list(quantiles)),
    )

    return model


if __name__ == "__main__":
    model = build_quantile_lstm()
    model.summary()