from sktime.regression.deep_learning import LSTMFCNRegressor

class LSTMFCNRegressorWithLR(LSTMFCNRegressor):
    """LSTM-FCN Regressor with explicit learning_rate parameter."""

    def __init__(
        self,
        n_epochs=2000,
        batch_size=128,
        dropout=0.8,
        kernel_sizes=(8, 5, 3),
        filter_sizes=(128, 256, 128),
        lstm_size=8,
        attention=False,
        callbacks=None,
        random_state=None,
        verbose=0,
        activation="linear",
        optimizer="adam",
        activation_hidden="relu",
        learning_rate=0.001,     # <-- added
    ):
        # store new parameter
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        # everything else: call the parent constructor explicitly
        super().__init__(
            n_epochs=n_epochs,
            batch_size=batch_size,
            dropout=dropout,
            kernel_sizes=kernel_sizes,
            filter_sizes=filter_sizes,
            lstm_size=lstm_size,
            attention=attention,
            callbacks=callbacks,
            random_state=random_state,
            verbose=verbose,
            activation=activation,
            activation_hidden=activation_hidden,
        )

    def build_model(self, input_shape, **kwargs):
        """Same model as original, but with Adam(lr=learning_rate)."""
        import tensorflow as tf
        from tensorflow import keras

        tf.random.set_seed(self.random_state)

        input_layers, output_layer = self._network.build_network(
            input_shape, **kwargs
        )

        output_layer = keras.layers.Dense(
            activation=self.activation,
            units=1,
        )(output_layer)

        # ⚠ USE learning_rate INSTEAD OF hardcoded "sgd"
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        model = keras.models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(
            loss="mean_squared_error",
            optimizer=optimizer,
            metrics=["accuracy"],
        )

        self._callbacks = self.callbacks or None
        return model
