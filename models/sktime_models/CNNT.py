from sktime.regression.deep_learning import CNNRegressor
from sktime.networks.cnn import CNNNetwork
from sklearn.utils import check_random_state


class CNNRegressorWithLR(CNNRegressor):
    """CNNRegressor that exposes learning_rate safely."""

    def __init__(
        self,
        n_epochs=2000,
        batch_size=16,
        kernel_size=7,
        avg_pool_size=3,
        n_conv_layers=2,
        callbacks=None,
        verbose=False,
        loss="mean_squared_error",
        metrics=None,
        random_state=0,
        activation="linear",
        use_bias=True,
        optimizer=None,
        learning_rate=None,   # <-- NEW PARAMETER
    ):
        # store new parameter
        self.learning_rate = learning_rate

        # parent init (no varargs allowed)
        super().__init__(
            n_epochs=n_epochs,
            batch_size=batch_size,
            kernel_size=kernel_size,
            avg_pool_size=avg_pool_size,
            n_conv_layers=n_conv_layers,
            callbacks=callbacks,
            verbose=verbose,
            loss=loss,
            metrics=metrics,
            random_state=random_state,
            activation=activation,
            use_bias=use_bias,
            optimizer=optimizer,
        )

    def build_model(self, input_shape, **kwargs):
        """Construct a compiled Keras CNN model with custom learning rate."""
        import tensorflow as tf
        from tensorflow import keras

        tf.random.set_seed(self.random_state)

        metrics = ["accuracy"] if self.metrics is None else self.metrics
        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(
            units=1,
            activation=self.activation,
            use_bias=self.use_bias,
        )(output_layer)

        # ---------- Safe optimizer creation ----------
        if self.learning_rate is not None:
            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            optimizer = (
                keras.optimizers.Adam(learning_rate=0.01)
                if self.optimizer is None
                else self.optimizer
            )

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(
            loss=self.loss,
            optimizer=optimizer,
            metrics=metrics,
        )
        return model
