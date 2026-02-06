from sktime.regression.deep_learning.inceptiontime import InceptionTimeRegressor
from tensorflow import keras


class InceptionTimeRegressorWithLR(InceptionTimeRegressor):
    """InceptionTimeRegressor that exposes learning_rate explicitly."""

    def __init__(
        self,
        n_epochs=1500,
        batch_size=64,
        kernel_size=40,
        n_filters=32,
        use_residual=True,
        use_bottleneck=True,
        bottleneck_size=32,
        depth=6,
        callbacks=None,
        random_state=None,
        verbose=False,
        loss="mean_squared_error",
        metrics=None,
        activation="linear",
        activation_hidden="relu",
        activation_inception="linear",
        optimizer=None,
        learning_rate=0.001,  # <-- added parameter
    ):
        # store the extra parameter
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        # call parent constructor with explicit signature
        super().__init__(
            n_epochs=n_epochs,
            batch_size=batch_size,
            kernel_size=kernel_size,
            n_filters=n_filters,
            use_residual=use_residual,
            use_bottleneck=use_bottleneck,
            bottleneck_size=bottleneck_size,
            depth=depth,
            callbacks=callbacks,
            random_state=random_state,
            verbose=verbose,
            loss=loss,
            metrics=metrics,
            activation=activation,
            activation_hidden=activation_hidden,
            activation_inception=activation_inception,
        )

    def build_model(self, input_shape, **kwargs):
        """Same as original but uses Adam(learning_rate)."""
        from tensorflow import keras

        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(
            units=1,
            activation=self.activation,
        )(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        metrics = ["accuracy"] if self.metrics is None else self.metrics

        # override optimizer with our LR parameter
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        model.compile(
            loss=self.loss,
            optimizer=optimizer,
            metrics=metrics,
        )

        return model
