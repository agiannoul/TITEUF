from sktime.regression.deep_learning.resnet import ResNetRegressor
from tensorflow import keras


class ResNetRegressorWithLR(ResNetRegressor):
    """ResNetRegressor with explicit learning_rate parameter."""

    def __init__(
        self,
        n_epochs=1500,
        callbacks=None,
        verbose=False,
        loss="mean_squared_error",
        metrics=None,
        batch_size=16,
        random_state=None,
        activation="linear",
        activation_hidden="relu",
        use_bias=True,
        optimizer=None,
        learning_rate=0.01,     # <-- added parameter
    ):
        # store new parameter
        self.learning_rate = learning_rate

        # call parent constructor with full explicit signature
        super().__init__(
            n_epochs=n_epochs,
            callbacks=callbacks,
            verbose=verbose,
            loss=loss,
            metrics=metrics,
            batch_size=batch_size,
            random_state=random_state,
            activation=activation,
            activation_hidden=activation_hidden,
            use_bias=use_bias,
            optimizer=optimizer,
        )

    def build_model(self, input_shape, **kwargs):
        """Same as original ResNetRegressor.build_model but uses Adam(lr)."""
        import tensorflow as tf
        from tensorflow import keras

        tf.random.set_seed(self.random_state)

        # use provided optimizer or override with learning rate
        self.optimizer_ = (
            keras.optimizers.Adam(learning_rate=self.learning_rate)
            if self.optimizer is None
            else self.optimizer
        )

        metrics = (
            ["mean_squared_error"]
            if self.metrics is None
            else self.metrics
        )

        input_layer, output_layer = self._network.build_network(
            input_shape, **kwargs
        )

        output_layer = keras.layers.Dense(
            units=1,
            activation=self.activation,
            use_bias=self.use_bias,
        )(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer_,
            metrics=metrics,
        )

        return model
    def __setstate__(self, state):
        # restore normal attributes
        self.__dict__.update(state)

        # If model_ was saved by pickle, great.
        # If not, rebuild the architecture (but weights will be random)
        if not hasattr(self, "model_") or self.model_ is None:
            # Rebuild empty model_ so that predict does not crash
            input_shape = self.input_shape
            self.model_ = self.build_model(input_shape)