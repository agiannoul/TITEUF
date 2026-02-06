from sktime.regression.deep_learning import CNTCRegressor
from sktime.networks.cntc import CNTCNetwork
from sklearn.utils import check_random_state
from sktime.utils.dependencies import _check_dl_dependencies


class CNTCRegressorWithLR(CNTCRegressor):
    """CNTC Regressor with explicit learning_rate support, sktime-compatible."""

    def __init__(
        self,
        n_epochs=2000,
        batch_size=16,
        filter_sizes=(16, 8),
        kernel_sizes=(1, 1),
        rnn_size=64,
        lstm_size=8,
        dense_size=64,
        callbacks=None,
        verbose=False,
        loss="mean_squared_error",
        metrics=None,
        random_state=0,
        activation="linear",
        activation_hidden="relu",
        activation_attention="sigmoid",
        learning_rate=None,        # <----- NEW PARAMETER
    ):
        _check_dl_dependencies(severity="error")

        # Store new parameter
        self.learning_rate = learning_rate

        # Call parent __init__ explicitly (no varargs allowed)
        super().__init__(
            n_epochs=n_epochs,
            batch_size=batch_size,
            filter_sizes=filter_sizes,
            kernel_sizes=kernel_sizes,
            rnn_size=rnn_size,
            lstm_size=lstm_size,
            dense_size=dense_size,
            callbacks=callbacks,
            verbose=verbose,
            loss=loss,
            metrics=metrics,
            random_state=random_state,
            activation=activation,
            activation_hidden=activation_hidden,
            activation_attention=activation_attention,
        )

    def build_model(self, input_shape, **kwargs):
        """Construct a compiled, un-trained Keras CNTC model with safe LR."""
        from tensorflow import keras

        metrics = ["accuracy"] if self.metrics is None else self.metrics
        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)

        # Final dense output
        output_layer = keras.layers.Dense(
            units=1,
            activation=self.activation,
        )(output_layer)

        # ----------- SAFE OPTIMIZER INITIALIZATION ------------
        if self.learning_rate is not None:
            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            optimizer = keras.optimizers.Adam()

        # Build final model
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(
            loss=self.loss,
            optimizer=optimizer,
            metrics=metrics,
        )
        return model
