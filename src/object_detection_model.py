import tensorflow as tf
from src import training_utils


class ObjectDetectionModel(tf.keras.Model):
    def __init__(self, model_architecture, lambdas=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model_architecture

        if lambdas is None:
            self.lambdas = {'bbox': 0.001, 'class': 1.0,
                            'obj': 1.0, 'obj_less': 1.0}
        else:
            self.lambdas = lambdas

        # Create a tracker for the main loss and each component
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.obj_loss_tracker = tf.keras.metrics.Mean(name="obj_loss")
        self.bbox_loss_tracker = tf.keras.metrics.Mean(name="bbox_loss")
        self.class_loss_tracker = tf.keras.metrics.Mean(name="class_loss")
        self.obj_less_loss_tracker = tf.keras.metrics.Mean(
            name="obj_less_loss")

    # Inside your ObjectDetectionModel class

    def get_config(self):
        # 1. Get the config from the parent class (tf.keras.Model)
        base_config = super().get_config()

        # 2. Add your custom argument(s).
        # You need to serialize your inner model.
        base_config.update({
            "model_architecture": tf.keras.utils.serialize_keras_object(self.model),
            "lambdas": self.lambdas
        })
        return base_config

    # Inside your ObjectDetectionModel class

    @classmethod
    def from_config(cls, config):
        # 1. Get your custom argument *out* of the config dict
        model_arch_config = config.pop("model_architecture")

        # 2. De-serialize it from a config back into a model object
        model_architecture = tf.keras.utils.deserialize_keras_object(
            model_arch_config)
        lambdas = config.pop("lambdas", None)

        # 3. Create and return a new instance of this class,
        #    passing the model to the constructor
        return cls(model_architecture, lambdas=lambdas, **config)

    def call(self, inputs):
        return self.model(inputs)

    def train_step(self, data):
        x, y_true = data

        with tf.GradientTape() as tape:
            # get predictions
            y_pred = self.model(x, training=True)

            # calculate loss
            loss_dict = training_utils.calculate_model_loss_dict(
                y_true=y_true, y_pred=y_pred, lambdas=self.lambdas)
            total_loss = loss_dict["loss"]

        # 4. Apply gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.obj_loss_tracker.update_state(loss_dict['objectness_loss'])
        self.bbox_loss_tracker.update_state(loss_dict['bbox_loss'])
        self.class_loss_tracker.update_state(loss_dict['class_loss'])
        self.obj_less_loss_tracker.update_state(loss_dict['objectless_loss'])

        # 6. Return the final dict for display
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        # List all the trackers Keras should monitor and display.
        # 'self.compiled_metrics' is the built-in one for the loss
        # passed to compile().
        return [
            self.total_loss_tracker,
            self.obj_loss_tracker,
            self.bbox_loss_tracker,
            self.class_loss_tracker,
            self.obj_less_loss_tracker,
        ]
