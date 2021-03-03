import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM
import os

class IntrinsicCuriosityForwardModel:
    def __init__(self, prediction_input_state_dim, target_input_state_dim, output_dim, lr, agent_name, model_path=None):
        self.prediction_input_state_dim = prediction_input_state_dim
        self.target_input_state_dim = target_input_state_dim
        self.output_dim = output_dim
        self.agent_name = agent_name
        self.lr = lr
        self.prediction_model = self.create_model(
            self.prediction_input_state_dim)
        self.target_model = self.create_model(self.target_input_state_dim)
        
        self.opt = tf.keras.optimizers.Adam(lr)

    def create_model(self, input_dim):
        return tf.keras.Sequential([
            Input(input_dim),
            Dense(512, activation='relu'),
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.output_dim, activation='linear')
        ])

    def compute_loss(self, prediction, ground_truth):        
        mse = tf.keras.losses.MeanSquaredError()
        loss = mse(ground_truth, prediction)
        return loss

    def get_loss(self, input_state, ground_truth):
        target = self.target_model.predict(ground_truth)
        prediction = self.prediction_model.predict(input_state)
        loss = self.compute_loss(prediction, target)
        return loss

    def train(self, input_state, output_state):
        with tf.GradientTape() as tape:
            v_pred = self.prediction_model(input_state, training=True)
            targets = self.target_model(output_state)
            assert v_pred.shape == targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(targets))
            grads = tape.gradient(
                loss, self.prediction_model.trainable_variables)
            self.opt.apply_gradients(
                zip(grads, self.prediction_model.trainable_variables))
        return loss

    def save_model(self, name=None, step=None):
        if name == None:
            name = self.agent_name
        directory = "saved_models/" + name + "/icm/"
        if step == None:
            directory += "final"
        else:
            directory += str(step)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.prediction_model.save(directory)
