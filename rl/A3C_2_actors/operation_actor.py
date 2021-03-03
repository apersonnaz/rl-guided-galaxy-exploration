import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM
import os

class OperationActor:
    def __init__(self, state_dim, action_dim, steps, lr, agent_name, model_path=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.steps = steps
        self.agent_name = agent_name
        self.lr = lr
        if model_path == None:
            self.model = self.create_model()
        else:
            self.model = tf.keras.models.load_model(model_path)
        self.opt = tf.keras.optimizers.Adam(lr)
        self.entropy_beta = 0.01

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.steps, self.state_dim)),
            Dense(2048, activation='relu'),
            Dense(2048, activation='relu'),
            LSTM(1024, return_sequences=False),
            Dense(self.action_dim, activation='softmax')
        ])

    def compute_loss(self, actions, logits, advantages):
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        entropy_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = ce_loss(
            actions, logits, sample_weight=tf.stop_gradient(advantages))
        entropy = entropy_loss(logits, logits)
        return policy_loss - self.entropy_beta * entropy

    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            loss = self.compute_loss(
                actions, logits, advantages)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(
                zip(grads, self.model.trainable_variables))
        return loss

    def save_model(self, name=None, step=None):
        if name == None:
            name = self.agent_name
        directory = "saved_models/" + name + "/operation_actor/"
        if step == None:
            directory += "final"
        else:
            directory += str(step)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        self.model.save(directory)

