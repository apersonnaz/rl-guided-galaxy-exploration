import tensorflow as tf
import numpy as np
from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
from rl.A3C_2_actors.state_encoder import StateEncoder
from rl.A3C_2_actors.action_manager import ActionManager


class ModelManager:
    def __init__(self, pipeline: PipelineWithPrecalculatedSets):
        self.pipeline = pipeline
        self.action_manager = ActionManager(self.pipeline)
        self.target_sets = ["Scattered"]
        self.curiosity_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
        self.models = {}
        self.lstm_steps = 3

        for target_set in self.target_sets:
            self.models[target_set] = {}
            for curiosity_weight in self.curiosity_weights:
                self.models[target_set][curiosity_weight] = {
                    "set": tf.keras.models.load_model(f'./app/app_models/{target_set}/{curiosity_weight}/set_actor'),
                    "operation": tf.keras.models.load_model(f'./app/app_models/{target_set}/{curiosity_weight}/operation_actor')
                }

    def get_prediction(self, datasets, target_set, curiosity_weight, target_items, found_items_with_ratio, previous_set_states=None, previous_operation_states=None):
        state_encoder = StateEncoder(
            self.pipeline, target_items=target_items, found_items_with_ratio=found_items_with_ratio)

        set_state, reward = state_encoder.encode_datasets(datasets=datasets)

        set_actor = self.models[target_set][curiosity_weight]["set"]
        operation_actor = self.models[target_set][curiosity_weight]["operation"]

        if previous_set_states == None:
            previous_set_states = [
                [-1]*len(state_encoder.set_description)*self.pipeline.discrete_categories_count]*self.lstm_steps

        new_set_states = previous_set_states
        new_set_states.pop(0)
        new_set_states.append(set_state)

        set_probs = set_actor.predict(np.array(new_set_states).reshape((1, self.lstm_steps, len(
            state_encoder.set_description)*self.pipeline.discrete_categories_count)))[0]
        print(set_probs)
        set_probs = self.action_manager.fix_possible_set_action_probs(
            datasets, set_probs)
        print(set_probs)
        set_action = np.random.choice(
            self.action_manager.set_action_dim, p=set_probs)
        operation_state, dummy_reward = state_encoder.encode_dataset(
            datasets[set_action], get_reward=False)

        if previous_operation_states == None:
            previous_operation_states = [
                [-1]*len(state_encoder.set_description)]*self.lstm_steps

        new_operation_states = previous_operation_states
        new_operation_states.pop(0)
        new_operation_states.append(operation_state)

        operation_probs = operation_actor.predict(np.array(
            new_operation_states).reshape((1, self.lstm_steps, len(state_encoder.set_description))))[0]
        print(operation_probs)
        operation_probs = self.action_manager.fix_possible_operation_action_probs(
            datasets[set_action], operation_probs)
        print(operation_probs)
        operation_action = np.random.choice(
            self.action_manager.operation_action_dim, p=operation_probs)
        print(self.action_manager.set_action_types[operation_action])

        action_array = self.action_manager.set_action_types[operation_action].split(
            '-&-')
        operation = action_array[0]
        if len(action_array) > 1:
            dimension = action_array[1].replace('galaxies.', '')
        else:
            dimension = None

        if datasets[set_action].set_id != None:
            set_id = int(datasets[set_action].set_id)
        else:
            set_id = None

        return {
            "predictedOperation": operation,
            "predictedDimension": dimension,
            "predictedSetId": set_id,
            "foundItemsWithRatio": state_encoder.found_items_with_ratio,
            "setStates": new_set_states,
            "operationStates": new_operation_states,
            "reward": reward
        }
