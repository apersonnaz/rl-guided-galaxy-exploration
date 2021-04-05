from app.pipelines.predicateitem import PredicateItem
from app.pipelines.dataset import Dataset
from app.pipelines import pipeline
import gym
import numpy as np
import json
import random
from gym import spaces
import random

from numpy.lib.function_base import piecewise
from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
from os import listdir
from .state_encoder import StateEncoder
from .target_set_generator import TargetSetGenerator
from .action_manager import ActionManager


class PipelineEnvironment(gym.Env):
    def __init__(self, pipeline: PipelineWithPrecalculatedSets, mode="simple", target_set_name=None, number_of_examples=3, agentId=-1, episode_steps=50, target_items=None, operators=[]):
        self.pipeline = pipeline
        self.mode = mode
        self.target_set_name = target_set_name
        self.number_of_examples = number_of_examples
        self.episode_steps = episode_steps
        self.agentId = agentId
        self.systemRandom = random.SystemRandom()
        self.exploration_dimensions = self.pipeline.exploration_columns
        self.target_set_index = -1
        if self.target_set_name != None and target_items == None:
            with open(f"./rl/targets/{self.target_set_name}.json") as f:
                self.state_encoder = StateEncoder(
                    pipeline, target_items=set(json.load(f)))
        elif self.mode == "scattered" and target_items == None:
            self.state_encoder = StateEncoder(pipeline,  target_items=TargetSetGenerator.get_diverse_target_set(
                number_of_samples=100), target_set_size=2000)
        elif self.mode == "concentrated":
            self.state_encoder = StateEncoder(pipeline)
        else:
            self.state_encoder = StateEncoder(
                pipeline, target_items=target_items)

        self.action_manager = ActionManager(pipeline, operators=operators)
        self.set_action_space = spaces.Discrete(
            self.pipeline.discrete_categories_count)
        self.operation_action_space = spaces.Discrete(
            len(self.action_manager.set_action_types))

        self.set_state_dim = self.pipeline.discrete_categories_count * \
            len(self.state_encoder.set_description)
        self.operation_state_dim = len(self.state_encoder.set_description)
        if self.mode == "by_example":
            self.set_state_dim += self.number_of_examples * \
                len(self.pipeline.exploration_columns)
            self.operation_state_dim += self.number_of_examples * \
                len(self.pipeline.exploration_columns)

    def reset(self):
        self.step_count = 0
        self.datasets = []
        self.input_set = None
        if self.mode == "by_example":
            self.get_target_set_and_examples()
        if self.mode == "concentrated":
            self.state_encoder = StateEncoder(
                self.pipeline, target_items=TargetSetGenerator.get_concentrated_target_set(), target_set_size=200)
            self.datasets = self.get_contentrated_start_datasets()
        self.sets_viewed = set()
        self.set_review_counter = 0
        self.state_encoder.reset()
        self.set_state, reward = self.get_set_state()
        self.operation_counter = {
            "by_superset": 0,
            "by_distribution": 0,
            "by_facet": 0,
            "by_neighbors": 0
        }

        self.episode_info = []
        return self.set_state

    def get_contentrated_start_datasets(self):
        while True:
            examples = random.choices(
                list(self.state_encoder.target_items), k=3)
            galaxies = self.pipeline.initial_collection[self.pipeline.initial_collection["galaxies.objID"].isin(
                examples)]
            dataset = Dataset()
            common_columns = []
            for column in self.pipeline.exploration_columns:
                if galaxies[column].nunique() == 1:
                    common_columns.append(column)
                    dataset.predicate.append(PredicateItem(
                        column, "==", galaxies.iloc[0][column], is_category=True))
            if len(dataset.predicate.components) > 4:
                break
        column_to_split_on = random.choice(common_columns)
        dataset.predicate.remove_attribute(column_to_split_on)
        self.pipeline.reload_set_data(dataset, apply_predicate=True)

        return self.pipeline.by_facet(dataset, attributes=[column_to_split_on], number_of_groups=self.pipeline.discrete_categories_count)

    def fix_possible_operation_action_probs(self, set_index, probs):
        if len(self.datasets) == 0:
            dataset = self.pipeline.get_dataset()
        else:
            dataset = self.datasets[set_index]
        return self.action_manager.fix_possible_operation_action_probs(dataset, probs)

    def fix_possible_set_action_probs(self, probs):
        return self.action_manager.fix_possible_set_action_probs(self.datasets, probs)

    def get_target_set_and_examples(self):
        available_target_sets = ["green-peas",
                                 "hiis", "lcgs", "low-metallicity-bcds"]
        self.target_set_name = random.choice(available_target_sets)
        self.target_set_index = available_target_sets.index(
            self.target_set_name)
        with open(f"./rl/targets/{self.target_set_name}.json") as f:
            self.initial_target_items = set(json.load(f))
        example_ids = random.choices(
            list(self.initial_target_items), k=self.number_of_examples)
        self.example_state = []
        for id in example_ids:
            item = self.pipeline.initial_collection.loc[
                self.pipeline.initial_collection["galaxies.objID"] == id].iloc[0]
            for column in self.pipeline.exploration_columns:
                self.example_state.append(
                    self.pipeline.ordered_dimensions[column].index(str(item[column])))

    def get_operation_state(self, set_index):
        if len(self.datasets) == 0:
            dataset = self.pipeline.get_dataset()
        else:
            dataset = self.datasets[set_index]
        state = []
        if self.mode == 'by_example':
            state += self.example_state

        encoded_set, reward = self.state_encoder.encode_dataset(
            dataset, get_reward=False)
        state += encoded_set
        return np.array(state)

    def get_set_state(self):
        encoded_sets, reward = self.state_encoder.encode_datasets(
            datasets=self.datasets)
        state = []

        if self.mode == 'by_example':
            state += self.example_state

        return np.array(state + encoded_sets), reward

    def render(self):
        i = 1

    def step(self, set_index, operation_index=-1):
        self.step_count += 1
        original_datasets = self.datasets
        original_input_set = self.input_set
        if len(self.datasets) == 0:
            self.input_set = self.pipeline.get_dataset()
        else:
            self.input_set = self.datasets[set_index]
        set_action_array = self.action_manager.set_action_types[operation_index].split(
            '-&-')
        self.operation_counter[set_action_array[0]] += 1
        if set_action_array[0] == "by_superset":
            self.datasets = self.pipeline.by_superset(
                dataset=self.input_set)
        elif set_action_array[0] == "by_distribution":
            self.datasets = self.pipeline.by_distribution(
                dataset=self.input_set)
        elif set_action_array[0] == "by_facet":
            self.datasets = self.pipeline.by_facet(dataset=self.input_set, attributes=[
                set_action_array[1]], number_of_groups=10)
        else:
            self.datasets = self.pipeline.by_neighbors(dataset=self.input_set, attributes=[
                set_action_array[1]])

        self.datasets = [
            d for d in self.datasets if d.set_id != None and d.set_id >= 0]

        if not isinstance(self.datasets, list):
            self.datasets = [self.datasets]

        if len(self.datasets) == 0:
            reward = 0
            print(
                f"Agent: {self.agentId} Operation: {self.step_count} Set index: {set_index} Action: {set_action_array} set id: {self.input_set.set_id} No more sets!!!!!!")
            self.datasets = original_datasets
            # self.input_set = original_input_set
        else:
            result_set_ids = set(map(lambda x: x.set_id, self.datasets))
            self.set_review_counter += len(result_set_ids & self.sets_viewed)
            self.sets_viewed.update(result_set_ids)
            self.set_state, reward = self.get_set_state()

            print(
                f"Agent: {self.agentId} Operation: {self.step_count}  Set index: {set_index} Action: {set_action_array} set id: {self.input_set.set_id} Reward: {reward}")

        self.episode_info.append({
            "input_set_index": set_index,
            "input_set_size": len(self.input_set.data),
            "input_set_id": self.input_set.set_id if self.input_set.set_id != None else -1,
            "operator": set_action_array[0],
            "parameter": set_action_array[1] if len(set_action_array) > 1 else None,
            "output_set_count": len(self.datasets),
            "output_set_average_size": sum(map(lambda x: len(x.data), self.datasets))/len(self.datasets),
            "reward": reward,
            "sets_viewed": len(self.sets_viewed),
            "sets_reviewed": self.set_review_counter
        })
        done = self.step_count == self.episode_steps
        set_op_id = f"{self.input_set.set_id}:{set_action_array}" if self.input_set != None else f"-1:{set_action_array}"
        return self.set_state, reward, done, set_op_id
