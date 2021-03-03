import gym
import numpy as np
import json
import random
from gym import spaces
import random
from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
from os import listdir
from .state_encoder import StateEncoder

class PipelineEnvironment(gym.Env):
    def __init__(self, pipeline: PipelineWithPrecalculatedSets, mode="simple", target_set_name=None, number_of_examples=3, agentId=-1, episode_steps=50, target_items=None):
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
                self.state_encoder = StateEncoder(pipeline, target_items=set(json.load(f)))
        elif self.mode == "scattered" and target_items == None:
            self.state_encoder = StateEncoder(pipeline,  target_items=self.get_diverse_target_set(
                number_of_samples=100))    
        else:
            self.state_encoder = StateEncoder(pipeline, target_items=target_items)

        self.set_action_types = ["by_superset",
                                 "by_distribution"]
        self.set_action_types += list(
            map(lambda x: f"by_facet-&-{x}", self.pipeline.exploration_columns))
        self.set_action_types += list(
            map(lambda x: f"by_neighbors-&-{x}", self.pipeline.exploration_columns))

        self.set_action_space = spaces.Discrete(self.pipeline.discrete_categories_count)
        self.operation_action_space = spaces.Discrete(
            len(self.set_action_types))

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

        return self.set_state

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

    def get_diverse_target_set(self, number_of_samples=10):
        self.target_set_index = -1
        initial_target_items = []
        for file in listdir("./rl/targets/"):
            with open("./rl/targets/"+file) as f:
                items = json.load(f)
                if len(items) > number_of_samples:
                    initial_target_items += random.choices(
                        items, k=number_of_samples)
                else:
                    initial_target_items += items
        return set(initial_target_items)

    def get_operation_state(self, set_index):
        if len(self.datasets) == 0:
            dataset = self.pipeline.get_dataset()
        else:
            dataset = self.datasets[set_index]
        state = []
        if self.mode == 'by_example':
            state += self.example_state

        encoded_set = self.state_encoder.encode_dataset(dataset)
        state += encoded_set
        return np.array(state)

    def get_set_state(self):
        encoded_sets, reward = self.state_encoder.encode_datasets(datasets=self.datasets, get_reward=True)
        state = []

        if self.mode == 'by_example':
            state += self.example_state

        return np.array(state + encoded_sets), reward

    def render(self):
        i = 1

    def fix_possible_set_action_probs(self, probs):
        if len(self.datasets) == 0:
            return [np.nan]*self.pipeline.discrete_categories_count
        else:
            probs[len(self.datasets):] = [0] * \
                (len(probs) - len(self.datasets))
            return [float(i)/sum(probs) for i in probs]

    def fix_possible_operation_action_probs(self, probs, set_index):
        if len(self.datasets) == 0:
            dataset = self.pipeline.get_dataset()
        else:
            dataset = self.datasets[set_index]
        for dimension in self.exploration_dimensions:
            predicate_item = next((
                x for x in dataset.predicate.components if x.attribute == dimension), None)
            if predicate_item != None:
                probs[self.set_action_types.index(
                    "by_facet-&-" + dimension)] = 0
            else:
               probs[self.set_action_types.index(
                   "by_neighbors-&-" + dimension)] = 0
        if len(dataset.predicate.components) <= 1:
            probs[self.set_action_types.index("by_superset")] = 0
        return [float(i)/sum(probs) for i in probs]


    def step(self, set_index, operation_index=-1):
        self.step_count += 1
        original_datasets = self.datasets
        original_input_set = self.input_set
        if len(self.datasets) == 0:
            self.input_set = self.pipeline.get_dataset()
        else:
            self.input_set = self.datasets[set_index]
        set_action_array = self.set_action_types[operation_index].split(
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
                f"Agent: {self.agentId} Set index: {set_index} Action: {set_action_array} set id: {self.input_set.set_id} No more sets!!!!!!")
            self.datasets = original_datasets
            self.input_set = original_input_set
        else:
            result_set_ids = set(map(lambda x: x.set_id, self.datasets))
            self.set_review_counter += len(result_set_ids & self.sets_viewed)
            self.sets_viewed.update(result_set_ids)
            self.set_state, reward = self.get_set_state()

            print(
                f"Agent: {self.agentId} Set index: {set_index} Action: {set_action_array} set id: {self.input_set.set_id} Reward: {reward}")

        done = self.step_count == self.episode_steps
        set_op_id = f"{self.input_set.set_id}:{set_action_array}" if self.input_set != None else f"-1:{set_action_array}"
        return self.set_state, reward, done, set_op_id
