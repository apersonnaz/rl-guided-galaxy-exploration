from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
import numpy as np
from app.pipelines.dataset import Dataset

from typing import List
from scipy.stats import entropy

class StateEncoder:
    def __init__(self, pipeline: PipelineWithPrecalculatedSets, target_items=None):
        self.pipeline = pipeline
        self.target_ratio = 0.1
        self.target_max_reward = 100
        self.initial_target_items = target_items
        self.reward_multiplier = self.target_max_reward / \
            (len(self.initial_target_items)*self.target_ratio)

        self.set_description = ["item count"]
        for column in self.pipeline.exploration_columns:
            self.set_description.append(f"description {column}")
            self.set_description.append(f"distinct {column}")
            self.set_description.append(f"entropy {column}")

    def reset(self):
        self.ratio_item_dict = {}
        self.target_items = self.initial_target_items

    def encode_datasets(self, datasets: List[Dataset], get_reward=False):
        encoded_sets = []
        rewards = 0
        for index, dataset in enumerate(datasets):
            encoded_set, reward = self.encode_dataset(
                dataset, get_reward=get_reward)
            encoded_sets += encoded_set
            rewards += reward
        encoded_sets += [-1] * (10-len(datasets)) * \
            len(self.set_description)
        if get_reward:
            return encoded_sets, rewards
        else:
            return encoded_sets

    def encode_dataset(self, dataset: Dataset, get_reward=False):
        encoded_set = []
        encoded_set.append(len(dataset.data))
        reward = 0
        if get_reward:
            new_target_found_in_dataset = set(
                dataset.data["galaxies.objID"].to_list()) & self.target_items
            original_target_found_in_dataset = set(
                dataset.data["galaxies.objID"].to_list()) & self.initial_target_items
            reward_set_size_ratio = len(
                original_target_found_in_dataset)/len(dataset.data)
            if len(new_target_found_in_dataset) > 0:
                self.target_items = set([
                    t for t in self.target_items if t not in new_target_found_in_dataset])
                reward = len(new_target_found_in_dataset) * \
                    reward_set_size_ratio*self.reward_multiplier
                ratio_item_dict = dict(zip(new_target_found_in_dataset, [
                                       reward_set_size_ratio]*len(new_target_found_in_dataset)))
                self.ratio_item_dict.update(ratio_item_dict)

            old_target_found_in_dataset = original_target_found_in_dataset - \
                new_target_found_in_dataset
            better_ratio_items = list(filter(
                lambda x: x in old_target_found_in_dataset and self.ratio_item_dict[x] < reward_set_size_ratio, self.ratio_item_dict))
            if len(better_ratio_items) > 0:
                reward += len(better_ratio_items) * \
                    reward_set_size_ratio*self.reward_multiplier
                ratio_item_dict = dict(
                    zip(better_ratio_items, [reward_set_size_ratio]*len(better_ratio_items)))
                self.ratio_item_dict.update(ratio_item_dict)

        data = dataset.data
        for dimension in self.pipeline.exploration_columns:
            predicate_item = next((
                x for x in dataset.predicate.components if x.attribute == dimension), None)
            if predicate_item != None:
                encoded_set.append(
                    self.pipeline.ordered_dimensions[dimension].index(str(predicate_item.value)))
            else:
                encoded_set.append(-1)
            encoded_set.append(data[dimension].nunique())
            counts = data[dimension].value_counts()
            encoded_set.append(entropy(counts))

        if get_reward:
            return encoded_set, reward
        else:
            return encoded_set
