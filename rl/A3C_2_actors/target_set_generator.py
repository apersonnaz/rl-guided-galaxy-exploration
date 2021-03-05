import json
import random
from os import listdir

class TargetSetGenerator:
    @staticmethod
    def get_diverse_target_set(number_of_samples=10):
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