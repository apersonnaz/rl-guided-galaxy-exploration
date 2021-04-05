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

    @staticmethod
    def get_concentrated_target_set():
        target_files = listdir("./rl/targets/")
        while True:
            target_file = random.choice(target_files)
            with open("./rl/targets/"+target_file) as f:
                items = json.load(f)
            if len(items) > 50 and len(items) < 2000:
                print(target_file)
                break
        return set(items)
