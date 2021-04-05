import argparse
import os
import json
import random

import numpy as np

from rl.A3C_2_actors.intrinsic_curiosity_model import IntrinsicCuriosityForwardModel
from rl.A3C_2_actors.operation_actor import OperationActor
from rl.A3C_2_actors.pipeline_environment import PipelineEnvironment
from rl.A3C_2_actors.set_actor import SetActor
from rl.A3C_2_actors.target_set_generator import TargetSetGenerator
from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str,
                    default="scattered-ccur-0.75-lstm-5-alr-3e-05-clr-3e-05-03082021_182550")

args = parser.parse_args()


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


model_path = "/home/aurelien/all_op_models"


class AgentRunner:
    def __init__(self, mode, curiosity_weight):
        data_folder = "./app/data/"

        self.episode_steps = 251
        self.agent_name = args.name
        self.steps = args.lstm_steps
        self.pipeline = PipelineWithPrecalculatedSets(
            "sdss", ["galaxies"], data_folder=data_folder, discrete_categories_count=10, min_set_size=10, exploration_columns=["galaxies.u", "galaxies.g", "galaxies.r", "galaxies.i", "galaxies.z", "galaxies.petroRad_r", "galaxies.redshift"])
        self.env = PipelineEnvironment(
            self.pipeline,  target_set_name=args.target_set, mode=args.mode, episode_steps=self.episode_steps, operators=["by_facet", "by_superset", "by_neighbors", "by_distribution"])

        self.set_state_dim = self.env.set_state_dim
        self.operation_state_dim = self.env.operation_state_dim

        self.set_action_dim = self.env.set_action_space.n
        self.operation_action_dim = self.env.operation_action_space.n
        self.set_actor = SetActor(
            self.set_state_dim, self.set_action_dim, self.steps, args.actor_lr, self.agent_name, model_path=f"{model_path}/{curiosity_weight}/set_actor")
        self.operation_actor = OperationActor(
            self.operation_state_dim, self.operation_action_dim, self.steps, args.actor_lr, self.agent_name, model_path=f"{model_path}/{curiosity_weight}/operation_actor")
        if os.path.exists(f"{model_path}/{curiosity_weight}/set_op_counters.json"):
            with open(f"{model_path}/{curiosity_weight}/set_op_counters.json") as f:
                self.set_op_counters = json.load(f)

        self.counter_curiosity_factor = 100/250

    def run(self, times=1):
        results = []
        for i in range(times):
            print(f"---------------------    RUN: {i}")
            done = False
            set_action_steps = [[-1] * self.set_state_dim] * self.steps
            operation_action_steps = [
                [-1] * self.operation_state_dim] * self.steps
            set_state = self.env.reset()
            curiosity_rewards = []
            while not done:
                probs = self.set_actor.model.predict(
                    np.array(set_action_steps).reshape((1, self.steps, self.set_state_dim)))
                probs = self.env.fix_possible_set_action_probs(probs[0])
                if all(np.isnan(x) for x in probs):
                    set_action = 0
                else:
                    set_action = np.random.choice(
                        self.set_action_dim, p=probs)

                operation_state = self.env.get_operation_state(set_action)
                operation_action_steps.pop(0)
                operation_action_steps.append(operation_state)
                probs = self.operation_actor.model.predict(
                    np.array(operation_action_steps).reshape((1, self.steps, self.operation_state_dim)))
                probs = self.env.fix_possible_operation_action_probs(
                    set_action, probs[0])
                if np.isnan(probs[0]):
                    operation_action = self.env.get_random_operation(
                        set_action)
                else:
                    operation_action = np.random.choice(
                        self.operation_action_dim, p=probs)

                next_set_state, reward, done, set_op_pair = self.env.step(
                    set_action, operation_action)
                if set_op_pair in self.set_op_counters:
                    self.set_op_counters[set_op_pair] += 1
                else:
                    self.set_op_counters[set_op_pair] = 1

                op_counter = self.set_op_counters[set_op_pair]
                next_set_action_steps = set_action_steps.copy()
                next_set_action_steps.pop(0)
                next_set_action_steps.append(next_set_state)
                curiosity_rewards.append({
                    "curiosity_reward": self.counter_curiosity_factor/op_counter
                })
            for i in range(len(self.env.episode_info)):
                self.env.episode_info[i].update(curiosity_rewards[i])
            results.append(self.env.episode_info)
        return results


mode = "Scattered"
curiosity_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
res = {}
for curiosity_weight in curiosity_weights:
    print(
        f"---------------------           LOADING: {mode} {curiosity_weight}")
    with open(f"{model_path}/{curiosity_weight}/info.json") as f:
        items = json.load(f)
        for key in items.keys():
            setattr(args, key, items[key])

        res[f"{mode}-{curiosity_weight}"] = AgentRunner(
            mode, curiosity_weight).run(10)
with open(f"./runs-data.json", 'w') as f:
    json.dump(res, f, indent=1, default=np_encoder)
