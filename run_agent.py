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

parser.add_argument('--name', type=str, default="scattered-ccur-0.75-lstm-5-alr-3e-05-clr-3e-05-03082021_182550")

args = parser.parse_args()
with open("./saved_models/"+args.name+"/info.json") as f:
    items = json.load(f)
    for key in items.keys():
        setattr(args, key, items[key])

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


class AgentRunner:
    def __init__(self):
        data_folder = "./app/data/"
        self.episode_steps = 250
        self.agent_name=args.name
        self.steps = args.lstm_steps
        self.pipeline = PipelineWithPrecalculatedSets(
            "sdss", ["galaxies"], data_folder=data_folder, discrete_categories_count=10, min_set_size=10, exploration_columns=["galaxies.u", "galaxies.g", "galaxies.r", "galaxies.i", "galaxies.z", "galaxies.petroRad_r", "galaxies.redshift"])
        self.env = PipelineEnvironment(
            self.pipeline,  target_set_name=args.target_set, mode=args.mode, episode_steps=self.episode_steps)

        self.set_state_dim = self.env.set_state_dim
        self.operation_state_dim = self.env.operation_state_dim

        self.set_action_dim = self.env.set_action_space.n
        self.operation_action_dim = self.env.operation_action_space.n
        self.set_actor = SetActor(
            self.set_state_dim, self.set_action_dim, self.steps, args.actor_lr,self.agent_name, model_path="./saved_models/scattered-ccur-0.75-lstm-3-alr-3e-05-clr-3e-05-03072021_090203/set_actor/200")
        self.operation_actor = OperationActor(
            self.operation_state_dim, self.operation_action_dim, self.steps, args.actor_lr, self.agent_name, model_path="./saved_models/scattered-ccur-0.75-lstm-3-alr-3e-05-clr-3e-05-03072021_090203/operation_actor/200") 
        self.set_op_counters = {}

    def run(self):
        done = False
        set_action_steps = [[-1] * self.set_state_dim] * self.steps
        operation_action_steps = [
            [-1] * self.operation_state_dim] * self.steps
        set_state = self.env.reset()
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
            # if set_op_pair in episode_set_op_counters:
            #     episode_set_op_counters[set_op_pair] += 1
            # else:
            #     episode_set_op_counters[set_op_pair] = 1
            next_set_action_steps = set_action_steps.copy()
            next_set_action_steps.pop(0)
            next_set_action_steps.append(next_set_state)

        with open( f"./{args.name}-run.json", 'w') as f:
            json.dump( self.env.episode_info, f, indent=1, default=np_encoder)
AgentRunner().run()