import numpy as np
from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets


class ActionManager:
    def __init__(self, pipeline: PipelineWithPrecalculatedSets, operators=[]):
        self.pipeline = pipeline
        self.set_action_types = []
        self.operators = operators
        if "by_superset" in operators:
            self.set_action_types.append("by_superset")
        if "by_distribution" in operators:
            self.set_action_types.append("by_distribution")
        if "by_facet" in operators:
            self.set_action_types += list(
                map(lambda x: f"by_facet-&-{x}", self.pipeline.exploration_columns))
        if "by_neighbors" in operators:
            self.set_action_types += list(
                map(lambda x: f"by_neighbors-&-{x}", self.pipeline.exploration_columns))
        self.set_action_dim = pipeline.discrete_categories_count
        self.operation_action_dim = len(self.set_action_types)

    def fix_possible_set_action_probs(self, datasets, probs):
        if len(datasets) == 0:
            return [np.nan]*self.pipeline.discrete_categories_count
        else:
            probs[len(datasets):] = [0] * \
                (len(probs) - len(datasets))
            total = sum(probs)
            return [float(i)/total for i in probs]

    def fix_possible_operation_action_probs(self, dataset, probs):
        for dimension in self.pipeline.exploration_columns:
            predicate_item = next((
                x for x in dataset.predicate.components if x.attribute == dimension), None)
            if predicate_item != None:
                probs[self.set_action_types.index(
                    "by_facet-&-" + dimension)] = 0
            elif "by_neighbors" in self.operators:
                probs[self.set_action_types.index(
                    "by_neighbors-&-" + dimension)] = 0
        if len(dataset.predicate.components) <= 1:
            probs[self.set_action_types.index("by_superset")] = 0
            if "by_distribution" in self.operators:
                probs[self.set_action_types.index("by_distribution")] = 0
        total = sum(probs)
        return [float(i)/total for i in probs]
