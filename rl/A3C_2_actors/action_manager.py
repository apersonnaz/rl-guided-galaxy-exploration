import numpy as np
from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets

class ActionManager:
    def __init__(self, pipeline: PipelineWithPrecalculatedSets):
        self.pipeline = pipeline
        self.set_action_types = ["by_superset",
                                 "by_distribution"]
        self.set_action_types += list(
            map(lambda x: f"by_facet-&-{x}", self.pipeline.exploration_columns))
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
                total= sum(probs)
                return [float(i)/total for i in probs]

    def fix_possible_operation_action_probs(self, dataset, probs):
        for dimension in self.pipeline.exploration_columns:
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
            probs[self.set_action_types.index("by_distribution")] = 0
        total= sum(probs)
        return [float(i)/total for i in probs]
