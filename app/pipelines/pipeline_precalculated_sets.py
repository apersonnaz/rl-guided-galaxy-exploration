import ast
import fnmatch
import os
import time
# from memory_profiler import profile
from datetime import datetime
from pprint import pprint

import inquirer
import numpy as np
import pandas as pd

from .dataset import Dataset
from .pipeline import Pipeline
from .predicateitem import JoinParameters, PredicateItem, PredicateItemGroup
from .tools.operator_logging import loggable_operator


class PipelineWithPrecalculatedSets(Pipeline):
    def __init__(self, database_name, initial_collection_names, min_set_size=10, data_folder="./data", discrete_categories_count=20, exploration_columns=[]):
        Pipeline.__init__(self,
                          database_name,
                          initial_collection_names, data_folder=data_folder, discrete_categories_count=discrete_categories_count, exploration_columns=exploration_columns)
        self.min_set_size = min_set_size
        self.index_folder = f"{self.data_folder}/{database_name}/{'_'.join(sorted(self.initial_collection_names))}_index"
        self.groups = pd.read_csv(
            f"{self.index_folder}/groups.csv",
            converters={
                "definition": ast.literal_eval
            }, index_col="id"
        )
        # self.index = pd.read_csv(
        #     f"{self.index_folder}/index.csv",
        #     index_col=["group1", "group2"],
        # )
        self.correspondences = pd.read_csv(
            f"{self.index_folder}/correspondences.csv")

    @loggable_operator()
    def by_filter(self, dataset, predicate_item, logger=None):
        dataset = Pipeline.by_filter(self, dataset, predicate_item)
        if not dataset.data.empty:
            dataset.set_id = self.find_group_id(dataset)
        return dataset

    @loggable_operator()
    def by_facet(self, dataset, attributes, number_of_groups, return_datasets=True, logger=None):
        if return_datasets:
            small_dataset_counter = 0
            datasets = Pipeline.by_facet(
                self, dataset, attributes, number_of_groups, return_datasets)
            for dataset in datasets:
                if len(dataset.data) < self.min_set_size:
                    small_dataset_counter -= 1
                    dataset.set_id = small_dataset_counter
                else:
                    dataset.set_id = self.find_group_id(dataset)
            return datasets
        else:
            return Pipeline.by_facet(self, dataset, attributes, number_of_groups, return_datasets)

    @loggable_operator()
    def by_neighbors(self, dataset, attributes, logger=None):
        datasets = super().by_neighbors(dataset, attributes)
        small_dataset_counter = 0
        for dataset in datasets:
            if len(dataset.data) < self.min_set_size:
                small_dataset_counter -= 1
                dataset.set_id = small_dataset_counter
            else:
                dataset.set_id = self.find_group_id(dataset)
        return datasets

    def select_facet(self, dataset, attributes, values):
        dataset = Pipeline.select_facet(self, dataset, attributes, values)
        dataset.set_id = self.find_group_id(dataset)
        return dataset

    def reload_set_data(self, dataset, apply_joins=False, apply_predicate=False, initialize_data=True):
        dataset = Pipeline.reload_set_data(
            self, dataset, apply_joins, apply_predicate)
        try:
            dataset.set_id = self.find_group_id(dataset)
        except IndexError:
            dataset.set_id = None
        return dataset

    @loggable_operator()
    def by_superset(self, dataset, set_ids_to_ignore=[], logger=None):
        small_dataset_counter = 0
        datasets = Pipeline.by_superset(
            self, dataset)
        for dataset in datasets:
            if len(dataset.data) < self.min_set_size:
                small_dataset_counter -= 1
                dataset.set_id = small_dataset_counter
            else:
                dataset.set_id = self.find_group_id(dataset)
        return datasets
        
        # if len(dataset.predicate.components) == 1:
        #     print("No superset was found!")
        #     return dataset
        # try:
        #     current_set = self.groups.loc[dataset.set_id]
        #     overlapping_groups_index = self.index.loc[
        #         dataset.set_id].sort_values(by="overlap", ascending=False)
        #     overlapping_group_ids = [id for id in list(
        #         overlapping_groups_index.index.values) if not id in set_ids_to_ignore]
        #     overlapping_groups = self.groups.iloc[overlapping_group_ids]
        #     # current_group.members = ast.literal_eval(current_group.members)
        #     new_group = None
        #     new_group_index = None
        #     for index, group in overlapping_groups.iterrows():
        #         # group.members = ast.literal_eval(group.members)
        #         if index != dataset.set_id and len(group.members & current_set.members) == len(
        #                 current_set.members):
        #             new_group = group
        #             new_group_index = index
        #             break

        #     if new_group_index == None:
        #         print("No superset was found!")
        #     else:
        #         print(
        #             f"Original set item count: {len(current_set.members)} new set item count: {len(new_group.members)}")
        #         dataset = self.get_groups_as_datasets(
        #             group_ids=[new_group_index])[0]
        # except (IndexError, KeyError):
        #     print(
        #         "Superset impossible: No set matching the current predicate was found"
        #     )
        # return dataset

    @loggable_operator()
    def by_overlap(self, dataset, number_of_groups=3, max_seconds=1, return_datasets=True, logger=None):
        try:
            overlapping_groups_index = self.index.loc[
                dataset.set_id].sort_values(by="overlap", ascending=True)
            overlapping_group_ids = list(overlapping_groups_index.index.values)
            overlapping_groups = self.groups.iloc[overlapping_group_ids]
            if len(overlapping_groups) <= number_of_groups:
                selected_groups = overlapping_groups
            else:
                startTime = datetime.now()
                overlapping_groups = pd.merge(
                    overlapping_groups,
                    overlapping_groups_index,
                    right_on="group2",
                    left_index=True,
                    sort=False,
                )
                selected_groups_ids = list(
                    overlapping_groups.index.values[0:number_of_groups])
                interesting_indexes = self.index.loc[(selected_groups_ids,
                                                      selected_groups_ids), :]
                selection_overlap_sum = 0
                if len(interesting_indexes) == 0:
                    overlapping_groups.loc[selected_groups_ids,
                                           "overlap_to_selection"] = 0
                else:
                    interesting_indexes = interesting_indexes.reindex(
                        selected_groups_ids, level="group2")
                    for id in selected_groups_ids:
                        other_selected_group_ids = selected_groups_ids.copy()
                        other_selected_group_ids.remove(id)
                        overlap_to_selection = interesting_indexes.loc[(
                            id, other_selected_group_ids), "overlap"].sum() if id in interesting_indexes.index else 0
                        overlapping_groups.loc[
                            id, "overlap_to_selection"] = overlap_to_selection
                        selection_overlap_sum += overlap_to_selection

                sets_counter = 0
                set_replacements_counter = 0
                for (unselected_group_id, unselected_group) in overlapping_groups.iterrows():
                    if not unselected_group_id in selected_groups_ids:
                        if (unselected_group.overlap > 0.1 or
                            (datetime.now() - startTime).total_seconds() >
                                max_seconds or selection_overlap_sum == 0):
                            break
                        sets_counter += 1
                        # Overlap indexes for the current unselected set
                        # indexes = self.index.loc[(unselected_group_id, selected_groups_ids), "overlap"].reindex(
                        #     selected_groups_ids, level="group2")
                        # Option below much faster than option above !
                        indexes = self.index.loc[unselected_group_id].reindex(
                            overlapping_group_ids)
                        # Loop on the selected sets to test replacement
                        for selected_group_id in selected_groups_ids:
                            other_selected_group_ids = selected_groups_ids.copy()
                            other_selected_group_ids.remove(selected_group_id)
                            overlap_to_selection_if_replaced = indexes.loc[
                                other_selected_group_ids, "overlap"].sum() if unselected_group_id in indexes.index else 0
                            overlapping_groups.loc[unselected_group_id,
                                                   "overlap_to_selection"] = overlap_to_selection_if_replaced
                            if (overlap_to_selection_if_replaced
                                    < overlapping_groups.loc[selected_group_id, "overlap_to_selection"]):
                                set_replacements_counter += 1
                                other_selected_group_ids.append(
                                    unselected_group_id)
                                selected_groups_ids = other_selected_group_ids
                                # Other selected sets overlap re-calculation
                                selection_overlap_sum = overlap_to_selection_if_replaced
                                interesting_indexes = self.index.loc[
                                    selected_groups_ids, :]
                                if len(interesting_indexes) == 0:
                                    overlapping_groups.loc[selected_groups_ids,
                                                           "overlap_to_selection"] = 0
                                else:
                                    interesting_indexes = interesting_indexes.reindex(
                                        selected_groups_ids, level="group2")
                                    for id in selected_groups_ids:
                                        if id != unselected_group_id:
                                            other_selected_group_ids = selected_groups_ids.copy()
                                            other_selected_group_ids.remove(id)
                                            overlap_to_selection = interesting_indexes.loc[
                                                (id, other_selected_group_ids), "overlap"].sum() if id in interesting_indexes.index else 0
                                            overlapping_groups.loc[id,
                                                                   "overlap_to_selection"] = overlap_to_selection
                                            selection_overlap_sum += overlap_to_selection
                                break
                print(
                    f"sets studied: {sets_counter} time spent: {(datetime.now() - startTime).total_seconds()}s sets replaced:{set_replacements_counter}"
                )
                # print(selected_groups_ids)
                # print(overlapping_groups.loc[selected_groups_ids])

                if return_datasets == False:
                    return overlapping_groups.loc[selected_groups_ids]
                else:
                    return self.get_groups_as_datasets(selected_groups_ids)
        except IndexError:
            print(
                "Overlap impossible: No set matching the current predicate was found"
            )

    @loggable_operator()
    def by_distribution(self, dataset, return_datasets=True, logger=None):
        if return_datasets:
            small_dataset_counter = 0
            datasets = Pipeline.by_distribution(
                self, dataset, return_datasets)
            for dataset in datasets:
                if len(dataset.data) < self.min_set_size:
                    small_dataset_counter -= 1
                    dataset.set_id = small_dataset_counter
                else:
                    dataset.set_id = self.find_group_id(dataset)
            return datasets
        else:
            return Pipeline.by_distribution(self, dataset, return_datasets)

 
    def attribute_value_selection(self, dataset, reverse_selectivities):
        set_selectivity_list = self.get_selectivity_list(
            dataset, ascending=reverse_selectivities)
        # We take the three with highest selectivity (or lowest if by dissim)
        selected_attribute_values = set_selectivity_list.iloc[0:3].apply(
            lambda x: (x.attribute, x.value), axis=1)
        print("Selected a=v: ")
        for (attribute_value_index, attribute_value) in set_selectivity_list.iloc[0:3].iterrows():
            print(
                f"  {attribute_value.attribute}={attribute_value.value} Selectivity: {attribute_value.selectivity}")
        return selected_attribute_values

    @loggable_operator()
    def by_subset(self, datasets, number_of_sets=3, minimum_set_size=10, logger=None):
        sets_with_info = []
        for dataset in datasets:
            if len(dataset.data) > minimum_set_size:
                sets_with_info.append({
                    "dataset": dataset,
                    "dataset_sum": len(
                        dataset.data[dataset.data.label == 1]) - len(dataset.data[dataset.data.label == 0]),
                    "dataset_size": len(dataset.data)
                })
        sets_with_info = sorted(sets_with_info, key=lambda i: (
            i["dataset_sum"], i["dataset_size"]), reverse=True)
        sets_with_info = sets_with_info[:number_of_sets]
        return (set_with_info["dataset"] for set_with_info in sets_with_info)

    @loggable_operator()
    def get_selectivity_list(self, dataset, attributes=None, ascending=False, logger=None):
        if attributes is None:
            attributes = self.interesting_attributes
        set_selectivity_list = pd.DataFrame(
            columns=["attribute", "value", "selectivity"])
        for attribute in attributes:
            values = (dataset.data.loc[:, attribute].value_counts(
                normalize=True, dropna=False).to_frame().reset_index())
            values.rename(
                columns={
                    attribute: "selectivity",
                    "index": "value"
                },
                inplace=True,
            )
            values.loc[:, "attribute"] = attribute
            values = values[values.selectivity != 0]
            set_selectivity_list = set_selectivity_list.append(
                values, ignore_index=True, sort=False)
        set_selectivity_list = set_selectivity_list.sort_values(
            by="selectivity", ascending=ascending)
        return set_selectivity_list

    def get_groups_as_datasets(self, group_ids):
        datasets = []
        for group_id in group_ids:
            group = self.groups.loc[group_id]
            dataset = Dataset(set_id=group_id)
            for element_id in group.definition:
                element = self.correspondences.loc[self.correspondences.id ==
                                                   element_id].iloc[0]
                value = float(element.value) if self.initial_collection[element.column].dtype == np.dtype(
                    float) else element.value
                dataset.predicate.append(
                    PredicateItem(
                        element.column,
                        "==",
                        value,
                        is_category=str(
                            self.initial_collection[element.column].dtype) == "category",
                    ))
            datasets.append(self.reload_set_data(
                dataset, apply_joins=True, apply_predicate=True))
        return datasets

    def find_group_id(self, dataset):
        if len(dataset.data)< self.min_set_size:
            return None
        try:
            group_definition = []
            for predicate_item in dataset.predicate.components:
                group_definition.append(
                    int(self.correspondences.loc[
                        (self.correspondences.value.astype("str") ==
                         str(predicate_item.value))
                        & (self.correspondences.column == predicate_item.attribute
                           )].iloc[0].id))
            group_definition = set(group_definition)
            groups = self.groups.index[self.groups.definition ==
                                       group_definition]
            group_id = None
            if len(groups) == 1:
                group_id = self.groups.index[self.groups.definition ==
                                             group_definition][0]
                # group = self.groups.iloc[group_id]
            elif len(groups) > 1:
                current_set_item_count = len(dataset.data)
                group_id = self.groups.index[
                    (group_definition - self.groups.definition == set())
                    &
                    (self.groups.member_count == current_set_item_count)][0]
            else:
                same_length_groups = self.groups[self.groups.member_count == len(dataset.data)]
                groups =  same_length_groups[same_length_groups.definition.apply(lambda x: group_definition.issubset(x))]
                if len(groups) == 1:
                    group_id = groups.index[0]
                # group = self.groups.iloc[group_id]

            return group_id
        except Exception as error:
            return None
