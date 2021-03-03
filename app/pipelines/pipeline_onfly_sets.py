import ast
import copy
import fnmatch
import os
import time
from datetime import datetime
from pprint import pprint

import inquirer
import numpy as np
import pandas as pd

from .dataset import Dataset
from .pipeline import Pipeline
from .predicateitem import JoinParameters, PredicateItem, PredicateItemGroup


class PipelineWithOnFlySets(Pipeline):
    def __init__(self, database_name, initial_collection_names, data_folder="./data"):
        Pipeline.__init__(self, database_name,
                          initial_collection_names, data_folder=data_folder)

    def by_superset(self, dataset: Dataset, set_ids_to_ignore=[]):
        if len(dataset.predicate.components) == 0:
            return dataset
        else:
            original_count = len(dataset.data)
            print(f"Original set item count: {original_count}")
            smallest_item_count = len(self.initial_collection)
            new_predicate = None
            new_set = None
            predicate_attributes = dataset.predicate.get_attributes()
            if len(predicate_attributes) == 1:
                raise Exception(
                    "Superset impossible, only one attribute is filtered")
            else:
                for attribute in predicate_attributes:
                    new_dataset = Dataset(predicate=copy.deepcopy(
                        dataset.predicate), joins=copy.deepcopy(dataset.joins))
                    new_dataset.predicate.remove_attribute(attribute)
                    self.reload_set_data(
                        new_dataset, apply_joins=True, apply_predicate=True)
                    new_count = len(new_dataset.data)
                    # if new_count == original_count:
                    #     dataset.predicate = new_predicate
                    # el
                    if new_count < smallest_item_count and new_count != original_count:
                        new_predicate = new_predicate
                        new_set = new_dataset
                        smallest_item_count = new_count
                print(f"New set item count: {smallest_item_count}")
                print(f"New set predicate: {str(new_set.predicate)}")
                return new_set

#         Boucler sur la structure de données, et à chaque fois enlever un attribut/valeur, lancer la requête sur D, et garder à chaque fois le plus petit D généré.
# Si A est donné, construire un prédicat avec les valeurs de A et l’attribut le plus sélectif, le lancer sur D.

    def by_overlap(self, dataset, number_of_groups=3, max_seconds=1, return_datasets=True, logger=None):
        startTime = datetime.now()
        current_set = set(dataset.data[dataset.data.columns[0]])
        set_selectivity_list = pd.DataFrame(
            columns=['attribute1', 'value1', 'attribute2', 'value2', 'attribute3', 'value3'])
        # , 'attribute4', 'value4', 'attribute5', 'value5', 'selectivity']
        interesting_attributes = self.find_interesting_attributes(
            dataset.data, count=10)
        second_attributes = interesting_attributes.copy()
        for attribute in interesting_attributes:
            values = dataset.data.loc[:, attribute].value_counts(
                normalize=True, dropna=False).to_frame().reset_index()
            values.rename(
                columns={attribute: 'selectivity', 'index': 'value1'}, inplace=True)
            if len(values) == 0:
                second_attributes.remove(attribute)
            else:
                values.loc[:, 'attribute1'] = attribute
                set_selectivity_list = set_selectivity_list.append(
                    values, ignore_index=True, sort=False)
                # groups of two attributes
                second_attributes.remove(attribute)
                third_attributes = second_attributes.copy()
                for second_attribute in second_attributes:
                    groups = dataset.data.groupby(by=[attribute, second_attribute], sort=False, observed=True).size(
                    ).reset_index().rename(columns={0: 'selectivity', attribute: "value1", second_attribute: "value2"})
                    if len(groups) == 0:
                        third_attributes.remove(second_attribute)
                    else:
                        groups.loc[:, 'attribute1'] = attribute
                        groups.loc[:, 'attribute2'] = second_attribute
                        groups.selectivity = groups.selectivity / \
                            len(dataset.data)
                        set_selectivity_list = set_selectivity_list.append(
                            groups, ignore_index=True, sort=False)
                        # third_attributes.remove(second_attribute)
                        # # fourth_attributes = third_attributes.copy()
                        # for third_attribute in third_attributes:
                        #     groups = dataset.data.groupby(by=[attribute, second_attribute, third_attribute], sort=False, observed=True).size(
                        #     ).reset_index().rename(columns={0: 'selectivity', attribute: "value1", second_attribute: "value2", third_attribute: "value3"})
                        #     if len(groups) > 0:
                        #         groups.loc[:, 'attribute1'] = attribute
                        #         groups.loc[:, 'attribute2'] = second_attribute
                        #         groups.loc[:, 'attribute3'] = third_attribute
                        #         groups.selectivity = groups.selectivity / \
                        #             len(dataset.data)
                        #         set_selectivity_list = set_selectivity_list.append(
                        #             groups, ignore_index=True, sort=False)
                    # fourth_attributes.remove(third_attribute)
                    # fifth_attributes = fourth_attributes.copy()
                    # for fourth_attribute in fourth_attributes:
                    #     groups = dataset.data.groupby(by=[attribute, second_attribute, third_attribute, fourth_attribute], sort=False).size(
                    #     ).reset_index().rename(columns={0: 'selectivity', attribute: "value1", second_attribute: "value2", third_attribute: "value3", fourth_attribute: "value4"})
                    #     groups.loc[:, 'attribute1'] = attribute
                    #     groups.loc[:, 'attribute2'] = second_attribute
                    #     groups.loc[:, 'attribute3'] = third_attribute
                    #     groups.loc[:, 'attribute4'] = fourth_attribute
                    #     groups.selectivity = groups.selectivity / \
                    #         len(dataset.data)
                    #     set_selectivity_list = set_selectivity_list.append(
                    #         groups, ignore_index=True, sort=False)
                    #     fifth_attributes.remove(fourth_attribute)
                    #     for fifth_attribute in fifth_attributes:
                    #         groups = dataset.data.groupby(by=[attribute, second_attribute, third_attribute, fourth_attribute, fifth_attribute], sort=False).size(
                    #         ).reset_index().rename(columns={0: 'selectivity', attribute: "value1", second_attribute: "value2", third_attribute: "value3", fourth_attribute: "value4", fifth_attribute: "value5"})
                    #         groups.loc[:, 'attribute1'] = attribute
                    #         groups.loc[:, 'attribute2'] = second_attribute
                    #         groups.loc[:, 'attribute3'] = third_attribute
                    #         groups.loc[:, 'attribute4'] = fourth_attribute
                    #         groups.loc[:, 'attribute5'] = fifth_attribute
                    #         groups.selectivity = groups.selectivity / \
                    #             len(dataset.data)
                    #         set_selectivity_list = set_selectivity_list.append(
                    #             groups, ignore_index=True, sort=False)

        set_selectivity_list = set_selectivity_list[set_selectivity_list.selectivity != 0].sort_values(
            by="selectivity")

        # selection of the k first sets
        item_sets = {}
        selected_set_selectivity_ids = list()
        for index, set_selectivity in set_selectivity_list.iterrows():
            # if type(set_selectivity.attribute5) == str:
            #     selectivity_set = set(
            #         self.initial_collection[(self.initial_collection[set_selectivity.attribute1] == set_selectivity.value1) &
            #                                 (self.initial_collection[set_selectivity.attribute2] == set_selectivity.value2) &
            #                                 (self.initial_collection[set_selectivity.attribute3] == set_selectivity.value3) &
            #                                 (self.initial_collection[set_selectivity.attribute4] == set_selectivity.value4) &
            #                                 (self.initial_collection[set_selectivity.attribute5] == set_selectivity.value5)].unics_id)
            # elif type(set_selectivity.attribute4) == str:
            #     selectivity_set = set(
            #         self.initial_collection[(self.initial_collection[set_selectivity.attribute1] == set_selectivity.value1) &
            #                                 (self.initial_collection[set_selectivity.attribute2] == set_selectivity.value2) &
            #                                 (self.initial_collection[set_selectivity.attribute3] == set_selectivity.value3) &
            #                                 (self.initial_collection[set_selectivity.attribute4] == set_selectivity.value4)].unics_id)
            # el
            if type(set_selectivity.attribute3) == str:
                selectivity_set = set(
                    self.initial_collection[(self.initial_collection[set_selectivity.attribute1] == set_selectivity.value1) &
                                            (self.initial_collection[set_selectivity.attribute2] == set_selectivity.value2) &
                                            (self.initial_collection[set_selectivity.attribute3] == set_selectivity.value3)][dataset.data.columns[0]])
            elif type(set_selectivity.attribute2) == str:
                selectivity_set = set(
                    self.initial_collection[(self.initial_collection[set_selectivity.attribute1] == set_selectivity.value1) &
                                            (self.initial_collection[set_selectivity.attribute2] == set_selectivity.value2)][dataset.data.columns[0]])
            else:
                selectivity_set = set(
                    self.initial_collection[self.initial_collection[set_selectivity.attribute1] == set_selectivity.value1][dataset.data.columns[0]])
            if len(selectivity_set) >= 10:
                item_sets[index] = selectivity_set
                selected_set_selectivity_ids.append(index)
                if len(selected_set_selectivity_ids) == number_of_groups:
                    break
        # overlap of the k first sets
        sets_overlap = {}
        sets_overlap_to_current_set = {}
        for id in selected_set_selectivity_ids:
            set_to_calculate = item_sets[id]
            sets_overlap_to_current_set[id] = self.calculate_overlap(
                current_set, set_to_calculate)
            overlap = 0
            for other_id in selected_set_selectivity_ids:
                if other_id != id:
                    other_set = item_sets[other_id]
                    overlap += self.calculate_overlap(
                        set_to_calculate, other_set)
            sets_overlap[id] = overlap
        # print(attribute_value_overlaps)
        sets_counter = 0
        set_replacements_counter = 0
        sets_replaced_by_overlap_to_s = 0
        for index, set_selectivity in set_selectivity_list.iterrows():
            if not index in selected_set_selectivity_ids:
                # if type(set_selectivity.attribute5) == str:
                #     new_set = set(
                #         self.initial_collection[(self.initial_collection[set_selectivity.attribute1] == set_selectivity.value1) &
                #                                 (self.initial_collection[set_selectivity.attribute2] == set_selectivity.value2) &
                #                                 (self.initial_collection[set_selectivity.attribute3] == set_selectivity.value3) &
                #                                 (self.initial_collection[set_selectivity.attribute4] == set_selectivity.value4) &
                #                                 (self.initial_collection[set_selectivity.attribute5] == set_selectivity.value5)].unics_id)
                # elif type(set_selectivity.attribute4) == str:
                #     new_set = set(
                #         self.initial_collection[(self.initial_collection[set_selectivity.attribute1] == set_selectivity.value1) &
                #                                 (self.initial_collection[set_selectivity.attribute2] == set_selectivity.value2) &
                #                                 (self.initial_collection[set_selectivity.attribute3] == set_selectivity.value3) &
                #                                 (self.initial_collection[set_selectivity.attribute4] == set_selectivity.value4)].unics_id)
                # el
                if type(set_selectivity.attribute3) == str:
                    new_set = set(
                        self.initial_collection[(self.initial_collection[set_selectivity.attribute1] == set_selectivity.value1) &
                                                (self.initial_collection[set_selectivity.attribute2] == set_selectivity.value2) &
                                                (self.initial_collection[set_selectivity.attribute3] == set_selectivity.value3)][dataset.data.columns[0]])
                elif type(set_selectivity.attribute2) == str:
                    new_set = set(
                        self.initial_collection[(self.initial_collection[set_selectivity.attribute1] == set_selectivity.value1) &
                                                (self.initial_collection[set_selectivity.attribute2] == set_selectivity.value2)][dataset.data.columns[0]])
                else:
                    new_set = set(
                        self.initial_collection[self.initial_collection[set_selectivity.attribute1] == set_selectivity.value1][dataset.data.columns[0]])
                if (datetime.now() - startTime).total_seconds() > max_seconds:
                    break
                if len(new_set) >= 10:
                    item_sets[index] = new_set
                    sets_overlap_to_current_set[index] = self.calculate_overlap(
                        current_set, new_set)
                    if sets_overlap_to_current_set[index] > 0.1:
                        break
                    sets_counter += 1
                    for id_to_replace in selected_set_selectivity_ids:
                        overlap_if_replaced = 0
                        for other_id in selected_set_selectivity_ids:
                            if other_id != id_to_replace:
                                overlap_if_replaced += self.calculate_overlap(
                                    new_set, item_sets[other_id])

                        if (overlap_if_replaced < sets_overlap[id_to_replace]) or ((overlap_if_replaced == sets_overlap[id_to_replace]) and
                                                                                   (sets_overlap_to_current_set[index] < sets_overlap_to_current_set[id_to_replace])):
                            set_replacements_counter += 1
                            if ((overlap_if_replaced == sets_overlap[id_to_replace]) and
                                    (sets_overlap_to_current_set[index] < sets_overlap_to_current_set[id_to_replace])):
                                sets_replaced_by_overlap_to_s += 1
                            selected_set_selectivity_ids.remove(id_to_replace)
                            selected_set_selectivity_ids.append(index)
                            sets_overlap[index] = overlap_if_replaced
                            # print("replaced")
                            break

        print(
            f"sets studied: {sets_counter} time spent: {(datetime.now() - startTime).total_seconds()}s sets replaced:{set_replacements_counter} replacement by overlap to S: {sets_replaced_by_overlap_to_s}")

        for id in selected_set_selectivity_ids:
            set_selectivity = set_selectivity_list.loc[id]
            # if type(set_selectivity.attribute5) == str:
            #     definition_string = f"{set_selectivity.attribute1}={set_selectivity.value1} {set_selectivity.attribute2}={set_selectivity.value2} {set_selectivity.attribute3}={set_selectivity.value3} {set_selectivity.attribute4}={set_selectivity.value4} {set_selectivity.attribute5}={set_selectivity.value5}"
            # elif type(set_selectivity.attribute4) == str:
            #     definition_string = f"{set_selectivity.attribute1}={set_selectivity.value1} {set_selectivity.attribute2}={set_selectivity.value2} {set_selectivity.attribute3}={set_selectivity.value3} {set_selectivity.attribute4}={set_selectivity.value4}"
            # el
            if type(set_selectivity.attribute3) == str:
                definition_string = f"{set_selectivity.attribute1}={set_selectivity.value1} {set_selectivity.attribute2}={set_selectivity.value2} {set_selectivity.attribute3}={set_selectivity.value3}"
            elif type(set_selectivity.attribute2) == str:
                definition_string = f"{set_selectivity.attribute1}={set_selectivity.value1} {set_selectivity.attribute2}={set_selectivity.value2}"
            else:
                definition_string = f"{set_selectivity.attribute1}={set_selectivity.value1}"
            definition_string += f"   overlap to original set: {sets_overlap_to_current_set[id]:.4f}   overlap to other sets: {sets_overlap[id]:.4f}   items:{len(item_sets[id])}"
            print(definition_string)
        results = []
        for index, set_selectivity in set_selectivity_list.loc[selected_set_selectivity_ids].iterrows():
            new_set = Dataset(joins=copy.deepcopy(dataset.joins))
            new_set.predicate.append(PredicateItem(set_selectivity.attribute1, "==", set_selectivity.value1, is_category=str(
                dataset.data[set_selectivity.attribute1].dtype) == "category"))
            if type(set_selectivity.attribute2) == str:
                new_set.predicate.append(PredicateItem(set_selectivity.attribute2, "==", set_selectivity.value2, is_category=str(
                    dataset.data[set_selectivity.attribute2].dtype) == "category"))
            if type(set_selectivity.attribute3) == str:
                new_set.predicate.append(PredicateItem(set_selectivity.attribute3, "==", set_selectivity.value3, is_category=str(
                    dataset.data[set_selectivity.attribute3].dtype) == "category"))
            self.reload_set_data(new_set, apply_joins=True,
                                 apply_predicate=True)
            results.append(new_set)
            # if type(set_selectivity.attribute4) == str:
            #     new_set.predicate.append(PredicateItem(set_selectivity.attribute4, "==", set_selectivity.value4, is_category=str(
            #         dataset.data[set_selectivity.attribute4].dtype) == "category"))
            # if type(set_selectivity.attribute5) == str:
            #     new_set.predicate.append(PredicateItem(set_selectivity.attribute5, "==", set_selectivity.value5, is_category=str(
            #         dataset.data[set_selectivity.attribute5].dtype) == "category"))

        return results
        # for id in selected_attribute_value_ids:
        #     print(attribute_values_selectivity.loc[id])
        #     print(
        #         f"Overlap to current set: {attribute_value_overlaps_to_current_set[id]:.4f}")
        #     print(f"overlap to other sets: {attribute_value_overlaps[id]:.4f}")

#         Pour les sets à la volée, il s’agit de savoir quels prédicats évaluer. S est accompagné d’une structure de données au format (attribut,valeur,%) et triée par
# ordre décroissant de sélectivité (de moins en moins sélectif).
# Lancer sur D les k attributs les plus sélectifs dans S, calculer l’overlap entre les k sets obtenus et appliquer Greedy en considérant le
# k+1ème attribut le plus sélectif de S et en bouclant tant que l’overlap avec S n’augmente pas au delà du seuil.
