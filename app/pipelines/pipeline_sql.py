
import collections
import os
import json
from typing import List
from operator import attrgetter
import psycopg2
import pandas as pd

from .dataset import Dataset
from .predicateitem import PredicateItem
from .tools.operator_logging import loggable_operator


class PipelineSql:
    databases = {
        "sdss": "skyserver_dr16_2020_11_30",
        "unics_cordis": "siris"
    }
    def __init__(self, database_name: str, data_folder="./data", discrete_categories_count=20):
        self.database_name = database_name
        self.discrete_categories_count = discrete_categories_count
        self.data_folder = data_folder
        self.database_folder = f"{self.data_folder}/{database_name}"
        self.sql_connection = psycopg2.connect(
            dbname=PipelineSql.databases[database_name], user='postgres', host='testbed.inode.igd.fraunhofer.de', password='vdS83DJSQz2xQ', port=18001)

        if os.path.isfile(f"{self.database_folder}/foreign_keys.csv"):
            self.foreign_keys = pd.read_csv(
                f"{self.database_folder}/foreign_keys.csv")

        all_columns_to_discretize = pd.read_csv(
            f"{self.database_folder}/columns_to_discretize.csv")
        self.columns_to_discretize = all_columns_to_discretize.apply(lambda column: f"{column.table}.{column.column}", axis=1).to_list()

        # Exploration dimensions selection
        self.ordered_dimensions = {}
        self.interval_indexes = {}
        with open(f"{self.database_folder}/bins.json") as bins_file:
            bins_data = json.load(bins_file)

        for column_name in bins_data:
            intervals = []
            if type(bins_data[column_name][0]) == list:
                if isinstance(bins_data[column_name][0][0], str):
                    for i in range(len(bins_data[column_name])):
                        bins_data[column_name][i] = list(map(lambda x:pd.Timestamp(x), bins_data[column_name][i]))    
                for bin_data in bins_data[column_name]:
                    intervals.append(pd.Interval(bin_data[0], bin_data[1]))
                self.interval_indexes[column_name] = pd.IntervalIndex(intervals)
            else:
                self.ordered_dimensions[column_name] = bins_data[column_name]


        print(self.interval_indexes)
        print(self.ordered_dimensions)

    @loggable_operator()
    def by_filter(self, dataset: Dataset, predicate_item: PredicateItem, logger=None):
        dataset.predicate.append(predicate_item)
        return dataset

    @loggable_operator()
    def by_facet(self, dataset: Dataset, attributes: List[str], number_of_groups: int, return_datasets=True, logger=None):
        attributes_possible_values = {}
        for attribute in attributes:
            if attribute in self.interval_indexes:
                res = dataset.get_data(self.sql_connection, [f"min({attribute})", f"max({attribute})"])
                bins = self.interval_indexes[attribute]
                bins_in_dataset_range = []
                for bin in bins:
                    if len(bins_in_dataset_range) == 0:
                        if res.iloc[0]["min"] >= bin.left  :
                            bins_in_dataset_range.append(bin)
                    else:
                        if bin.right <= res.iloc[0]["max"]:
                            bins_in_dataset_range.append(bin)
                attributes_possible_values[attribute] = bins_in_dataset_range
            else:
                res = dataset.get_data(self.sql_connection, [f"distinct {attribute} as \"{attribute}\" "], not_null_attribute=attribute)
                if len(res[attribute]) < 500:
                    attributes_possible_values[attribute] = res[attribute]
        
        possible_sets = [dataset]

        for attribute in attributes:
            new_possible_sets = []
            for set in possible_sets:
                for value in attributes_possible_values[attribute]:
                    new_set = set.copy()
                    new_set.predicate.append(PredicateItem(attribute, "==", value, is_category=type(value) == pd.Interval))
                    new_possible_sets.append(new_set)
            possible_sets = new_possible_sets

        for set in possible_sets:
            set.get_count(self.sql_connection)

        possible_sets.sort(key=attrgetter("count"), reverse=True)

        return possible_sets[:number_of_groups]

    @loggable_operator()
    def by_neighbors(self, dataset: Dataset, attributes: List[str], logger=None):
        if len(attributes) == 0:
            attributes = map(
                lambda item: item.attribute in self.interval_indexes, dataset.predicate.components) + map(
                lambda item: item.attribute in self.ordered_dimensions, dataset.predicate.components)

        results = []
        for attribute in attributes:
            if attribute in self.interval_indexes or attribute in self.ordered_dimensions:
                if attribute in self.interval_indexes:  
                    collection = self.interval_indexes
                else:
                    collection = self.ordered_dimensions
                predicate_item = next(
                    (item for item in dataset.predicate.components if item.attribute == attribute), None)
                if predicate_item != None:
                    index = list(collection[attribute]).index(
                        predicate_item.value)
                    if index != 0:
                        lower_set = dataset.copy()
                        predicate_item = next(
                            (item for item in lower_set.predicate.components if item.attribute == attribute), None)
                        predicate_item.value = collection[attribute][index - 1]    
                        results.append(lower_set)
                    if index + 1 != len(collection[attribute]):
                        higher_set = dataset.copy()
                        predicate_item = next(
                            (item for item in higher_set.predicate.components if item.attribute == attribute), None)
                        predicate_item.value = collection[attribute][index + 1]    
                        results.append(higher_set)
        return results if len(results) > 0 else [dataset]

    def calculate_overlap(self, set1: set, set2: set):
        return len(set1 & set2) / len(set1 | set2)

    def by_superset(self, dataset: Dataset, set_ids_to_ignore=[]):
        if len(dataset.predicate.components) == 0:
            return dataset
        else:
            original_count = dataset.get_count(self.sql_connection)
            smallest_item_count = None
            print(f"Original set item count: {original_count}")
            new_predicate = None
            new_set = None
            possible_sets = []
            result = dataset
            for i in range(len(dataset.predicate.components)):
                new_set = dataset.copy()
                new_set.predicate.components.pop(i)
                new_count = new_set.get_count(self.sql_connection)
                possible_sets.append(new_set)
                if (smallest_item_count == None or new_count < smallest_item_count ) and new_count != original_count:
                    result = new_set
                    smallest_item_count = new_count
            return result