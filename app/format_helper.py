from collections import defaultdict
from typing import List

import pandas as pd

from .models import FilterDefinition, OperandType, SetDefinition
from .pipelines.dataset import Dataset
from .pipelines.pipeline import Pipeline
from .pipelines.predicateitem import (JoinParameters, PredicateItem)


class FormatHelper:

    @classmethod
    def get_interval_predicate_item(cls, attribute: str, filters: List[FilterDefinition], pipeline: Pipeline):
        low_value=None
        high_value = None
        for filter_definition in filters:
            if filter_definition.operator in ["<", "<="]:
                
                if filter_definition.leftOperand.type == OperandType.Column:
                    if filter_definition.rightOperand.type == OperandType.Number:
                        high_value = float(filter_definition.rightOperand.value) 
                    else:
                        high_value = pd.Timestamp(filter_definition.rightOperand.value)
                else:
                    if filter_definition.leftOperand.type == OperandType.Number:
                        low_value = float(filter_definition.leftOperand.value)
                    else:
                        low_value = pd.Timestamp(filter_definition.leftOperand.value)
            else:
                if filter_definition.leftOperand.type == OperandType.Column:
                    if filter_definition.rightOperand.type == OperandType.Number:
                        low_value = float(filter_definition.rightOperand.value) 
                    else:
                        low_value = pd.Timestamp(filter_definition.rightOperand.value)
                else:
                    if filter_definition.leftOperand.type == OperandType.Number:
                        high_value = float(filter_definition.leftOperand.value)
                    else:
                        high_value = pd.Timestamp(filter_definition.leftOperand.value)

        categories = pipeline.interval_indexes[attribute]
        if low_value != None:
            for index, category in enumerate(categories):
                if category.left >= low_value or index == len(categories)-1:
                    return PredicateItem(attribute=attribute, operator="==", value=category, is_category=True)
        else:
            if high_value < categories[0].right:
                return PredicateItem(attribute=attribute, operator="==", value=categories[0], is_category=True)

            for category in categories.sort_values(ascending=False):
                if category.right <= high_value:
                    return PredicateItem(attribute=attribute, operator="==", value=category, is_category=True)

    @ classmethod
    def get_predicate_item(cls, filter_definition: FilterDefinition):
        attribute = None
        value = None
        operator = filter_definition.operator
        if operator == '=':
            operator = '=='
        if filter_definition.leftOperand.type == OperandType.Column:
            attribute = filter_definition.leftOperand.value
            value = float(
                filter_definition.rightOperand.value) if filter_definition.rightOperand.type == OperandType.Number else filter_definition.rightOperand.value
        else:
            attribute = filter_definition.rightOperand.value
            value = float(
                filter_definition.leftOperand.value) if filter_definition.leftOperand.type == OperandType.Number else filter_definition.leftOperand.value
        return PredicateItem(attribute, operator, value)

    @ classmethod
    def get_dataset(cls, pipeline: Pipeline, set_definition: SetDefinition = None):
        if set_definition == None:
            return pipeline.get_dataset()
        else:
            dataset = Dataset(tables=set_definition.tables)
            
            if len(dataset.tables) > 1:
                joined_tables = [dataset.tables[0]]
                tables = dataset.tables[1:]
                for table in tables:
                    relations = pipeline.foreign_keys[(
                        (pipeline.foreign_keys["table1"].isin(joined_tables)) & (pipeline.foreign_keys["table2"] == table)) | ((
                            pipeline.foreign_keys["table2"].isin(joined_tables)) & (pipeline.foreign_keys["table1"] == table))]

                    if len(relations) == 1:
                        relation = relations.iloc[0]
                        joined_tables.append(table)
                        if table == relation.table1:
                            dataset.joins.append(JoinParameters(target_collection_name=relation.table1,
                                                                left_attribute=f"{relation.table2}.{relation.attribute2}", right_attribute=f"{relation.table1}.{relation.attribute1}", other_collection=relation.table2))
                        else:
                            dataset.joins.append(JoinParameters(target_collection_name=relation.table2,
                                                                left_attribute=f"{relation.table1}.{relation.attribute1}", right_attribute=f"{relation.table2}.{relation.attribute2}", other_collection=relation.table1))
                    elif len(relations) == 0:
                        if tables.index(table) == len(tables) - 1:
                            raise Exception(
                                "Unable to join " + table)
                        tables.append(table)
                    else:
                        raise Exception(
                            "Multiple relations possible to join " + table)
            # pipeline.reload_set_data(
            #     dataset, apply_joins=True, apply_predicate=False)

            filters_by_attributes = defaultdict(list)
            for filter_definition in set_definition.valueFilters:
                if filter_definition.leftOperand.type == OperandType.Column:
                    filters_by_attributes[filter_definition.leftOperand.value].append(
                        filter_definition)
                else:
                    filters_by_attributes[filter_definition.rightOperand.value].append(
                        filter_definition)

            for attribute, filter_list in filters_by_attributes.items():
                if len(filter_list) == 1:
                    filter_definition = filter_list[0]
                    if attribute in pipeline.interval_indexes:
                        dataset.predicate.append(cls.get_interval_predicate_item(
                            attribute=attribute, filters=filter_list, pipeline=pipeline))
                    else:
                        if filter_definition.operator in ["<", "<=", ">", ">="]:
                            filter_definition.operator = "="
                        dataset.predicate.append(
                            cls.get_predicate_item(filter_definition))
                else:
                    dataset.predicate.append(cls.get_interval_predicate_item(
                        attribute=attribute, filters=filter_list, pipeline=pipeline))

            return dataset #pipeline.reload_set_data(dataset, apply_joins=False, apply_predicate=True, initialize_data=False)

    @ classmethod
    def get_sql_query(cls, pipeline: Pipeline, dataset: Dataset):
        tables = [pipeline.initial_collection_names[0]] + list(map(lambda x: x.target_collection_name, pipeline.initial_joins)) + list(
            map(lambda x: x.target_collection_name, dataset.joins))
        joined_tables = ', '.join(tables)
        query = f"SELECT { ', '.join(Pipeline.find_interesting_attributes(dataset.data))} FROM {joined_tables}"

        if len(dataset.predicate.components) > 0:
            join_filters = list(map(lambda join: f"{join.left_attribute} = {join.right_attribute}", pipeline.initial_joins)) + list(
                map(lambda join: f"{join.left_attribute} = {join.right_attribute}", dataset.joins))
            if len(join_filters) == 0:
                query += " WHERE "
            else:
                joined_join_filters = " AND ".join(join_filters)
                query += " WHERE " + joined_join_filters + " AND "
            query += dataset.predicate.to_sql()
        return query
