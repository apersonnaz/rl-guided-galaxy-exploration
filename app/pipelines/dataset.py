from typing import List
import pandas as pd
import copy

from .predicateitem import JoinParameters, PredicateItemGroup


class Dataset:
    def __init__(self, set_id: int = None, data: pd.DataFrame = None, predicate: PredicateItemGroup = None, joins: List[JoinParameters] = None, tables = None):
        self.set_id: int = set_id
        self.data: pd.DataFrame = data
        self.predicate: PredicateItemGroup = predicate if predicate != None else PredicateItemGroup()
        self.joins: List[JoinParameters] = joins if joins != None else []
        self.tables = tables if tables != None else []
        self.count = None

    def copy(self):
        return copy.deepcopy(self) 

    def get_count(self, sql_connection):
        self.count = pd.read_sql(self.get_sql_query(attributes=["count(*)"]), sql_connection).iloc[0]["count"]
        return self.count

    def get_data(self, sql_connection, attributes=["*"], not_null_attribute=None):
        return pd.read_sql(self.get_sql_query(attributes=attributes, not_null_attribute=not_null_attribute), sql_connection)

    def get_sql_query(self, attributes=None, not_null_attribute=None):
        joined_tables = ', '.join(self.tables)
        if attributes != None:
            query = f"SELECT { ', '.join(attributes)} FROM {joined_tables}"
        elif self.data != None:
            query = f"SELECT { ', '.join(self.find_interesting_attributes())} FROM {joined_tables}"
        else:
            query = f"SELECT * FROM {joined_tables}"

        join_filters =list(map(lambda join: f"{join.left_attribute} = {join.right_attribute}", self.joins))
        if len(self.predicate.components) > 0 or len(join_filters) != 0 or not_null_attribute != None:
            if len(join_filters) == 0 and len(self.predicate.components) > 0:
                query += " WHERE "
            elif len(join_filters) != 0 :
                joined_join_filters = " AND ".join(join_filters)
                query += " WHERE " + joined_join_filters 
                if len(self.predicate.components) > 0:
                    query += " AND "
            query += self.predicate.to_sql()
            if not_null_attribute != None:
                query += f" AND {not_null_attribute} is not null"
        return query

    def find_interesting_attributes(self):
        interesting_attributes = []
        na_ratio = self.data.isna().sum()/len(self.data)
        density_ratio = self.data.nunique() / len(self.data)
        attributes = pd.DataFrame(
            data={"attribute": self.data.columns, "na_ratio": na_ratio, "density_ratio": density_ratio})
        attributes = attributes.sort_values(by=["density_ratio", "na_ratio"])

        interesting_attributes = attributes[(attributes.na_ratio < 0.3) & (
            attributes.density_ratio < 0.3)].attribute.tolist()
        if len(interesting_attributes) == 0:
            interesting_attributes = self.data.columns
        return interesting_attributes
       