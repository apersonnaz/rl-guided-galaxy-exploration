from typing import Union, List
import pandas as pd


class PredicateItem:
    def __init__(self, attribute: str, operator: str, value: Union[str, int, float, pd.Interval], is_category=False):
        self.attribute = attribute
        self.operator = operator
        self.is_multiple_line_string = type(value) == str and '\n' in value
        self.value = value
        self.is_category = is_category

    def __str__(self):
        if type(self.value) == list:
            return f"(`{self.attribute}` {self.operator} {self.value})"
        elif self.is_category:
            return f"(`{self.attribute}`.astype('str') {self.operator} '{self.value}')"
        elif self.operator == "contains":
            return f"(`{self.attribute}`.str.contains('{self.value}'))"
        else:
            return f"(`{self.attribute}` {self.operator} '{self.value}')"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.attribute == other.attribute and self.operator == other.operator and self.value == other.value
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_attributes(self):
        return [self.attribute]

    def to_sql(self):
        if self.is_category:
            if type(self.value) == str:
                interval = self.value.replace(
                    "(", "").replace("]", "").split(", ")
                return f"{self.attribute} > {interval[0]} and {self.attribute} <= {interval[1]}"
            else:
                if isinstance(self.value.left, pd.Timestamp):
                    return f"{self.attribute} > '{self.value.left}' and {self.attribute} <= '{self.value.right}'"
                else:
                    return f"{self.attribute} > {self.value.left} and {self.attribute} <= {self.value.right}"
        else:
            operator = "=" if self.operator == "==" else self.operator
            value = f"'{self.value}'" if type(
                self.value) == str else self.value
            return f"{self.attribute} {operator} {value}"


class PredicateItemGroup:
    def __init__(self, logical_operator="&"):
        self.components: List[PredicateItem] = []
        self.logical_operator = logical_operator

    def append(self, predicate):
        if type(predicate) == PredicateItemGroup:
            for component in predicate.components:
                if component not in self.components:
                    self.components.append(component)
        elif predicate not in self.components:
            self.components.append(predicate)

    def __str__(self):
        components = filter(
            lambda component: not component.is_multiple_line_string, self.components)
        return f"({self.logical_operator.join(map(lambda component: str(component), components))})"

    def to_sql(self):
        if self.logical_operator == "&":
            return f" and ".join(map(lambda component: component.to_sql(), self.components))
        else:
            return f" or ".join(map(lambda component: component.to_sql(), self.components))

    def get_attributes(self):
        attributes = []
        for component in self.components:
            for attribute in component.get_attributes():
                if attribute not in attributes:
                    attributes.append(attribute)
        return attributes

    def remove_attribute(self, attribute):
        for component in self.components.copy():
            attributes = component.get_attributes()
            if attribute in attributes:
                if len(attributes) == 1:
                    self.components.remove(component)
                else:
                    component.remove_attribute(attribute)

    def is_empty(self):
        return len(self.components) == 0
    
    def get_filter_values(self, attribute):
        return [x.value for x in self.components if x.attribute == attribute]

class JoinParameters:
    def __init__(self, target_collection_name: str, left_attribute: str, right_attribute: str, other_collection: str = ""):
        self.target_collection_name = target_collection_name
        self.other_collection = other_collection
        self.left_attribute = left_attribute
        self.right_attribute = right_attribute
