from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class OperandType(str, Enum):
    Column = "Column"
    String = "String"
    Number = "Number"


class Operand(BaseModel):
    value: str
    type: OperandType


class JoinDefinition(BaseModel):
    table1: str
    attribute1: str
    table2: str
    attribute2: str

    class Config:
        schema_extra = {
            "example": {
                "table1": "projects",
                "attribute1": "unics_id",
                "table2": "project_members",
                "attribute2": "project"
            }
        }


class FilterDefinition(BaseModel):
    leftOperand: Operand
    rightOperand: Operand
    operator: str

    class Config:
        schema_extra = {
            "example": {
                "leftOperand": {
                    "value": "projects.start_year",
                    "type": "Column"
                },
                "rightOperand": {
                    "value": 2018,
                    "type": "Number"
                },
                "operator": "="
            }
        }


class DatabaseName(str, Enum):
    CORDIS = "unics_cordis"
    SDSS = "sdss"


class SetDefinition(BaseModel):
    tables: List[str] = Field(..., description="The list of tables joined")
    joinFilters: List[FilterDefinition] = Field(
        [], description="The join conditions")
    valueFilters: List[FilterDefinition] = Field(
        [], description="The filters defining the set")

    class Config:
        schema_extra = {
            "example": {
                "tables": ["projects", "project_topics", "topics"],
                "joinFilters": [
                    {
                        "leftOperand": {
                            "value": "project_topics.project",
                            "type": "Column"
                        },
                        "rightOperand": {
                            "value": "projects.unics_id",
                            "type": "Column"
                        },
                        "operator": "="
                    },
                    {
                        "leftOperand": {
                            "value": "project_topics.topic",
                            "type": "Column"
                        },
                        "rightOperand": {
                            "value": "topics.code",
                            "type": "Column"
                        },
                        "operator": "="
                    }
                ],
                "valueFilters": [
                    {
                        "leftOperand": {
                            "value": "projects.end_year",
                            "type": "Column"
                        },
                        "rightOperand": {
                            "value": 2020,
                            "type": "Number"
                        },
                        "operator": "="
                    },
                    {
                        "leftOperand": {
                            "value": "projects.framework_program",
                            "type": "Column"
                        },
                        "rightOperand": {
                            "value": "FP7",
                            "type": "String"
                        },
                        "operator": "="
                    }
                ]
            }
        }


class OperatorRequestBody(BaseModel):
    database: DatabaseName = Field(..., title="Database name",
                                   description="The name of the database to work on")
    inputSet: SetDefinition = Field(
        ..., description="The definition of the operator input set (parsed SQL query)")

    class Config:
        schema_extra = {
            "example": {
                "database":  "unics_cordis",
                "inputSet": {
                    "tables": ["projects", "project_topics", "topics"],
                    "joinFilters": [
                        {
                            "leftOperand": {
                                "value": "project_topics.project",
                                "type": "Column"
                            },
                            "rightOperand": {
                                "value": "projects.unics_id",
                                "type": "Column"
                            },
                            "operator": "="
                        },
                        {
                            "leftOperand": {
                                "value": "project_topics.topic",
                                "type": "Column"
                            },
                            "rightOperand": {
                                "value": "topics.code",
                                "type": "Column"
                            },
                            "operator": "="
                        }
                    ],
                    "valueFilters": [
                        {
                            "leftOperand": {
                                "value": "projects.end_year",
                                "type": "Column"
                            },
                            "rightOperand": {
                                "value": 2020,
                                "type": "Number"
                            },
                            "operator": "="
                        },
                        {
                            "leftOperand": {
                                "value": "projects.framework_program",
                                "type": "Column"
                            },
                            "rightOperand": {
                                "value": "FP7",
                                "type": "String"
                            },
                            "operator": "="
                        }
                    ]
                }
            }
        }


class ByFilterBody(OperatorRequestBody):
    filter: FilterDefinition = Field(...,
                                     description="The new filter to be applied")

    class Config:
        schema_extra = {
            "example": {
                "database": "unics_cordis",
                "inputSet": {
                    "tables": ["projects", "project_topics", "topics"],
                    "joinFilters": [
                        {
                            "leftOperand": {
                                "value": "project_topics.project",
                                "type": "Column"
                            },
                            "rightOperand": {
                                "value": "projects.unics_id",
                                "type": "Column"
                            },
                            "operator": "="
                        },
                        {
                            "leftOperand": {
                                "value": "project_topics.topic",
                                "type": "Column"
                            },
                            "rightOperand": {
                                "value": "topics.code",
                                "type": "Column"
                            },
                            "operator": "="
                        }
                    ],
                    "valueFilters": [
                        {
                            "leftOperand": {
                                "value": "projects.end_year",
                                "type": "Column"
                            },
                            "rightOperand": {
                                "value": 2020,
                                "type": "Number"
                            },
                            "operator": "="
                        }
                    ]
                },
                "filter": {
                    "leftOperand": {
                        "value": "projects.start_year",
                        "type": "Column"
                    },
                    "rightOperand": {
                        "value": 2016,
                        "type": "Number"
                    },
                    "operator": "="
                }
            }
        }


class ByFacetBody(OperatorRequestBody):
    attributes: List[str] = Field(
        ..., description="The list of attributes to group the set items by")
    numberOfFacets: int = Field(
        4, gt=0, description="The number of facets to be returned")

    class Config:
        schema_extra = {
            "example": {
                "database": "unics_cordis",
                "inputSet": {
                    "tables": ["projects", "project_topics", "topics"],
                    "joinFilters": [
                        {
                            "leftOperand": {
                                "value": "project_topics.project",
                                "type": "Column"
                            },
                            "rightOperand": {
                                "value": "projects.unics_id",
                                "type": "Column"
                            },
                            "operator": "="
                        },
                        {
                            "leftOperand": {
                                "value": "project_topics.topic",
                                "type": "Column"
                            },
                            "rightOperand": {
                                "value": "topics.code",
                                "type": "Column"
                            },
                            "operator": "="
                        }
                    ],
                    "valueFilters": [
                        {
                            "leftOperand": {
                                "value": "projects.end_year",
                                "type": "Column"
                            },
                            "rightOperand": {
                                "value": 2020,
                                "type": "Number"
                            },
                            "operator": "="
                        }
                    ]
                },
                "attributes": [
                    "projects.start_date",
                    "projects.ec_fund_scheme"
                ],
                "numberOfFacets": 4
            }
        }

class ByNeighborsBody(OperatorRequestBody):
    attributes: List[str] = Field(
        ..., description="The list of ordonned attributes to look for neighbor sets")
    class Config:
        schema_extra = {
            "example": {
                "database": "unics_cordis",
                "inputSet": {
                    "tables": ["projects", "project_topics", "topics"],
                    "joinFilters": [
                        {
                            "leftOperand": {
                                "value": "project_topics.project",
                                "type": "Column"
                            },
                            "rightOperand": {
                                "value": "projects.unics_id",
                                "type": "Column"
                            },
                            "operator": "="
                        },
                        {
                            "leftOperand": {
                                "value": "project_topics.topic",
                                "type": "Column"
                            },
                            "rightOperand": {
                                "value": "topics.code",
                                "type": "Column"
                            },
                            "operator": "="
                        }
                    ],
                    "valueFilters": [
                        {
                            "leftOperand": {
                                "value": "projects.end_year",
                                "type": "Column"
                            },
                            "rightOperand": {
                                "value": 2020,
                                "type": "Number"
                            },
                            "operator": "="
                        }
                    ]
                },
                "attributes": [
                    "projects.total_cost",
                    "projects.ec_fund_scheme"
                ]
            }
        }

class ByOverlapBody(OperatorRequestBody):
    numberOfSets: int = Field(
        4, gt=0, description="The number of facets to be returned")
    maxDuration: float = Field(
        5, gt=0, description="The maximum duration in seconds allowed to run the operation")

    class Config:
        schema_extra = {
            "example": {
                "database": "unics_cordis",
                "inputSet": {
                    "tables": ["projects", "project_topics", "topics"],
                    "joinFilters": [
                        {
                            "leftOperand": {
                                "value": "project_topics.project",
                                "type": "Column"
                            },
                            "rightOperand": {
                                "value": "projects.unics_id",
                                "type": "Column"
                            },
                            "operator": "="
                        },
                        {
                            "leftOperand": {
                                "value": "project_topics.topic",
                                "type": "Column"
                            },
                            "rightOperand": {
                                "value": "topics.code",
                                "type": "Column"
                            },
                            "operator": "="
                        }
                    ],
                    "valueFilters": [
                        {
                            "leftOperand": {
                                "value": "projects.end_year",
                                "type": "Column"
                            },
                            "rightOperand": {
                                "value": 2020,
                                "type": "Number"
                            },
                            "operator": "="
                        }
                    ]
                },
                "maxDuration": 5,
                "numberOfSets": 4
            }
        }


class ByJoinBody(OperatorRequestBody):
    joinedTables: List[str] = Field(
        ..., description="The list of tables to be joined")

    class Config:
        schema_extra = {
            "example": {
                "database": "unics_cordis",
                "inputSet": {
                    "tables": ["projects", "project_topics", "topics"],
                    "joinFilters": [
                            {
                                "leftOperand": {
                                    "value": "project_topics.project",
                                    "type": "Column"
                                },
                                "rightOperand": {
                                    "value": "projects.unics_id",
                                    "type": "Column"
                                },
                                "operator": "="
                            },
                        {
                                "leftOperand": {
                                    "value": "project_topics.topic",
                                    "type": "Column"
                                },
                                "rightOperand": {
                                    "value": "topics.code",
                                    "type": "Column"
                                },
                                "operator": "="
                            }
                    ],
                    "valueFilters": [
                        {
                            "leftOperand": {
                                "value": "projects.end_year",
                                "type": "Column"
                            },
                            "rightOperand": {
                                "value": 2020,
                                "type": "Number"
                            },
                            "operator": "="
                        }
                    ]
                },
                "joinedTables": ["project_members", "institutions"]
            }
        }


class OperatorRequestResponse(BaseModel):
    error: int = Field(
        0, description="The error status, 1 if an error has occurred, 0 otherwise")
    errorMsg: str = Field(None, description="The error message")
    payload: List[str] = Field(None,
                               description='The list of queries resulting of the operation')

    class Config:
        schema_extra = {
            "example": {
                "error": 0,
                "errorMsg": None,
                "payload": [
                    "select * from projects where projects.end_year = 2020 and projects.framework_program = 'H2020' and projects.start_year = 2018",
                    "select * from projects where projects.end_year = 2020 and projects.framework_program = 'H2020' and projects.start_year = 2015",
                    "select * from projects where projects.end_year = 2020 and projects.framework_program = 'H2020' and projects.start_year = 2017",
                ]
            }
        }
