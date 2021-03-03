import traceback
from typing import List
import json

from fastapi import Body, Depends, FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from .format_helper import FormatHelper
from .models import (ByFacetBody, ByFilterBody, ByJoinBody, ByOverlapBody, ByNeighborsBody,
                     DatabaseName, OperatorRequestBody,
                     OperatorRequestResponse, SetDefinition)
from .pipelines.pipeline import Pipeline
from .pipelines.pipeline_precalculated_sets import \
    PipelineWithPrecalculatedSets
# from .pipelines.pipeline_sql import PipelineSql
from .pipelines.predicateitem import JoinParameters
from .galaxy_methods import *

app = FastAPI(title="CNRS pipelines API",
              description="API providing access to the CNRS pipelines operators",
              version="1.0.0",)

data_folder = "./app/data/"
database_pipeline_cache = {}
database_pipeline_cache["galaxies"] = PipelineWithPrecalculatedSets(
    "sdss", ["galaxies"], data_folder=data_folder, discrete_categories_count=10, min_set_size=10, exploration_columns=["galaxies.u", "galaxies.g", "galaxies.r", "galaxies.i", "galaxies.z", "galaxies.petroRad_r", "galaxies.redshift"])
# database_pipeline_cache["unics_cordis"] = PipelineSql(
#     "unics_cordis", data_folder=data_folder, discrete_categories_count=10)
# database_pipeline_cache["sdss"] = PipelineSql(
#     "sdss", data_folder=data_folder, discrete_categories_count=10)

app.mount("/test", StaticFiles(directory="test"), name="test")


def getPipeline(request: OperatorRequestBody):
    return database_pipeline_cache[request.database]


@app.put("/operators/by-filter",
         description="Returns the input set filtered by the provided attribute=value",
         response_model=OperatorRequestResponse,
         tags=["operators"])
async def by_filter(requestBody: ByFilterBody):
    try:
        print(requestBody.json())
        pipeline: Pipeline = getPipeline(requestBody)
        dataset = FormatHelper.get_dataset(pipeline, requestBody.inputSet)
        attribute = requestBody.filter.leftOperand.value
        if attribute in pipeline.interval_indexes:
            result_set = pipeline.by_filter(
                dataset=dataset, predicate_item=FormatHelper.get_interval_predicate_item(attribute=attribute, filters=[requestBody.filter], pipeline=pipeline))
        else:
            result_set = pipeline.by_filter(
                dataset=dataset, predicate_item=FormatHelper.get_predicate_item(requestBody.filter))

        return OperatorRequestResponse(payload=[result_set.get_sql_query()])
    except Exception as error:
        print(error)
        print(requestBody.json())
        traceback.print_tb(error.__traceback__)
        return OperatorRequestResponse(error=1, errorMsg=str(error))


@app.put("/operators/by-facet",
         description="Groups the input set items by a list of provided attributes and returns the n biggest resulting sets",
         response_model=OperatorRequestResponse,
         tags=["operators"])
async def by_facet(requestBody: ByFacetBody):
    try:
        print(requestBody.json())
        pipeline: Pipeline = getPipeline(requestBody)
        dataset = FormatHelper.get_dataset(pipeline, requestBody.inputSet)
        result_sets = pipeline.by_facet(
            dataset=dataset, attributes=requestBody.attributes, number_of_groups=requestBody.numberOfFacets)
        return OperatorRequestResponse(payload=list(map(lambda x: x.get_sql_query(), result_sets)))
    except Exception as error:
        print(error)
        print(requestBody.json())
        traceback.print_tb(error.__traceback__)
        return OperatorRequestResponse(error=1, errorMsg=str(error))


@app.put("/operators/by-neighbors",
         description="Gets the lower and higher sets on a list of ordonned attributes.",
         response_model=OperatorRequestResponse,
         tags=["operators"])
async def by_neighbors(requestBody: ByNeighborsBody):
    try:
        print(requestBody.json())
        pipeline: Pipeline = getPipeline(requestBody)
        dataset = FormatHelper.get_dataset(pipeline, requestBody.inputSet)
        result_sets = pipeline.by_neighbors(
            dataset=dataset, attributes=requestBody.attributes)
        return OperatorRequestResponse(payload=list(map(lambda x: x.get_sql_query(), result_sets)))
    except Exception as error:
        print(error)
        print(requestBody.json())
        traceback.print_tb(error.__traceback__)
        return OperatorRequestResponse(error=1, errorMsg=str(error))


@app.put("/operators/by-join",
         description="Returns the input set joined with one or more tables identified by the provided table names",
         response_model=OperatorRequestResponse,
         tags=["operators"])
async def by_join(requestBody: ByJoinBody):
    try:
        print(requestBody.json())
        pipeline: Pipeline = getPipeline(requestBody)
        dataset = FormatHelper.get_dataset(pipeline, requestBody.inputSet)
        joined_tables = pipeline.initial_collection_names.copy()
        for table in requestBody.joinedTables:
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
                if requestBody.joinedTables.index(table) == len(requestBody.joinedTables) - 1:
                    raise Exception(
                        "Unable to join " + table)
                requestBody.joinedTables.append(table)
            else:
                raise Exception(
                    "Multiple relations possible to join " + table)
        pipeline.reload_set_data(
            dataset, apply_joins=True, apply_predicate=True)
        # joined_tables = list(
        #     map(lambda x: x.target_collection_name, dataset.joins))
        # joined_tables += pipeline.initial_collection_names
        # if requestBody.join.table2 in joined_tables and not requestBody.join.table1 in joined_tables:
        #     result_set = pipeline.by_join(
        #         dataset=dataset,
        #         target_collection_name=requestBody.join.table1,
        #         left_attribute=f"{requestBody.join.table2}.{requestBody.join.attribute2}",
        #         right_attribute=f"{requestBody.join.table1}.{requestBody.join.attribute1}",
        #         other_collection=requestBody.join.table2)
        # elif requestBody.join.table1 in joined_tables and not requestBody.join.table2 in joined_tables:
        #     result_set = pipeline.by_join(
        #         dataset=dataset,
        #         target_collection_name=requestBody.join.table2,
        #         left_attribute=f"{requestBody.join.table1}.{requestBody.join.attribute1}",
        #         right_attribute=f"{requestBody.join.table2}.{requestBody.join.attribute2}",
        #         other_collection=requestBody.join.table1)
        # elif requestBody.join.table1 in joined_tables and requestBody.join.table2 in joined_tables:
        #     raise Exception("The table is already joined")
        # else:
        #     raise Exception(
        #         "The relation does not match a table of the input set.")
        return OperatorRequestResponse(payload=[FormatHelper.get_sql_query(pipeline, dataset)])
    except Exception as error:
        print(error)
        print(requestBody.json())
        traceback.print_tb(error.__traceback__)
        return OperatorRequestResponse(error=1, errorMsg=str(error))


@app.put("/operators/by-superset",
         description="Returns the smallest set completely overlapping with the input set",
         response_model=OperatorRequestResponse,
         tags=["operators"])
async def by_superset(requestBody: OperatorRequestBody):
    try:
        print(requestBody.json())
        pipeline: Pipeline = getPipeline(requestBody)
        dataset = FormatHelper.get_dataset(pipeline, requestBody.inputSet)
        result_set = pipeline.by_superset(
            dataset=dataset)
        return OperatorRequestResponse(payload=[result_set.get_sql_query()])
    except Exception as error:
        print(error)
        print(requestBody.json())
        traceback.print_tb(error.__traceback__)
        return OperatorRequestResponse(error=1, errorMsg=str(error))


@app.put("/operators/by-distribution",
         description="Return a list of sets with similar description values distribution (for ordered desctiption attributes)",
         response_model=OperatorRequestResponse,
         tags=["operators"])
async def by_distribution(requestBody: OperatorRequestBody):
    try:
        print(requestBody.json())
        pipeline: Pipeline = getPipeline(requestBody)
        dataset = FormatHelper.get_dataset(pipeline, requestBody.inputSet)
        result_sets = pipeline.by_distribution(
            dataset=dataset)
        return OperatorRequestResponse(payload=list(map(lambda x: x.get_sql_query(), result_sets)))
    except Exception as error:
        print(error)
        print(requestBody.json())
        traceback.print_tb(error.__traceback__)
        return OperatorRequestResponse(error=1, errorMsg=str(error))


@app.put("/operators/by-overlap",
         description="Returns n neighbouring sets slightly overlapping with the input set and minimizing the overlap to each other",
         response_model=OperatorRequestResponse,
         tags=["operators"])
async def by_overlap(requestBody: ByOverlapBody):
    try:
        print(requestBody.json())
        pipeline: Pipeline = getPipeline(requestBody)
        dataset = FormatHelper.get_dataset(pipeline, requestBody.inputSet)
        result_sets = pipeline.by_overlap(
            dataset=dataset, number_of_groups=requestBody.numberOfSets, max_seconds=requestBody.maxDuration)
        return OperatorRequestResponse(payload=list(map(lambda x: FormatHelper.get_sql_query(pipeline, x), result_sets)))
    except Exception as error:
        print(error)
        traceback.print_tb(error.__traceback__)
        return OperatorRequestResponse(error=1, errorMsg=str(error))


@app.get("/operators/by-facet-g",
         description="Groups the input set items by a list of provided attributes and returns the n biggest resulting sets",
         tags=["operators"])
async def by_facet_g(input_set_id: int = -1, dimensions: List[str] = Query([]), get_scores: bool = False, get_predicted_scores: bool = False, seen_predicates: List[str] = Query([])):
    result = []
    try:
        # print(requestBody.json())
        pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache["galaxies"]
        if input_set_id == -1:
            dataset = pipeline.get_dataset()
        else:
            dataset = pipeline.get_groups_as_datasets([input_set_id])[0]
        number_of_groups = 10 if len(dimensions) == 1 else 5
        result_sets = pipeline.by_facet(
            dataset=dataset, attributes=dimensions, number_of_groups=number_of_groups)
        result = get_galaxies_sets(result_sets, pipeline, get_scores,
                                   get_predicted_scores, seen_predicates=set(seen_predicates))
        return result
    except Exception as error:
        print(error)
        # print(requestBody.json())
        traceback.print_tb(error.__traceback__)
        try:
            print(json.dumps(result))
        except Exception as err:
            print(err)
        return 0


@app.get("/operators/by-superset-g",
         description="Returns the smallest set completely overlapping with the input set",
         tags=["operators"])
async def by_superset_g(input_set_id: int = -1, get_scores: bool = False, get_predicted_scores: bool = False, seen_predicates: List[str] = Query([])):
    result = []
    try:
        pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache["galaxies"]
        if input_set_id == -1:
            dataset = pipeline.get_dataset()
        else:
            dataset = pipeline.get_groups_as_datasets([input_set_id])[0]
        result_sets = pipeline.by_superset(
            dataset=dataset)
        result = get_galaxies_sets(result_sets, pipeline, get_scores,
                                   get_predicted_scores, seen_predicates=set(seen_predicates))
        return result
    except Exception as error:
        print(error)
        # print(requestBody.json())
        traceback.print_tb(error.__traceback__)
        try:
            print(json.dumps(result))
        except Exception as err:
            print(err)
        return 0


@app.get("/operators/by-neighbors-g",
         description="",
         tags=["operators"])
async def by_neighbors_g(input_set_id: int, dimensions: List[str] = Query([]), get_scores: bool = False, get_predicted_scores: bool = False, seen_predicates: List[str] = Query([])):
    result = []
    try:
        # print(requestBody.json())
        pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache["galaxies"]
        dataset = pipeline.get_groups_as_datasets([input_set_id])[0]
        result_sets = pipeline.by_neighbors(
            dataset=dataset, attributes=dimensions)
        result = get_galaxies_sets(result_sets, pipeline, get_scores,
                                   get_predicted_scores, seen_predicates=set(seen_predicates))
        return result
    except Exception as error:
        print(error)
        # print(requestBody.json())
        traceback.print_tb(error.__traceback__)
        try:
            print(json.dumps(result))
        except Exception as err:
            print(err)
        return 0


@app.get("/operators/by-distribution-g",
         description="",
         tags=["operators"])
async def by_distribution_g(input_set_id: int, get_scores: bool = False, get_predicted_scores: bool = False, seen_predicates: List[str] = Query([])):
    result = []
    try:
        # print(requestBody.json())
        pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache["galaxies"]
        dataset = pipeline.get_groups_as_datasets([input_set_id])[0]
        result_sets = pipeline.by_distribution(
            dataset=dataset)
        result = get_galaxies_sets(result_sets, pipeline, get_scores,
                                   get_predicted_scores, seen_predicates=set(seen_predicates))
        return result
    except Exception as error:
        print(error)
        # print(requestBody.json())
        traceback.print_tb(error.__traceback__)
        try:
            print(json.dumps(result))
        except Exception as err:
            print(err)
        return 0


@app.get("/operators/get-dataset-information",
         description="",
         tags=["info"])
async def get_dataset_information():
    pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache["galaxies"]
    return {
        "dimensions": pipeline.exploration_columns,
        "ordered_dimensions": pipeline.ordered_dimensions,
        "length": len(pipeline.initial_collection)
    }
