from os import stat
import traceback
from typing import List, Dict, Optional
import json

from fastapi import Body, Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel
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
from rl.A3C_2_actors.target_set_generator import TargetSetGenerator
from app.model_manager import ModelManager

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

model_manager = ModelManager(database_pipeline_cache["galaxies"])

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


class GalaxyRequest(BaseModel):
    input_set_id: Optional[int]=None
    dimensions: Optional[List[str]]=None
    get_scores: Optional[bool]=False
    get_predicted_scores: Optional[bool]=False
    target_set: Optional[str]=None
    curiosity_weight: Optional[float]=None
    found_items_with_ratio: Optional[Dict[str, float]]=None
    target_items: Optional[List[int]] = None
    previous_set_states: Optional[List[List[float]]] = None
    previous_operation_states: Optional[List[List[float]]] = None
    seen_predicates: Optional[List[str]] = []
    dataset_ids: Optional[List[int]] = None

@app.put("/operators/by_facet-g",
         description="Groups the input set items by a list of provided attributes and returns the n biggest resulting sets",
         tags=["operators"])
async def by_facet_g( galaxy_request:GalaxyRequest):
    result = []
    try:
        # print(requestBody.json())
        pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache["galaxies"]
        if galaxy_request.input_set_id == -1:
            dataset = pipeline.get_dataset()
        else:
            dataset = pipeline.get_groups_as_datasets([galaxy_request.input_set_id])[0]
        number_of_groups = 10 if len(galaxy_request.dimensions) == 1 else 5
        result_sets = pipeline.by_facet(
            dataset=dataset, attributes=galaxy_request.dimensions, number_of_groups=number_of_groups)
        result_sets = [d for d in result_sets if d.set_id != None and d.set_id >= 0]
        if len(result_sets) == 0:
            result_sets = [dataset] 
        prediction_result = {}
        if galaxy_request.target_set != None and galaxy_request.curiosity_weight != None:
            prediction_result = model_manager.get_prediction( result_sets, galaxy_request.target_set, galaxy_request.curiosity_weight, galaxy_request.target_items, 
                galaxy_request.found_items_with_ratio, galaxy_request.previous_set_states, galaxy_request.previous_operation_states)

        result = get_galaxies_sets(result_sets, pipeline, galaxy_request.get_scores,
                                   galaxy_request.get_predicted_scores, seen_predicates=set(galaxy_request.seen_predicates))
        result.update(prediction_result)
        
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


@app.put("/operators/by_superset-g",
         description="Returns the smallest set completely overlapping with the input set",
         tags=["operators"])
async def by_superset_g(galaxy_request:GalaxyRequest):
    result = []
    try:
        pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache["galaxies"]
        if galaxy_request.input_set_id == -1:
            dataset = pipeline.get_dataset()
        else:
            dataset = pipeline.get_groups_as_datasets([galaxy_request.input_set_id])[0]
        result_sets = pipeline.by_superset(
            dataset=dataset)
        result_sets = [d for d in result_sets if d.set_id != None and d.set_id >= 0]
        if len(result_sets) == 0:
            result_sets = [dataset] 
        prediction_result = {}
        if galaxy_request.target_set != None and galaxy_request.curiosity_weight != None:
            prediction_result = model_manager.get_prediction(result_sets, galaxy_request.target_set, galaxy_request.curiosity_weight, galaxy_request.target_items, 
                galaxy_request.found_items_with_ratio, galaxy_request.previous_set_states, galaxy_request.previous_operation_states)

        result = get_galaxies_sets(result_sets, pipeline, galaxy_request.get_scores,
                                   galaxy_request.get_predicted_scores, seen_predicates=set(galaxy_request.seen_predicates))
        result.update(prediction_result)
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


@app.put("/operators/by_neighbors-g",
         description="",
         tags=["operators"])
async def by_neighbors_g(galaxy_request:GalaxyRequest):
    result = []
    try:
        # print(requestBody.json())
        pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache["galaxies"]
        dataset = pipeline.get_groups_as_datasets([galaxy_request.input_set_id])[0]
        result_sets = pipeline.by_neighbors(
            dataset=dataset, attributes=galaxy_request.dimensions)
        result_sets = [d for d in result_sets if d.set_id != None and d.set_id >= 0]
        if len(result_sets) == 0:
            result_sets = [dataset] 

        prediction_result = {}
        if galaxy_request.target_set != None and galaxy_request.curiosity_weight != None:
            prediction_result = model_manager.get_prediction(result_sets, galaxy_request.target_set, galaxy_request.curiosity_weight, galaxy_request.target_items, 
                galaxy_request.found_items_with_ratio, galaxy_request.previous_set_states, galaxy_request.previous_operation_states)

        result = get_galaxies_sets(result_sets, pipeline, galaxy_request.get_scores,
                                   galaxy_request.get_predicted_scores, seen_predicates=set(galaxy_request.seen_predicates))
        result.update(prediction_result)
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


@app.put("/operators/by_distribution-g",
         description="",
         tags=["operators"])
async def by_distribution_g(galaxy_request:GalaxyRequest):
    result = []
    try:
        # print(requestBody.json())
        pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache["galaxies"]
        dataset = pipeline.get_groups_as_datasets([galaxy_request.input_set_id])[0]
        result_sets = pipeline.by_distribution(
            dataset=dataset)
        result_sets = [d for d in result_sets if d.set_id != None and d.set_id >= 0]
        if len(result_sets) == 0:
            result_sets = [dataset] 
        prediction_result = {}
        if galaxy_request.target_set != None and galaxy_request.curiosity_weight != None:
            prediction_result = model_manager.get_prediction( result_sets, galaxy_request.target_set, galaxy_request.curiosity_weight, galaxy_request.target_items, 
                galaxy_request.found_items_with_ratio, galaxy_request.previous_set_states, galaxy_request.previous_operation_states)

        result = get_galaxies_sets(result_sets, pipeline, galaxy_request.get_scores,
                                   galaxy_request.get_predicted_scores, seen_predicates=set(galaxy_request.seen_predicates))
        result.update(prediction_result)
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


@app.get("/app/get-dataset-information",
         description="",
         tags=["info"])
async def get_dataset_information():
    pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache["galaxies"]
    return {
        "dimensions": pipeline.exploration_columns,
        "ordered_dimensions": pipeline.ordered_dimensions,
        "length": len(pipeline.initial_collection)
    }

@app.get("/app/get-target-items-and-prediction",
         description="",
         tags=["info"])
async def get_target_items_and_prediction( target_set: str = None, curiosity_weight: float = None, dataset_ids: List[int] = []):    
    pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache["galaxies"]
    target_items = TargetSetGenerator.get_diverse_target_set(number_of_samples=50)
    items_found_with_ratio = {}
    if len(dataset_ids) == 0:
        datasets = [pipeline.get_dataset()] 
    else:
        datasets = pipeline.get_groups_as_datasets(dataset_ids)
    
    prediction_results = model_manager.get_prediction(datasets, target_set, curiosity_weight, target_items, items_found_with_ratio)
    prediction_results["targetItems"] = target_items
    return prediction_results

@app.put("/app/load-model",
         description="",
         tags=["info"])
async def load_model( galaxy_request:GalaxyRequest):    
    pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache["galaxies"]
    if len(galaxy_request.dataset_ids) == 0:
        datasets = [pipeline.get_dataset()] 
    else:
        datasets = pipeline.get_groups_as_datasets(galaxy_request.dataset_ids)
    
    prediction_results = model_manager.get_prediction(datasets, galaxy_request.target_set, galaxy_request.curiosity_weight, galaxy_request.target_items, galaxy_request.found_items_with_ratio,
        previous_set_states=galaxy_request.previous_set_states, previous_operation_states=galaxy_request.previous_operation_states)

    return prediction_results
