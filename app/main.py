import traceback
from typing import List, Dict, Optional
import json

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

from .pipelines.pipeline import Pipeline
from .pipelines.pipeline_precalculated_sets import \
    PipelineWithPrecalculatedSets
# from .pipelines.pipeline_sql import PipelineSql
from .pipelines.predicateitem import JoinParameters
from rl.A3C_2_actors.target_set_generator import TargetSetGenerator
from app.model_manager import ModelManager

app = FastAPI(title="CNRS pipelines API",
              description="API providing access to the CNRS pipelines operators",
              version="1.0.0",)

data_folder = "./app/data/"
database_pipeline_cache = {}
database_pipeline_cache["galaxies"] = PipelineWithPrecalculatedSets(
    "sdss", ["galaxies"], data_folder=data_folder, discrete_categories_count=10, min_set_size=10, exploration_columns=["galaxies.u", "galaxies.g", "galaxies.r", "galaxies.i", "galaxies.z", "galaxies.petroRad_r", "galaxies.redshift"])


model_manager = ModelManager(database_pipeline_cache["galaxies"])

app.mount("/test", StaticFiles(directory="test"), name="test")


def get_galaxies_sets(sets, pipeline, get_scores, get_predicted_scores, seen_predicates=None):
    results = {
        "sets": []
    }
    for dataset in sets:
        res = {
            "length": len(dataset.data),
            "id": int(dataset.set_id) if dataset.set_id else -1,
            "data": [],
            "predicate": []
        }

        for predicate in dataset.predicate.components:
            res["predicate"].append(
                {"dimension": predicate.attribute, "value": str(predicate.value)})

        # set.data = set.data[["galaxies.ra", "galaxies.dec"]]
        if len(dataset.data) > 12:
            data = dataset.data.sample(n=12)
        else:
            data = dataset.data
        for index, galaxy in data[["galaxies.ra", "galaxies.dec"]].iterrows():
            res["data"].append(
                {"ra": float(galaxy["galaxies.ra"]), "dec": float(galaxy["galaxies.dec"])})

        results["sets"].append(res)

    return results


class GalaxyRequest(BaseModel):
    input_set_id: Optional[int] = None
    dimensions: Optional[List[str]] = None
    get_scores: Optional[bool] = False
    get_predicted_scores: Optional[bool] = False
    target_set: Optional[str] = None
    curiosity_weight: Optional[float] = None
    found_items_with_ratio: Optional[Dict[str, float]] = None
    target_items: Optional[List[str]] = None
    previous_set_states: Optional[List[List[float]]] = None
    previous_operation_states: Optional[List[List[float]]] = None
    seen_predicates: Optional[List[str]] = []
    dataset_ids: Optional[List[int]] = None


@app.put("/operators/by_facet-g",
         description="Groups the input set items by a list of provided attributes and returns the n biggest resulting sets",
         tags=["operators"])
async def by_facet_g(galaxy_request: GalaxyRequest):
    result = []
    try:
        # print(requestBody.json())
        pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache["galaxies"]
        galaxy_request.target_items = set(
            map(lambda x: int(x), galaxy_request.target_items))
        if galaxy_request.input_set_id == -1:
            dataset = pipeline.get_dataset()
        else:
            dataset = pipeline.get_groups_as_datasets(
                [galaxy_request.input_set_id])[0]
        number_of_groups = 10 if len(galaxy_request.dimensions) == 1 else 5
        result_sets = pipeline.by_facet(
            dataset=dataset, attributes=galaxy_request.dimensions, number_of_groups=number_of_groups)
        result_sets = [d for d in result_sets if d.set_id !=
                       None and d.set_id >= 0]
        if len(result_sets) == 0:
            result_sets = [dataset]
        prediction_result = {}
        if galaxy_request.target_set != None and galaxy_request.curiosity_weight != None:
            prediction_result = model_manager.get_prediction(result_sets, galaxy_request.target_set, galaxy_request.curiosity_weight, galaxy_request.target_items,
                                                             galaxy_request.found_items_with_ratio, galaxy_request.previous_set_states, galaxy_request.previous_operation_states)

        result = get_galaxies_sets(result_sets, pipeline, galaxy_request.get_scores,
                                   galaxy_request.get_predicted_scores, seen_predicates=set(galaxy_request.seen_predicates))
        result.update(prediction_result)
        result["curiosityReward"] = model_manager.get_curiosity_reward(
            galaxy_request.target_set, galaxy_request.curiosity_weight, dataset, galaxy_request.dimensions[0], "by_facet")
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
async def by_superset_g(galaxy_request: GalaxyRequest):
    result = []
    try:
        pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache["galaxies"]
        galaxy_request.target_items = set(
            map(lambda x: int(x), galaxy_request.target_items))
        if galaxy_request.input_set_id == -1:
            dataset = pipeline.get_dataset()
        else:
            dataset = pipeline.get_groups_as_datasets(
                [galaxy_request.input_set_id])[0]
        result_sets = pipeline.by_superset(
            dataset=dataset)
        result_sets = [d for d in result_sets if d.set_id !=
                       None and d.set_id >= 0]
        if len(result_sets) == 0:
            result_sets = [dataset]
        prediction_result = {}
        if galaxy_request.target_set != None and galaxy_request.curiosity_weight != None:
            prediction_result = model_manager.get_prediction(result_sets, galaxy_request.target_set, galaxy_request.curiosity_weight, galaxy_request.target_items,
                                                             galaxy_request.found_items_with_ratio, galaxy_request.previous_set_states, galaxy_request.previous_operation_states)

        result = get_galaxies_sets(result_sets, pipeline, galaxy_request.get_scores,
                                   galaxy_request.get_predicted_scores, seen_predicates=set(galaxy_request.seen_predicates))
        result["curiosityReward"] = model_manager.get_curiosity_reward(
            galaxy_request.target_set, galaxy_request.curiosity_weight, dataset, None, "by_superset")
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
async def by_neighbors_g(galaxy_request: GalaxyRequest):
    result = []
    try:
        # print(requestBody.json())
        pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache["galaxies"]
        galaxy_request.target_items = set(
            map(lambda x: int(x), galaxy_request.target_items))
        dataset = pipeline.get_groups_as_datasets(
            [galaxy_request.input_set_id])[0]
        result_sets = pipeline.by_neighbors(
            dataset=dataset, attributes=galaxy_request.dimensions)
        result_sets = [d for d in result_sets if d.set_id !=
                       None and d.set_id >= 0]
        if len(result_sets) == 0:
            result_sets = [dataset]

        prediction_result = {}
        if galaxy_request.target_set != None and galaxy_request.curiosity_weight != None:
            prediction_result = model_manager.get_prediction(result_sets, galaxy_request.target_set, galaxy_request.curiosity_weight, galaxy_request.target_items,
                                                             galaxy_request.found_items_with_ratio, galaxy_request.previous_set_states, galaxy_request.previous_operation_states)

        result = get_galaxies_sets(result_sets, pipeline, galaxy_request.get_scores,
                                   galaxy_request.get_predicted_scores, seen_predicates=set(galaxy_request.seen_predicates))
        result["curiosityReward"] = model_manager.get_curiosity_reward(
            galaxy_request.target_set, galaxy_request.curiosity_weight, dataset, galaxy_request.dimensions[0], "by_neighbors")
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
async def by_distribution_g(galaxy_request: GalaxyRequest):
    result = []
    try:
        # print(requestBody.json())
        pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache["galaxies"]
        galaxy_request.target_items = set(
            map(lambda x: int(x), galaxy_request.target_items))
        dataset = pipeline.get_groups_as_datasets(
            [galaxy_request.input_set_id])[0]
        result_sets = pipeline.by_distribution(
            dataset=dataset)
        result_sets = [d for d in result_sets if d.set_id !=
                       None and d.set_id >= 0]
        if len(result_sets) == 0:
            result_sets = [dataset]
        prediction_result = {}
        if galaxy_request.target_set != None and galaxy_request.curiosity_weight != None:
            prediction_result = model_manager.get_prediction(result_sets, galaxy_request.target_set, galaxy_request.curiosity_weight, galaxy_request.target_items,
                                                             galaxy_request.found_items_with_ratio, galaxy_request.previous_set_states, galaxy_request.previous_operation_states)

        result = get_galaxies_sets(result_sets, pipeline, galaxy_request.get_scores,
                                   galaxy_request.get_predicted_scores, seen_predicates=set(galaxy_request.seen_predicates))
        result["curiosityReward"] = model_manager.get_curiosity_reward(
            galaxy_request.target_set, galaxy_request.curiosity_weight, dataset, None, "by_distribution")
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
async def get_target_items_and_prediction(target_set: str = None, curiosity_weight: float = None, dataset_ids: List[int] = []):
    pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache["galaxies"]
    target_items = TargetSetGenerator.get_diverse_target_set(
        number_of_samples=100)
    items_found_with_ratio = {}
    if len(dataset_ids) == 0:
        datasets = [pipeline.get_dataset()]
    else:
        datasets = pipeline.get_groups_as_datasets(dataset_ids)

    prediction_results = model_manager.get_prediction(
        datasets, target_set, curiosity_weight, target_items, items_found_with_ratio)
    prediction_results["targetItems"] = list(
        map(lambda x: str(x), target_items))
    return prediction_results


@app.put("/app/load-model",
         description="",
         tags=["info"])
async def load_model(galaxy_request: GalaxyRequest):
    pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache["galaxies"]
    if len(galaxy_request.dataset_ids) == 0:
        datasets = [pipeline.get_dataset()]
    else:
        datasets = pipeline.get_groups_as_datasets(galaxy_request.dataset_ids)

    prediction_results = model_manager.get_prediction(datasets, galaxy_request.target_set, galaxy_request.curiosity_weight, galaxy_request.target_items, galaxy_request.found_items_with_ratio,
                                                      previous_set_states=galaxy_request.previous_set_states, previous_operation_states=galaxy_request.previous_operation_states)

    return prediction_results
