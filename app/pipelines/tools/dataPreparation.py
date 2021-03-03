import multiprocessing
import os
import ast
import csv
import gc
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from ..pipeline import Pipeline
from itertools import repeat
from multiprocessing import Manager
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager


def get_thread_boundaries(number_of_sets, number_of_threads):
    boundaries = []
    iterations_per_thread = number_of_sets * \
        (number_of_sets - 1) / 2 / number_of_threads
    set_counter = 0
    iterations_counter = 0
    while len(boundaries) < number_of_threads - 1:
        while iterations_counter < iterations_per_thread:
            set_counter += 1
            iterations_counter = iterations_counter + number_of_sets - 1 - set_counter
        boundaries.append(set_counter)
        iterations_counter = 0
    boundaries.append(number_of_sets)
    return boundaries


def selectivity_thread_function(step, id_attribute_name, manager_dict, group_count, threadNumber):
    start_index = threadNumber * step
    if (threadNumber + 2) * step > group_count:
        end_index = group_count
    else:
        end_index = (threadNumber+1) * step
    if start_index != end_index:
        groups = manager_dict["groups"]
        pipeline = manager_dict["pipeline"]
        print(f"started n{threadNumber} {start_index} {end_index}")
        indexes = {}
        selectivities = {}

        for set_id, group in groups[start_index:end_index].iterrows():
            # group_members_set = eval(group.members)
            group_members_set = group.members
            group_df = pipeline.initial_collection[pipeline.initial_collection[id_attribute_name].isin(
                group_members_set)]
            for attribute in pipeline.interesting_attributes:
                values = group_df.loc[:, attribute].value_counts(
                    normalize=True, dropna=False)
                for value, selectivity in values.iteritems():
                    try:
                        if type(value) == str:
                            value = value.replace(
                                '/', Pipeline.SLASH_ESCAPER)
                        if selectivity != 0.0:
                            key = f"{attribute}={value}"
                            if not key in selectivities:
                                selectivities[key] = {"set_id": [
                                    set_id], "selectivity": [selectivity]}
                            else:
                                selectivities[key]["set_id"].append(set_id)
                                selectivities[key]["selectivity"].append(
                                    selectivity)
                    except TypeError:
                        print(f"TypeError: {value}")
        return selectivities


def index_thread_second_apply(group1, group2, overlaps):
    if group1.name != group2.name:
        intersection = len(
            group1 & group2)
        if intersection:
            overlap_value = intersection / \
                len(group1 | group2)
            overlaps["set_1"].append(group1.name)
            overlaps["set_2"].append(group2.name)
            overlaps["overlap"].append(overlap_value)


def index_thread_first_apply(group1, groups, threadNumber, start_index, overlaps):
    print(
        f"n{threadNumber} index {group1.name-start_index} {datetime.now().strftime('%H:%M:%S')} resLen {len(overlaps['set_1'])}")
    groups.apply(lambda group2: index_thread_second_apply(
        group1, group2, overlaps), axis=1)


# def index_thread_function(id_attribute_name, manager_dict, threadNumber, thread_boundaries):
#     start_index = 0 if threadNumber == 0 else thread_boundaries[threadNumber-1]
#     end_index = thread_boundaries[threadNumber]
#     groups = manager_dict["groups"]
#     print(f"started n{threadNumber} {start_index} {end_index}")
#     overlaps = {
#         # "optim_counter": 0,
#         # "normal_counter": 0,
#         "set_1": [],
#         "set_2": [],
#         "overlap": []
#     }

#     pd.DataFrame(groups.loc[start_index:end_index, "members"]).apply(
#         lambda group: index_thread_first_apply(group, pd.DataFrame(groups.loc[group.name:len(groups)]), threadNumber, start_index, overlaps), axis=1)

#     return overlaps

def index_thread_function(id_attribute_name, manager_dict, threadNumber, thread_boundaries, temp_folder):
    start_index = 0 if threadNumber == 0 else thread_boundaries[threadNumber-1]
    end_index = thread_boundaries[threadNumber]
    groups = manager_dict["groups"]
    print(f"started n{threadNumber} {start_index} {end_index}")
    overlaps = {
        # "optim_counter": 0,
        # "normal_counter": 0,
        "set_1": [],
        "set_2": [],
        "overlap": []
    }

    for set_1_id, set_1 in groups.loc[start_index:end_index, "members"].iteritems():
        print(
            f"n{threadNumber} index {set_1_id-start_index}/{end_index-start_index} {datetime.now().strftime('%H:%M:%S')} resLen {len(overlaps['set_1'])}")

        for set_2_id, set_2 in groups.loc[set_1_id+1:len(groups), "members"].iteritems():
            # if set_1.definition.issubset(set_2.definition):
            #     overlaps["set_1"].append(set_1_id)
            #     overlaps["set_2"].append(set_2_id)
            #     overlaps["overlap"].append(len(set_2) / len(set_1))
            #     overlaps["optim_counter"] += 1
            # else:
            intersection = len(
                set_1 & set_2)
            if intersection:
                overlap_value = intersection / \
                    len(set_1 | set_2)
                # if overlap_value < 0.2 or overlap_value > 0.8:
                overlaps["set_1"].append(set_1_id)
                overlaps["set_2"].append(set_2_id)
                overlaps["overlap"].append(overlap_value)
                # overlaps["normal_counter"] += 1
    index = pd.DataFrame(
        data={"group1": overlaps["set_1"], "group2": overlaps["set_2"], "overlap": overlaps["overlap"]})
    index.to_csv(f"{temp_folder}/index_{threadNumber}.csv", index=False)


def selectivity_function(executionParameters):
    set_id = executionParameters[0]
    manager_dict = executionParameters[1]
    groups = manager_dict["groups"]
    pipeline = manager_dict["pipeline"]
    id_attribute_name = manager_dict["id_attribute_name"]
    print(f"started {set_id}")
    indexes = {}
    selectivities = {}
    group = groups.loc[set_id]
    group_members_set = eval(group.members)
    group_df = pipeline.initial_collection[pipeline.initial_collection[id_attribute_name].isin(
        group_members_set)]
    for attribute in pipeline.interesting_attributes:
        values = group_df.loc[:, attribute].value_counts(
            normalize=True, dropna=False)
        for value, selectivity in values.iteritems():
            try:
                if type(value) == str:
                    value = value.replace(
                        '/', Pipeline.SLASH_ESCAPER)
                if selectivity != 0.0:
                    key = f"{attribute}={value}"
                    if not key in selectivities:
                        selectivities[key] = {"set_id": [
                            set_id], "selectivity": [selectivity]}
                    else:
                        selectivities[key]["set_id"].append(set_id)
                        selectivities[key]["selectivity"].append(
                            selectivity)
            except TypeError:
                print(f"TypeError: {value}")
    return selectivities


def index_function(executionParameters):
    set_id = executionParameters[0]
    sets_members = executionParameters[1]
    set_members = executionParameters[2]
    print(f"started {set_id}")
    overlaps = {
        "set_1": [],
        "set_2": [],
        "overlap": []
    }
    set_1_members = eval(set_members)
    # groups.drop(set_1_id)
    for set_2_id, set_2_members_str in enumerate(sets_members):
        set_2_members = eval(set_2_members_str)
        intersection = len(
            set_1_members & set_2_members)
        if intersection:
            overlap_value = intersection / \
                len(set_1_members | set_2_members)
            overlaps["set_1"].append(set_id)
            overlaps["set_2"].append(set_2_id)
            overlaps["overlap"].append(overlap_value)
    # index = pd.DataFrame(result_array, columns=[
    #     "group1", "group2", "overlap"])
    # index.sort_values(by=["group1", "overlap"], inplace=True)
    # index.to_csv(
    #     f"{temp_folder}/index_{threadNumber}.csv", index=False)
    # groups.drop(set_id)
    return overlaps


def prepare_data(data_folder="", database_name="", initial_collection_names=[], id_attribute_name="",
                 build_index=True, build_selectivity_index=True, build_groups=True, index_build_process_count=62,
                 min_group_size=10, discrete_categories_count=20, exploration_columns=[]):
    pipeline = None
    index_folder = f"{data_folder}/{database_name}/{'_'.join(sorted(initial_collection_names))}_index"
    selectivities_folder = index_folder + "/selectivities"
    Path(selectivities_folder).mkdir(parents=True, exist_ok=True)
    temp_folder = index_folder + "/temp"
    Path(temp_folder).mkdir(parents=True, exist_ok=True)

    start_time = datetime.now().strftime('%H:%M:%S')

    groups = pd.DataFrame()
    if build_groups:
        pipeline = Pipeline(
            database_name, initial_collection_names, data_folder=data_folder, discrete_categories_count=discrete_categories_count, exploration_columns=exploration_columns)
        df = pipeline.get_dataset().data[exploration_columns]
        counter = 0
        correspondences_df = pd.DataFrame(columns=["id", "value", "column"])
        item_ids = df[id_attribute_name].to_list()
        for column_name in df:
            df[column_name+"_category"] = pd.Categorical(df[column_name])
            df[column_name+"_correspondence_id"] = df[column_name+"_category"].cat.codes
            df[column_name+"_correspondence_id"] += counter
            df[column_name +
                "_correspondence_id"] = df[column_name +
                                           "_correspondence_id"].replace({counter - 1: np.NaN})
            counter = df[column_name+"_correspondence_id"].max() + 1
            column_correspondence = df[[
                column_name+"_correspondence_id", column_name]].drop_duplicates()
            column_correspondence = column_correspondence.rename(
                columns={column_name+"_correspondence_id": "id", column_name: "value"}).sort_values(by="id")
            column_correspondence["column"] = column_name
            if column_name != id_attribute_name:
                correspondences_df = correspondences_df.append(
                    column_correspondence, ignore_index=True)
            df = df.drop(columns=[column_name, column_name+"_category"])

        correspondences_df.to_csv(
            f"{index_folder}/correspondences.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

        df = df.astype(str)
        df = df.replace(to_replace=r"\.0+$", value="", regex=True)
        df = df.replace(to_replace="nan", value="")
        print(df.info())
        df.to_csv(f"{temp_folder}/items.dat",
                  index=False, sep=" ", header=False, )
        print("Running LCM")
        # subprocess.run(["lcm", "CI", "-l", "1", "-U", str(2*len(df)/3) , "./data/dataConversion/items.dat", "10", "./data/dataConversion/groups.dat"])
        subprocess.run(["lcm", "CI", "-l", "1", "-u", "25", f"{temp_folder}/items.dat",
                        str(min_group_size), f"{temp_folder}/groups.dat"])

        group_definitions = []
        group_members = []
        group_member_counts = []

        with open(f"{temp_folder}/groups.dat", 'rb') as group_file:
            while True:
                group_definition_line = group_file.readline().rstrip()
                if not group_definition_line:
                    break
                members_line = group_file.readline().rstrip()
                members = np.fromstring(members_line, dtype=int, sep=" ")
                group_member_counts.append(len(members))
                # members = map(lambda x: item_ids[x], members)
                group_definitions.append(
                    set(np.fromstring(group_definition_line, dtype=int, sep=" ")))
                # group_members.append(set(members))

        groups = pd.DataFrame(data={"definition": group_definitions, "member_count": group_member_counts})
        groups = groups.sort_values("member_count", ascending=False)
        groups = groups.reset_index()
        groups = groups.drop(["index"], axis=1)
        groups.index = groups.index.rename("id")
        print(groups.info())
        groups.to_csv(f"{index_folder}/groups.csv")
        # groups = pd.DataFrame(data={"definition": group_definitions,
        #                             "members": group_members, "member_count": group_member_counts})
        # groups = groups.sort_values("member_count", ascending=False)
        # groups = groups.reset_index()
        # groups = groups.drop(["index", "member_count"], axis=1)
        # groups.index = groups.index.rename("id")
        # print(groups.info())
        # groups.to_csv(f"{index_folder}/groups.csv")

    d = {}
    if groups.empty:
        print("Reading Groups")
        # mgr = Manager()

        # d = mgr.dict()

        groups = pd.read_csv(
            f"{index_folder}/groups.csv",
            converters={
                "definition": eval,
                "members": eval
            },
            index_col="id"
        )

    d["groups"] = groups
    print(f"There are {len(d['groups'])} groups")
    # tt = d["groups"].members
    # tt = tt.to_list()
    # d["groups"].members = list(map(eval, tt))

    d["id_attribute_name"] = id_attribute_name

    group_count = len(d["groups"])
    step = group_count // index_build_process_count
    if step == 0:
        step = 1
    if build_selectivity_index:
        print("Building selectivity indexes")
        if pipeline == None:
            pipeline = Pipeline(
                database_name, initial_collection_names, data_folder=data_folder)
        d["pipeline"] = pipeline
        futures = list()
        selectivities = {}
        with ProcessPoolExecutor(max_workers=index_build_process_count) as executor:
            # futures = executor.map(selectivity_function, executionParameters)
            for i in range(index_build_process_count):
                futures.append(executor.submit(selectivity_thread_function, step=step,
                                               id_attribute_name=id_attribute_name,
                                               manager_dict=d,
                                               group_count=group_count,
                                               threadNumber=i))

            for future in futures:
                thread_selectivities = future.result()
                if thread_selectivities != None:
                    for key, selectivity in thread_selectivities.items():
                        if not key in selectivities:
                            selectivities[key] = selectivity
                        else:
                            selectivities[key]["set_id"] += selectivity["set_id"]
                            selectivities[key]["selectivity"] += selectivity["selectivity"]

        print("Joining selectivity indexes")
        for key, selectivity in selectivities.items():
            set_df = pd.DataFrame(
                data={"set_id": selectivity["set_id"], "selectivity": selectivity["selectivity"]}).sort_values(by="selectivity")
            set_df.to_csv(
                os.path.join(selectivities_folder, f"{key[0:128]}.csv"), index=False)
        selectivities = None
        print(gc.collect())

    if build_index:
        print("Building index")
        futures = list()
        overlaps = {
            # "optim_counter": 0,
            # "normal_counter": 0,
            "set_1": [],
            "set_2": [],
            "overlap": []
        }

        # smm = SharedMemoryManager()
        # smm.start()
        # sets_members = smm.ShareableList(d["groups"].members.to_list())
        # executionParameters = []

        # executionParameters = zip(range(len(d["groups"])), [
        #                           sets_members]*len(d["groups"]), d["groups"].members.to_list())
        # executionParameters = zip(range(5), [
        #                           sets_members]*5, d["groups"][0:5].members.to_list())
        with ProcessPoolExecutor(max_workers=index_build_process_count) as executor:
            # futures = executor.map(index_function, executionParameters)
            thread_boundaries = get_thread_boundaries(
                len(groups), index_build_process_count)
            for i in range(index_build_process_count):

                futures.append(executor.submit(index_thread_function,
                                               id_attribute_name=id_attribute_name,
                                               manager_dict=d,
                                               threadNumber=i,
                                               thread_boundaries=thread_boundaries,
                                               temp_folder=temp_folder))

            for future in futures:
                future.result()
            #     if thread_overlaps != None:
            #         # overlaps["optim_counter"] += thread_overlaps["optim_counter"]
            #         # overlaps["normal_counter"] += thread_overlaps["normal_counter"]
            #         overlaps["set_1"] += thread_overlaps["set_1"]
            #         overlaps["set_2"] += thread_overlaps["set_2"]
            #         overlaps["overlap"] += thread_overlaps["overlap"]

        print("joining results")
        # partial_set_1 = overlaps["set_1"].copy()
        # overlaps["set_1"] += overlaps["set_2"]
        # overlaps["set_2"] += partial_set_1
        # overlaps["overlap"] += overlaps["overlap"]
        frames = []
        for i in range(index_build_process_count):
            frames.append(pd.read_csv(
                f"{temp_folder}/index_{i}.csv"))
        index = pd.concat(frames)
        reverted_index = index.copy()
        reverted_index.rename(
            columns={"group1": "group2", "group2": "group1"}, inplace=True)
        index = pd.concat([index, reverted_index])
        index.sort_values(by=["group1", "overlap"], inplace=True)
        index.to_csv(f"{index_folder}/index.csv", index=False)
    shutil.rmtree(temp_folder)
    end_time = datetime.now().strftime('%H:%M:%S')
    # print("optim_counter: " + str(overlaps["optim_counter"]))
    # print("normal_counter: " + str(overlaps["normal_counter"]))
    print(f"start: {start_time} end: {end_time}")
