import numpy as np
from scipy.spatial import distance_matrix


def get_utility_scores(datasets, pipeline):
    if len([d for d in datasets if len(d.data) > 0]) <= 1:
        return 0, [0]
    mean_distances_to_self_set = []
    mean_distances_to_other_sets = []
    datas = []
    for i, set in enumerate(datasets):
        data = set.data[:]
        for column in pipeline.exploration_columns:
            data[column] = data[column].cat.codes
        data = data[pipeline.exploration_columns]
        mean_distances_to_other_sets.append(
            np.zeros((len(datasets)-1, len(data))))
        datas.append(data)

    for index, set in enumerate(datasets):
        data = datas[index]
        matrix_dist_to_itself = distance_matrix(
            data.to_numpy(), data.to_numpy())
        if len(matrix_dist_to_itself) > 1:
            mean_distances_to_self_set.append(
                matrix_dist_to_itself.sum(1)/(len(matrix_dist_to_itself)-1))
        else:
            mean_distances_to_self_set.append(
                matrix_dist_to_itself.sum(1)/len(matrix_dist_to_itself))
        for other_index, other_set in enumerate(datasets):
            other_data = datas[other_index]
            if other_index > index:
                matrix = distance_matrix(
                    data.to_numpy(), other_data.to_numpy())
                mean_distances_to_other_sets[index][other_index -
                                                    1] = matrix.mean(1)
                mean_distances_to_other_sets[other_index][index] = matrix.mean(
                    0)

    min_distances_to_other_sets = []
    for matrix in mean_distances_to_other_sets:
        min_distances_to_other_sets.append(matrix.min(0))

    sets_silhouette_scores = []
    global_sum = 0
    global_count = 0
    for i in range(len(datasets)):
        sil_numerators = min_distances_to_other_sets[i] - \
            mean_distances_to_self_set[i]
        sil_denominators = np.row_stack(
            (min_distances_to_other_sets[i], mean_distances_to_self_set[i])).max(0)
        sil_scores = sil_numerators/sil_denominators
        sets_silhouette_scores.append(sil_scores.mean())
        global_sum += sil_scores.sum()
        global_count += len(sil_scores)
    return global_sum/global_count, sets_silhouette_scores
    # print(f"Set scores: {set_silhouette_score}, Summary score: {global_sum/global_count}")


def get_novelty_scores(datasets, seen_predicates, pipeline):
    possible_predicate_count = pipeline.discrete_categories_count * \
        len(pipeline.exploration_columns)
    sets_novelty_scores = []
    summary_predicates = set()
    for dataset in datasets:
        set_predicates = set(map(
            lambda x: f"{x.attribute}{x.operator}{x.value}", dataset.predicate.components))
        summary_predicates.update(set_predicates)
        set_predicates.update(seen_predicates)
        set_score = len(set_predicates)/possible_predicate_count - \
            len(seen_predicates)/possible_predicate_count
        sets_novelty_scores.append(set_score)

    summary_predicates.update(seen_predicates)
    summary_score = len(summary_predicates)/possible_predicate_count - \
        len(seen_predicates)/possible_predicate_count
    return summary_score, sets_novelty_scores, summary_predicates


def get_future_scores(sets, pipeline, seen_predicates):
    operations = []
    for index, set in enumerate(sets):
        if set.set_id != None and set.set_id >= 0:
            predicate_attributes = set.predicate.get_attributes()
            for attribute in pipeline.exploration_columns:
                if attribute in predicate_attributes:
                    operation = "neighbors"
                    resulting_sets = pipeline.by_neighbors(
                        dataset=set, attributes=[attribute])
                else:
                    operation = "facet"
                    resulting_sets = pipeline.by_facet(
                        dataset=set, attributes=[attribute], number_of_groups=10)
                global_score, sets_scores = get_utility_scores(
                    resulting_sets, pipeline)
                summary_novelty_score, sets_novelty_scores, final_predicates = get_novelty_scores(
                    resulting_sets, seen_predicates, pipeline)
                operations.append({
                    "setId": int(set.set_id),
                    "operation": operation,
                    "attribute": attribute.replace("galaxies.", ""),
                    "utility_score": global_score if not np.isnan(global_score) else 0,
                    "novelty_score": summary_novelty_score if not np.isnan(summary_novelty_score) else 0
                })
            resulting_sets = pipeline.by_distribution(dataset=set)
            global_score, sets_scores = get_utility_scores(
                resulting_sets, pipeline)
            summary_novelty_score, sets_novelty_scores, final_predicates = get_novelty_scores(
                resulting_sets, seen_predicates, pipeline)
            operations.append({
                "setId": int(set.set_id),
                "operation": "distribution",
                "attribute": "",
                "utility_score": global_score if not np.isnan(global_score) else 0,
                "novelty_score": summary_novelty_score if not np.isnan(summary_novelty_score) else 0
            })
    return sorted(operations, key=lambda k: k['utility_score'], reverse=True)


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
    if get_scores:
        if len(sets) == 1:
            results["utility"] = 0
            for dataset in results["sets"]:
                dataset["silhouette"] = 0
        else:
            summary_utility_score, sets_utility_scores = get_utility_scores(
                sets, pipeline)
            results["utility"] = summary_utility_score
            for index, score in enumerate(sets_utility_scores):
                results["sets"][index]["silhouette"] = score
        summary_novelty_score, sets_novelty_scores, seen_predicates = get_novelty_scores(
            sets, seen_predicates, pipeline)
        results["novelty"] = summary_novelty_score
        for index, score in enumerate(sets_novelty_scores):
            results["sets"][index]["novelty"] = score
        results["seenPredicates"] = seen_predicates
    else:
        results["utility"] = None
        results["novelty"] = None

        for dataset in results["sets"]:
            dataset["silhouette"] = None
            dataset["novelty"] = None
        for dataset in sets:
            seen_predicates.update(set(map(
                lambda x: f"{x.attribute}{x.operator}{x.value}", dataset.predicate.components)))
        results["seenPredicates"] = seen_predicates
    if get_predicted_scores:
        results["predictedScores"] = get_future_scores(
            sets, pipeline, seen_predicates)
    else:
        results["predictedScores"] = {}

    return results
