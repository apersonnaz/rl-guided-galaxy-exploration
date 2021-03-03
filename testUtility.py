
from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
from scipy.spatial import distance_matrix
import numpy as np

data_folder = "./app/data/"

pipeline = PipelineWithPrecalculatedSets(
        "sdss", ["galaxies"], data_folder=data_folder, discrete_categories_count=10, min_set_size=10, exploration_columns=["galaxies.u", "galaxies.g", "galaxies.r", "galaxies.i", "galaxies.z", "galaxies.petroRad_r"])

dataset = pipeline.get_dataset()

datasets = pipeline.by_facet(dataset, ["galaxies.u", "galaxies.r", "galaxies.petroRad_r"], number_of_groups=5)

def get_scores(datasets, pipeline):
    mean_distances_to_self_set = []
    mean_distances_to_other_sets = []
    for i, set in enumerate(datasets):
        for column in pipeline.exploration_columns:
            set.data[column] = set.data[column].cat.codes
        set.data = set.data[pipeline.exploration_columns]
        mean_distances_to_other_sets.append(np.zeros((len(datasets)-1, len(set.data))))

    for index, set in enumerate(datasets): 
        matrix_dist_to_itself = distance_matrix(set.data.to_numpy(), set.data.to_numpy())
        mean_distances_to_self_set.append(matrix_dist_to_itself.sum(1)/(len(matrix_dist_to_itself)-1))
        for other_index, other_set in enumerate(datasets):
            if other_index > index:
                matrix = distance_matrix(set.data.to_numpy(), other_set.data.to_numpy())
                mean_distances_to_other_sets[index][other_index-1] = matrix.mean(1)
                mean_distances_to_other_sets[other_index][index] = matrix.mean(0)

    min_distances_to_other_sets = []
    for matrix in mean_distances_to_other_sets:
        min_distances_to_other_sets.append(matrix.min(0))

    set_silhouette_score = []
    global_sum = 0
    global_count = 0
    for i in range(len(datasets)):
        sil_numerators = min_distances_to_other_sets[i] - mean_distances_to_self_set[i]
        sil_denominators = np.row_stack((min_distances_to_other_sets[i], mean_distances_to_self_set[i])).max(0)
        sil_scores = sil_numerators/sil_denominators
        # set_silhouette_score.append(sil_scores.mean())
        global_sum += sil_scores.sum()
        global_count += len(sil_scores)

    print(f"Set scores: {set_silhouette_score}, Summary score: {global_sum/global_count}")
