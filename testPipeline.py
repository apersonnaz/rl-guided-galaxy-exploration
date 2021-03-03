from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
data_folder = "./app/data/"
pipel = PipelineWithPrecalculatedSets(
    "sdss", ["galaxies"], data_folder=data_folder, discrete_categories_count=10, exploration_columns=["galaxies.u", "galaxies.g", "galaxies.r", "galaxies.i", "galaxies.z", "galaxies.petrorad_r"])
ds = pipel.get_dataset()
res = pipel.by_facet(
    ds, ["galaxies.r", "galaxies.petroRad_r"], 5)
res = pipel.by_facet(res[3], ["galaxies.g"], 5)
pipel.by_distribution(res[2])
print("end")
