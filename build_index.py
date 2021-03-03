from app.pipelines.tools.dataPreparation import prepare_data
import pandas as pd


prepare_data(database_name="sdss", initial_collection_names=["galaxies"], discrete_categories_count=10,
             id_attribute_name="galaxies.objID", index_build_process_count=3, data_folder="./app/data/", build_groups=True,
             build_selectivity_index=False, build_index=False, min_group_size=10, exploration_columns=["galaxies.objID", "galaxies.u", "galaxies.g", "galaxies.r",
                                                                                                      "galaxies.i", "galaxies.z", "galaxies.petroRad_r", "galaxies.redshift"])

#   s.class = 'GALAXY'
#   AND s.z between 0.11 AND 0.36
#   AND p.r >= 18 and p.r <= 20.5
#   AND p.petrorad_r < 2
#   And p.u-p.r <= 2.5 and p.r-p.i <= 0.2 and p.r-p.z <= 0.5
#   And p.g-p.r >= p.r-p.i+0.5 and p.u-p.r  >= 2.5*(p.r-p.z)
#   AND oiii_5007_eqw < -100
#   AND h_beta_eqw < -50
