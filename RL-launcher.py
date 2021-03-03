from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
from rl.A3C_2_actors.A3C import Agent


def main():
    # env_name = 'CartPole-v1'
    env_name = 'pipeline'
    data_folder = "./app/data/"
    pipeline = PipelineWithPrecalculatedSets(
        "sdss", ["galaxies"], data_folder=data_folder, discrete_categories_count=10, min_set_size=10, exploration_columns=["galaxies.u", "galaxies.g", "galaxies.r", "galaxies.i", "galaxies.z", "galaxies.petroRad_r", "galaxies.redshift"])
    agent = Agent(env_name, pipeline)
    agent.train()


if __name__ == "__main__":
    main()
