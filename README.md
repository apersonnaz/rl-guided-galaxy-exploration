# rl-guided-galaxy-exploration

Requires python 3.7+

Install required dependencies:

    python3 -m pip install -r requirements.txt

Unzip galaxies data CSV (too big for github):

    cd app/data/sdss
    cat galaxies.tar.gz.* | tar xzvf -

Then to run a training, from the root of the project:

    python3 RL-launcher.py

Possible arguments:
- --gamma default=0.99
- --update_interval default=50
- --actor_lr default=0.00002
- --critic_lr default=0.00008
- --icm_lr default=0.05
- --workers default=6
- --lstm_steps default=10
- --target_set default=None
- --notes default=
- --mode default=scattered
- --curiosity_ratio default=0
- --counter_curiosity_ratio default=0
- --name default=mode and hyperparameters and date
                    