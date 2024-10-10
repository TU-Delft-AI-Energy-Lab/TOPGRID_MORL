import grid2op
env_name = "rte_case5_example"  # only compatible with what comes next (at time of writing)
env = grid2op.make(env_name)
nb_year = 1  # or any "big" number...
env.generate_data(nb_year=1)  # generates 50 years of data
# (takes roughly 50s per week, around 45mins per year, in this case 50 * 45 mins = 37.5 hours)