"""
Run this file only once to generate train/val/testset
"""
import grid2op
env_name= 'l2rpn_case14_sandbox' # or any other...
env = grid2op.make(env_name)

nm_env_train, nm_env_val = env.train_val_split_random(pct_val=10.)

# and now you can use the training set only to train your agent:
print(f"The name of the training environment is \"{nm_env_train}\"")
print(f"The name of the validation environment is \"{nm_env_val}\"")
env_train = grid2op.make(nm_env_train)
env_val = grid2op.make(nm_env_val)