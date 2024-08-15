import json
import dill as pickle  # Use dill instead of pickle
import os 
import numpy as np
from grid2op.Space import GridObjects

def load_agent_and_weights(directory, index):
    # Load the agent
    agent_path = os.path.join(directory, f"agent_{index}.pkl")
    with open(agent_path, "rb") as f:
        agent = pickle.load(f)
    
    # Load the weights
    weights_path = os.path.join(directory, f"weights_{index}.json")
    with open(weights_path, "r") as f:
        weights = json.load(f)
    
    return agent, np.array(weights)

# Test loading
load_agent_and_weights(r"morl_logs\OLS\rte_case5_example\2024-08-15\['ScaledL2RPN', 'ScaledTopoDepth']\seed_42", 1)
