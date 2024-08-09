import gymnasium as gym
import mo_gymnasium as mo_gym

from topgrid_morl.wrapper.ols import LinearSupport
from morl_baselines.single_policy.ser.mo_q_learning import MOQLearning


def main():
      
    
    GAMMA = 0.99
    env = mo_gym.MORecordEpisodeStatistics(mo_gym.make('deep-sea-treasure-v0'), gamma=GAMMA)

    ols = LinearSupport(num_objectives=2, epsilon=0.01, verbose=True)
    policies = []
    values = []
    while not ols.ended():
        w = ols.next_weight(algo='ols')
        print(f"this weights will be given to the MOQ_learning{w}")
        if ols.ended(): 
            break
        new_policy = MOQLearning(env, weights=w, learning_rate=0.3, gamma=GAMMA, initial_epsilon=1, final_epsilon=0.01, epsilon_decay_steps=int(1e5))
        new_policy.train(0, total_timesteps=int(2e4))

        _, _, vec, discounted_vec = new_policy.policy_eval(eval_env=env, weights=w,)
        values.append(discounted_vec)
        policies.append(new_policy)

        removed_inds, _ = ols.add_solution(discounted_vec, w)
    print(policies)
    print(values)
    

if __name__ == "__main__":
    main()