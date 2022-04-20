from environments import env_1
import numpy as np
from actor_critic import Agent
import os
import random
import pandas as pd

if __name__ == '__main__':
    random.seed(42)
    WORKING_DIR = '/Users/timhartnett/Downloads/RL_bayesian_optimization/'
    data_df = pd.read_csv(WORKING_DIR+'Dataset/train.csv')
    data_df = data_df.drop('number_of_elements',axis=1)
    data_df = data_df[['mean_atomic_mass','std_atomic_mass','mean_fie',
                  'mean_atomic_radius','std_atomic_radius',
                  'mean_ElectronAffinity','mean_FusionHeat',
                  'std_FusionHeat','mean_ThermalConductivity',
                  'std_ThermalConductivity','mean_Valence','entropy_Valence','std_Valence',
                  'critical_temp']]
    data_df = data_df.dropna()
    print('... data loaded...')
    print('training dataset of size %s' % data_df.shape[0])
    #env = gym.make('LunarLander-v2')
    train_sample = data_df.sample(n=500,axis=0)
    data = train_sample.values
    env = env_1(data)
    env.reset(20)
    agent = Agent(alpha=1e-5, n_actions=6)
    n_episodes = 100
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    score_history=[]
    best_score = data.shape[0]*-1
    
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
    
    for i in range(n_episodes):
        print(agent.action_space)
        print('#### Episode %s ####'%i )
        data = data_df.values
        env = env_1(data)
        env.reset(50)
        observation = env.current_state
        done = False
        score = 0
        exp_num = 0
        mean, var = env.predict_virtual()
        print((mean-max(mean)).shape)
        print(var.shape)
        while not done:
            exp_num +=1
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action,experiment_number=exp_num)
            print(info)
            score += reward
            if not load_checkpoint:
                agent.learn(observation, reward, observation_, done)
            observation = observation_
            env.update_model()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if i % 10 == 0:
            agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
    agent.save_models()
    
    if not load_checkpoint:
        x = [i+1 for i in range(n_episodes)]