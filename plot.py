import numpy as np
import gym_causal
import argparse
import matplotlib.pyplot as plt
import gym
import networkx as nx
import random
import scipy.special as spc
import gym_causal.agents.causal_agent as ga

def get_data_for_causal_agent(probs, episode_count = 100):
     
    # for the plot
    x = np.arange(0, episode_count*11, 1)

    # INIT PARAMETERS
    probs1, probs2 = probs
    list_rewards = []
    reward = 0
    done = False
    temp = 1000000
    #INIT ENV
    env = gym.make('gym_causal:causal-v0')
    env.reset()
    env.set_probs(probs1, probs2)
    env.set_rewards(np.array([0, 1]))
    #CREATE AGENT
    agent = ga.Agent(env.action_space,
                  env.observation_space,
                  env.reward_space)
    # INIT REWARD RECORD
    rewards = np.zeros(episode_count*11)
    total_reward = 0


    j = 0
    plays = 0
    winsplays = 0
    p_win_play = np.zeros(episode_count*11)


    for i in range(0, episode_count):
        # RESET BEFORE EPISODE
        env.reset()
        done = False
        while not done:
            # Causal agent
            temp *= 0.99
            action = agent.act(temp)
            ob, reward, done, _ = env.step(action)
            if (action == 1) and done:
                plays += 1
                if (reward == 1):
                    winsplays += 1
            if (plays != 0):
                p_win_play[i] =  winsplays/plays
                
            # update agent memory & causal graph
            agent.update_memory(action, ob, reward, done)

            total_reward += reward
            rewards[j] = total_reward

            j += 1
        


        list_rewards.append(rewards)
        env.close()
    return np.stack(p_win_play, axis = 0)

def get_data_for_random_agent_not_play(probs, episode_count = 100):
    
    # INIT PARAMETERS
    probs1, probs2 = probs
    list_rewards = []
    reward = 0
    done = False
    temp = 1000000
    #random agent
    randenv = gym.make('gym_causal:causal-v0')
    randenv.reset()
    randenv.set_probs(probs1, probs2)
    randenv.set_rewards(np.array([0, 1]))

    randag = ga.RandomAgent(randenv.action_space,
                  randenv.observation_space,
                  randenv.reward_space)
    randrewards = np.zeros(episode_count*11)
    randtotal_reward = 0
    
    l = 0
    plays1 = 0
    winsplays1 = 0
    p_win_play1 = np.zeros(episode_count*11)
    for i in range(0, episode_count):
        randenv.reset()
        done = False
        while not done:
            # random agent
            action = randag.act()
            #print("action", action)
            ob, reward, done, _ = randenv.step(action)
            #print("reward", reward)
            #print(randenv.time)
            if (action == 0) and done:
                plays1 += 1
                if (reward == 1):
                    winsplays1 += 1
            if (plays1 != 0):
                p_win_play1[i] = winsplays1/plays1
            randtotal_reward += reward
            #print(randtotal_reward)
            randrewards[l] = randtotal_reward

            l += 1


        randenv.close()
    return np.stack(p_win_play1, axis = 0)

def get_data_for_random_agent(probs, episode_count = 100):
    
    # INIT PARAMETERS
    probs1, probs2 = probs
    list_rewards = []
    reward = 0
    done = False
    temp = 1000000
    #random agent
    randenv = gym.make('gym_causal:causal-v0')
    randenv.reset()
    randenv.set_probs(probs1, probs2)
    randenv.set_rewards(np.array([0, 1]))

    randag = ga.RandomAgent(randenv.action_space,
                  randenv.observation_space,
                  randenv.reward_space)
    randrewards = np.zeros(episode_count*11)
    randtotal_reward = 0
    
    l = 0
    plays1 = 0
    winsplays1 = 0
    p_win_play1 = np.zeros(episode_count*11)
    for i in range(0, episode_count):
        randenv.reset()
        done = False
        while not done:
            # random agent
            action = randag.act()
            #print("action", action)
            ob, reward, done, _ = randenv.step(action)
            #print("reward", reward)
            #print(randenv.time)
            if (action == 1) and done:
                plays1 += 1
                if (reward == 1):
                    winsplays1 += 1
            if (plays1 != 0):
                p_win_play1[i] = winsplays1/plays1
            randtotal_reward += reward
            #print(randtotal_reward)
            randrewards[l] = randtotal_reward

            l += 1


        randenv.close()
    return np.stack(p_win_play1, axis = 0)

def main(agent_type, prob_number):
    list_probs = np.array([[0.8,0.2],
                        [0.5,0.5],
                        [0.2, 0.8],
                        [1.0, 0.0],
                        [0.0, 1.0]])
    probs = list_probs[prob_number]

    if (agent_type == "r"):
        p = [get_data_for_random_agent(probs, episode_count = 100) for i in range(10)]
    elif (agent_type == "c"):
        p = [get_data_for_causal_agent(probs, episode_count = 100) for i in range(10)]
    p = np.stack(p, axis = 0)
    n, m = np.shape(p)
    means = np.asarray([np.mean(p[:, j]) for j in range(m)])
    stds = np.asarray([np.std(p[:, j]) for j in range(m)])

    x = np.arange(m)

    plt.figure()
    plt.plot(x[:100], means[:100])
    plt.fill_between(x[:100], means[:100] - stds[:100], means[:100] + stds[:100], alpha = 0.2)
    plt.title('P(Win|Play)')
    plt.show()

def print_all_main(agent_type):
    
    plt.figure()
    for i in range(4):
        
        list_probs = np.array([[0.8,0.2],
                                [0.7,0.3],
                                [0.5, 0.5],
                            [0.2, 0.8]])
        probs = list_probs[i]

        if (agent_type == "r"):
            p = [get_data_for_random_agent(probs, episode_count=100)-get_data_for_random_agent_not_play(probs, episode_count = 100) for i in range(10)]

        elif (agent_type == "c"):
            p = [get_data_for_causal_agent(probs, episode_count = 100) for i in range(10)]
        p = np.stack(p, axis = 0)
        n, m = np.shape(p)
        means = np.asarray([np.mean(p[:, j]) for j in range(m)])
        stds = np.asarray([np.std(p[:, j]) for j in range(m)])

        x = np.arange(m)

        plt.plot(x[:100], stds[:100])#plt.plot(x[:100], means[:100])
        means_of_stds = np.asarray([np.std(stds[j]) for j in range(m)])
        plt.fill_between(x[:100], stds[:100] - means_of_stds[:100], stds[:100] + means_of_stds[:100])
                
        #plt.fill_between(x[:100], means[:100] - stds[:100], means[:100] + stds[:100], alpha = 0.2)
    
    plt.title('Uncertainty of choice')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Plot p(Win | Play) for different agents on the Volleyball Task.')
    parser.add_argument('agent_type', metavar='t', type=str)
    parser.add_argument('probs', metavar='p', type=int)
    parser.add_argument('all_probs', metavar='a', type=str)
    args = parser.parse_args()

    
    if (args.all_probs == "all"):
        print_all_main(args.agent_type)
    else:
        main(args.agent_type, args.probs)
