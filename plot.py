import numpy as np
import gym_causal

def get_data_for_plot(different_probs):
    # PARAMETERS
    list_rewards = []
    episode_count = 100 # total number of game is episode_count*10 or 11?
    # for the plot
    x = np.arange(1, episode_count*11, 1)

    for j in range(len(different_probs)):
        # INIT PARAMETERS
        probs1, probs2 = different_probs[j]
        reward = 0
        done = False
        temp = 1000000
        #INIT ENV
        env = gym.make('gym_causal:causal-v0')
        env.reset()
        env.set_probs(probs1, probs2)
        env.set_rewards(np.array([0, 1]))

        #CREATE AGENT
        agent = Agent(env.action_space,
                      env.observation_space,
                      env.reward_space)

        # INIT REWARD RECORD
        rewards = np.zeros(episode_count*11)
        total_reward = 0

        #random agent
        randenv = gym.make('gym_causal:causal-v0')
        randenv.reset()
        randenv.set_probs(probs1, probs2)
        randenv.set_rewards(np.array([0, 1]))

        randag = randomAgent(env.action_space,
                      env.observation_space,
                      env.reward_space)
        randrewards = np.zeros(episode_count*11)
        randtotal_reward = 0

        j = 0
        l = 0
        plays = 0
        winsplays = 0
        p_win_play = []

        plays1 = 0
        winsplays1 = 0
        p_win_play1 = []
        for i in range(1, episode_count+1):
            # RESET BEFORE EPISODE
            env.reset()
            randenv.reset()
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
                    p_win_play.append(winsplays/plays)
                # update agent memory & causal graph
                agent.update_memory(action, ob, reward, done)

                total_reward += reward
                rewards[j] = total_reward

                j += 1


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
                    p_win_play1.append(winsplays1/plays1)
                randtotal_reward += reward
                #print(randtotal_reward)
                randrewards[l] = randtotal_reward

                l += 1

        list_rewards.append(rewards)
        env.close()
        randenv.close()
        return p_win_play, p_win_play1
