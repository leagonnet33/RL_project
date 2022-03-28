'''
The file that contains the training utility functions
'''
import numpy as np
import pickle as pk

def run_agent(agent, environment, portfolio):
    '''
    A function that trains any agent in any environment with any portfolio
    '''
    history = {"cash": [], "coins": []}
    is_done = False
    agent.agent_init()
    action = agent.agent_start(state=None)
    i = 0
    while not is_done:
        i += 1
        portfolio.apply_action(environment.get_current_price(), action)
        is_done, state = environment.step()
        current_reward = environment.get_reward(action)

        if i%1000 == 1:
            print(f"Iteration: {i}\n")
            print(f"Current holdings: {portfolio.get_current_holdings(environment.get_current_price())}\n")
            print(f"Reward: {environment.get_reward(action)}\n")
            print(f"Current Price: {environment.get_current_price()}\n")

        current_price = environment.get_current_price()
        current_cash = portfolio.portfolio_cash_
        current_coins = portfolio.portfolio_coin_
        action = agent.agent_step(current_reward, state, current_cash, current_coins, current_price)

        # Save in history to watch later
        history["cash"].append(current_cash)
        history["coins"].append(current_coins)

        if i%1000 == 1:
            print(f"Action: {action}\n")
    print(f"Final holdings: {portfolio.get_current_holdings(environment.get_current_price())}")
    return history

def train_dqn_agent(agent, environment, portfolio, episodes=10, batch_size=32, max_memory_size=4_000, seed=97428979, save=None):
    '''
    A function to train a dqn agent over multiple episodess
    '''
    assert max_memory_size >= batch_size, "The maximum memory size must be superior to the batch size"
    random_gen = np.random.default_rng(seed=seed)

    portfolio_history = {}
    for episode in range(episodes):
        portfolio_history[episode] = []
        _, _ = environment.reset(), portfolio.reset()
        i, processed_samples, tot_loss, is_done, memory = 0, 0, 0., False, []

        # Get initial state
        state = [environment.get_metrics(), portfolio.get_states()]
        state = np.concatenate(state)

        while not is_done:
            # Choose action
            action = agent.step(state)

            # Update the portfolio and retrieve the new state
            current_price = environment.get_current_price()
            portfolio.apply_action(current_price, action)
            portfolio_state = portfolio.get_states()

            # Update the environment and retrieve the new state as well as the reward
            is_done, env_state = environment.step()
            reward = environment.get_reward(action)

            # Build the full state
            new_state = np.concatenate([env_state, portfolio_state])

            # Update the memory
            memory.append([state, action, new_state, reward, is_done])

            # Update state and action
            state = new_state
            
            # Save the current portfolio value in history
            portfolio_history[episode].append(portfolio.get_current_value(current_price))

            if len(memory) > max_memory_size:
                memory.pop(random_gen.integers(max_memory_size))

            # We train every 100 steps
            if i % 100 == 1:
                if len(memory) < batch_size:
                    batch = memory
                else:
                    batch = random_gen.choice(np.array(memory, dtype=object), size=batch_size)

                tot_loss += agent.train(batch)
                processed_samples += len(batch)

                print(
                    f"Episode: {episode + 1}/{episodes} -  Avg. loss: {tot_loss/processed_samples: .4f}",
                    end='\r'
                    )
            i += 1
        print(f"\nEPISODE OVER: {portfolio.get_current_holdings(current_price)}")

    if save:
        with open(save, 'wb') as f:
            pickler = pk.Pickler(f)
            pickler.dump(agent.model)
            
    return portfolio_history

def test_dqn_agent(agent, environment, portfolio):
    '''
    A function to run a dqn agent through a whole episode
    '''
    portfolio_history = []
    _, _ = environment.reset(), portfolio.reset()
    i, is_done = 0, False

    # Get initial state
    state = [environment.get_metrics(), portfolio.get_states()]
    state = np.concatenate(state)

    while not is_done:
        # Choose action
        action = agent.step(state)

        current_price = environment.get_current_price()
        portfolio_history.append(portfolio.get_current_value(current_price))
        
        # Update the portfolio and retrieve the new state
        portfolio.apply_action(current_price, action)
        portfolio_state = portfolio.get_states()

        # Update the environment and retrieve the new state
        is_done, env_state = environment.step()

        # Build the full state
        state = np.concatenate([env_state, portfolio_state])

        i += 1

    print(f"\nEPISODE OVER: {portfolio.get_current_holdings(current_price)}")
    return portfolio_history
