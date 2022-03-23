'''
The file that contains the training utility functions
'''

def run_agent(agent, environment, portfolio):
    '''
    A function that trains any agent in any environment with any portfolio
    '''
    history = {"cash": [], "coins": []}
    is_done = False
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

        current_price = environment.get_price_at(i)
        current_cash = portfolio.portfolio_cash_
        current_coins = portfolio.portfolio_coin_
        action = agent.agent_step(current_reward,state,current_cash,current_coins,current_price)

        # Save in history to watch later
        history["cash"].append(current_cash)
        history["coins"].append(current_coins)

        if i%1000 == 1:
            print(f"Action: {action}\n")
    print(f"Final holdings: {portfolio.get_current_holdings(environment.get_current_price())}")
    return history
