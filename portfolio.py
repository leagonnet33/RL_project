import numpy as np

METRICS = ["coin", "cash", "total_value", "is_holding_coin", "return_since_entry"]
SPREAD = .01

class BasePortfolio:
    '''
    A base class with minimal methods to implement when defining a portfolio class
    '''
    def __init__(self) -> None:
        pass

    def __buy(self, current_price):
        '''
        The private method that corresponds to buying cryptos
        '''
        raise NotImplementedError
    
    def __sell(self, current_price):
        '''
        The private method that corresponds to selling cryptos
        '''
        raise NotImplementedError
    
    # reset portfolio
    def reset(self):
        '''
        A method to reset the portfolio
        '''
        raise NotImplementedError

    def apply_action(self, current_price, action):
        '''
        A method to apply an action (either buy, sell or hold) to the portfolio and
        update the internal state after the action.
        '''
        raise NotImplementedError


class Portfolio:
    '''
    The portfolio object. It is part of the environment and describes what the agent owns
    at every time step, mainly.
    '''
    def __init__(self, spending_limit=20_000, num_coins_per_order=1., metrics=METRICS, verbose=False, final_price=0.0, spread=SPREAD):
        self.verbose_ = verbose
        self.final_price_ = final_price
        self.portfolio_coin_ = 0.0
        self.portfolio_cash_ = 0.
        self.spending_limit_ = spending_limit
        self.num_coins_per_order_ = num_coins_per_order
        self.metrics_ = metrics
        
        # Creating the state dictionary
        self.state_dict_ = {}
        self.state_dict_["coin"] = self.portfolio_coin_
        self.state_dict_["cash"] = self.portfolio_cash_
        self.state_dict_["total_value"] = self.portfolio_cash_
        self.state_dict_["is_holding_coin"] = 0
        self.state_dict_["return_since_entry"] = 0
        
        self.bought_price_ = 0.0
        self.cash_used_ = 0.0
        self.spread_ = spread # 1 bps
    
    def get_current_value(self, current_price):
        '''
        Private method to get the current total value of the portfolio
        '''
        sell_price = current_price * (1 - self.spread_)
        return self.portfolio_coin_ * sell_price + self.portfolio_cash_

    def get_returns_percent(self, current_price):
        '''
        A private method that return the returns since the experiment started
        '''
        if self.cash_used_ == 0.0:
            return 0.0
        return 100 * (self.get_current_value(current_price) - self.cash_used_) / self.cash_used_

    def __buy(self, current_price):
        '''
        The private method that corresponds to buying cryptos
        '''
        if not current_price:
            return 0

        buy_price = current_price * (1 + self.spread_)
        coin_to_buy = self.num_coins_per_order_

        if self.verbose_:
            print(f"original coin: {self.portfolio_coin_}, original cash: {self.portfolio_cash_}, price: {buy_price}, original cash used: {self.cash_used_}")

        self.portfolio_coin_ += coin_to_buy
        self.cash_used_ += coin_to_buy * buy_price

        if self.verbose_:
            print(f"coin to buy: {coin_to_buy}, coin now: {self.portfolio_coin_}, cash now: {self.portfolio_cash_}, cash used now: {self.cash_used_}")

        return coin_to_buy, buy_price

    def __sell(self, current_price):
        '''
        The private method that corresponds to selling cryptos
        '''
        if not current_price:
            return 0

        sell_price = current_price * (1 - self.spread_)

        # There's a min so that we don't sell more coins than available
        coin_to_sell = min(self.num_coins_per_order_, self.portfolio_coin_)
        
        if self.verbose_:
            print(f"original coin: {self.portfolio_coin_}, original cash: {self.portfolio_cash_}, price: {sell_price}, original cash used: {self.cash_used_}")

        self.portfolio_coin_ -= coin_to_sell
        self.portfolio_cash_ += coin_to_sell * sell_price

        if self.verbose_:
            print(f"coin to sell: {coin_to_sell}, coin now: {self.portfolio_coin_}, cash now: {self.portfolio_cash_}, cash used now: {self.cash_used_}")
 
        return coin_to_sell, sell_price

    def __check_spending_limit(self, current_price):
        '''
        A method to make sure that we're not about to override the spending limit
        we initially set
        '''
        buy_price = current_price * (1 + self.spread_)
        should_proceed = (self.cash_used_ + buy_price) <= self.spending_limit_
        return should_proceed

    def apply_action(self, current_price, action):
        '''
        A method to apply an action (either buy, sell or hold) to the portfolio and
        update the internal state after the action.
        '''
        assert action in [0, 1, 2], "Action should be one of: 0, 1 or 2"

        self.state_dict_["total_value"] = self.get_current_value(current_price)

        if self.verbose_:
            print("Action start", action, "Total value before action", self.state_dict_["total_value"])

        # BUY
        if action == 1:
            if self.__check_spending_limit(current_price):
                coin_to_buy, _ = self.__buy(current_price)
                if coin_to_buy <= 0:
                    # HOLD
                    action = 0
        # SELL
        elif action == 2:
            coin_to_sell, _ = self.__sell(current_price)
            if coin_to_sell <= 0:
                # HOLD
                action = 0
        
        # Update states
        self.state_dict_["coin"] = self.portfolio_coin_
        self.state_dict_["cash"] = self.portfolio_cash_
        self.state_dict_["total_value"] = self.get_current_value(current_price)
        self.state_dict_["is_holding_coin"] = (self.portfolio_coin_ > 0) * 1
        self.state_dict_["return_since_entry"] = self.get_returns_percent(current_price)
            
        return action

    def reset(self):
        '''
        A method to reset the portfolio
        '''
        self.__init__(spending_limit=self.spending_limit_, num_coins_per_order=self.num_coins_per_order_, metrics=self.metrics_, verbose=self.verbose_, final_price=self.final_price_, spread=self.spread_)
        
    def get_states(self, metrics=None):
        '''
        A method to return the internal portfolio state
        '''
        if not metrics:
            metrics = self.metrics_
        return np.array([self.state_dict_[metric] for metric in metrics])

    def get_current_holdings(self, current_price):
        '''
        A method that returns a string describing the current position of the portfolio
        '''
        return f"{self.portfolio_coin_: .2f} coins, {self.portfolio_cash_: .2f} cash, {self.get_current_value(current_price): .2f} current value, {self.get_returns_percent(current_price): .2f}% returns"


if __name__ == "__main__":
    from loader import TradingDataLoader
    from environments import TradingBotEnv

    ## TEST PORTFOLIO METHODS ##
    data = TradingDataLoader().data()
    env = TradingBotEnv(data)

    # portfolio.apply_action()
    portfolio = Portfolio(verbose=True)
    actions = [1, 1, 1, 0, 2, 0, 2, 1]
    for action in actions:
        current_price = env.get_current_price()
        portfolio.apply_action(current_price, action)
        _ = env.step()

    # # portfolio.reset() and portfolio.get_current_holdings()
    # portfolio = Portfolio(verbose=False)
    # actions = [1, 1, 1, 0, 2, 0, 2, 1]
    # for action in actions:
    #     current_price = env.get_current_price()
    #     portfolio.apply_action(current_price, action)
    #     _ = env.step()
    # portfolio.reset()
    # env.reset()
    # print(portfolio.get_current_holdings(env.get_current_price()))

    # # portfolio.get_states()
    # portfolio = Portfolio(verbose=False)
    # actions = [1, 1, 1, 0, 2, 0, 2, 1]
    # for action in actions:
    #     current_price = env.get_current_price()
    #     portfolio.apply_action(current_price, action)
    #     _ = env.step()
    # print(portfolio.get_states(metrics=METRICS))
    # print(portfolio.get_states(metrics=["coin", "cash"]))

    # # portfolio.get_current_holdings()
    # portfolio = Portfolio(verbose=False)
    # actions = [1, 1, 1, 0, 2, 0, 2, 1]
    # for action in actions:
    #     current_price = env.get_current_price()
    #     portfolio.apply_action(current_price, action)
    #     _ = env.step()
    # print(portfolio.get_current_holdings(env.get_current_price()))
