import pandas as pd

class TradingDataLoader():
    '''
    A class to load our trading data in a pandas data frame
    '''
    def __init__(self, crypto_pair="ETH/USDT", time_step="hourly", start_date="2021-01-01 00:00:00", end_date="2021-12-31 23:00:00"):
        self.crypto_pair_ = crypto_pair
        self.time_step_ = time_step
        self.start_date_ = start_date
        self.end_date_ = end_date

        filename = "_".join(crypto_pair.lower().split("/")) + "_" + time_step
        self.data_ = self.__load_data_set(filename, start_date, end_date)
        
    def __load_data_set(self, filename, start_date, end_date):
        '''
        A private method to load a pandas data frame with our crypto data in it
        '''
        path = f"./data/{filename}.csv"
        df = pd.read_csv(path)
        df.drop(labels=["unix", "symbol"], axis=1, inplace=True)
        df.sort_values("date", inplace=True)
        df.set_index("date", inplace=True)
        return df.loc[start_date:end_date]

    def data(self):
        ''' A public method that returns the loaded data set
        '''
        return self.data_

if __name__ == "__main__":
    dataloader = TradingDataLoader()
    data = dataloader.data()
    print(data.tail(5))
    print(len(data))
