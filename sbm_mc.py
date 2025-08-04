import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# Function to Grab Final Trading Day in Year
def final_day(data_set, year):

    # Grab Year
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    new_data = data_set.loc[start:end].copy()

    date = new_data.index[-1]

    return date


# Funtion to Run a Monte Carlo Simulation when given stock ticker
def sbm(ticker, interval="1d", backtest_year=2023, simulation_number=10000):
    # Load In Stock
    stock_data = yf.download(ticker, start="2020-01-01", end="2025-01-01", interval = interval , auto_adjust = False)

    # Create Backtest Window
    backtest_start = f"{backtest_year}-01-01"
    backtest_end = f"{backtest_year}-12-31"
    backtest_data = stock_data.loc[backtest_start:backtest_end].copy()

    # Calculate Daily and Log Return
    backtest_data.loc[:, "Daily Return"] = backtest_data["Adj Close"].pct_change()
    backtest_data.loc[:, "Log Return"] = np.log(backtest_data["Adj Close"]/backtest_data["Adj Close"].shift(periods = 1))
    backtest_clean = backtest_data.dropna()

    # Calculate Drift and Volatility
    drift = backtest_clean["Log Return"].mean()
    Volatility = backtest_clean["Log Return"].std()
    print(f'{backtest_year} Drift: {drift}')
    print(f'{backtest_year} Volatility: {Volatility }')

    # Monte Caro Sumulaton, (Simulate Many Paths)
    mu = drift
    sigma = Volatility
    p_0 = backtest_clean["Adj Close"].iloc[-1]  # Change this to manip start data and pred year
    t = len(backtest_clean["Adj Close"])
    dt = 1/t
    all_paths = np.zeros((simulation_number, t))


    for i in range(simulation_number):
        S = np.zeros(t)
        S[0] = p_0
        for j in range(1,t):
            rand = np.random.normal(0,1)
            S[j] = S[j-1] * np.exp((mu - .5 * sigma**2)* dt + sigma * np.sqrt(dt)*rand)
        all_paths[i] = S

    # Summary Statistics
    last_day = final_day(stock_data, backtest_year+1) # Change this to change what year we are displaying!!!!!
    final_prices = all_paths[:, -1]
    mean_final_price = np.mean(final_prices)
    median_final_price = np.median(final_prices)
    percentile_5 = np.percentile(final_prices, 5)
    percentile_95 = np.percentile(final_prices, 95)
    max_final_price = np.max(final_prices)
    min_final_price = np.min(final_prices)
    actual_final_price = stock_data["Adj Close"].loc[last_day] # FIX THIS!! CREATE FINAL PRICE FUNCTION

    # Comparison
    print(f"Min predicted price: {min_final_price}")
    print(f"Median predicted price: {median_final_price}")
    print(f"Mean predicted price: {mean_final_price}")
    print(f"Max predicted price: {max_final_price}")
    print(f"Actual Price: {actual_final_price}")
    print(f"Start Positon: {p_0}")

    # Plot the results
    plt.figure(figsize=(10,6))
    plt.plot(all_paths.T, color='blue', alpha=0.1)  # Transparency for better visibility
    plt.title(f'Monte Carlo Simulation of {ticker} Stock Price for {backtest_year+1}')
    plt.xlabel('Time (days)')
    plt.ylabel('Stock Price (USD)')
    plt.show()

   
    
