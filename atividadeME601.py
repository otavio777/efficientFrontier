import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def average_window_volatility(data_frame, quota_name, window):

    col_name = quota_name + '_log_return'

    data_frame[col_name] = np.log((data_frame[quota_name] / data_frame[quota_name].shift(1)))

    serie_array = np.asarray(data_frame[col_name], dtype=float)

    # take out everything that's nan
    serie_array = serie_array[~np.isnan(serie_array)]

    # taking out inf and -inf
    serie_array = serie_array[serie_array < 1E20]
    serie_array = serie_array[serie_array > -1E20]

    return (np.nanstd(serie_array) * np.sqrt(window)).mean()

# returns risk (volatility with a 365-days window) from the beginning
def get_security_risk(assets_df, quota_name):

    asset_volatility = average_window_volatility(assets_df.copy(), quota_name, 252)

    return asset_volatility


def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record


def display_simulated_ef_with_random(returns, mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, weights = random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)

    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index=returns.columns, columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx], index=returns.columns, columns=['allocation'])
    min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    print("-" * 80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp, 2))
    print("Annualised Volatility:", round(sdp, 2))
    print("\n")
    print(max_sharpe_allocation)
    print("-" * 80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min, 2))
    print("Annualised Volatility:", round(sdp_min, 2))
    print("\n")
    print(min_vol_allocation)

    plt.figure(figsize=(10, 7))
    plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp, rp, marker='*', color='r', s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min, rp_min, marker='*', color='g', s=500, label='Minimum volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)
    plt.savefig('fronteira_effic.png')

bbdc_df = pd.read_csv('/home/otavio/Documentos/Unicamp/ME601/AtividadeCorrPesos/Dados Brutos/BBDC4.csv')

itub_df = pd.read_csv('/home/otavio/Documentos/Unicamp/ME601/AtividadeCorrPesos/Dados Brutos/ITUB4.csv')

mglu_df = pd.read_csv('/home/otavio/Documentos/Unicamp/ME601/AtividadeCorrPesos/Dados Brutos/MGLU3.csv')

petr_df = pd.read_csv('/home/otavio/Documentos/Unicamp/ME601/AtividadeCorrPesos/Dados Brutos/PETR4.csv')

vale_df = pd.read_csv('/home/otavio/Documentos/Unicamp/ME601/AtividadeCorrPesos/Dados Brutos/VALE3.csv')

main_dataframe = pd.DataFrame()

main_dataframe['DATE'] = pd.to_datetime(bbdc_df['Data'], format='%d.%m.%Y')

main_dataframe['BBDC4'] = bbdc_df['Último'].str.replace(',', '.').astype(float)
main_dataframe['ITBU4'] = itub_df['Último'].str.replace(',', '.').astype(float)
main_dataframe['MGLU3'] = mglu_df['Último'].str.replace(',', '.').astype(float)
main_dataframe['PETR4'] = petr_df['Último'].str.replace(',', '.').astype(float)
main_dataframe['VALE3'] = vale_df['Último'].str.replace(',', '.').astype(float)

main_dataframe = main_dataframe.set_index('DATE')

plt.figure(figsize=(14, 7))
for c in main_dataframe.columns.values:
    plt.plot(main_dataframe.index, main_dataframe[c], lw=3, alpha=0.8,label=c)
plt.legend(loc='upper left', fontsize=12)
plt.ylabel('price in $')
plt.savefig('hist_preco.png')

mean_bbdc = main_dataframe['BBDC4'].mean()
mean_itbu = main_dataframe['ITBU4'].mean()
mean_mglu = main_dataframe['MGLU3'].mean()
mean_petr = main_dataframe['PETR4'].mean()
mean_vale = main_dataframe['VALE3'].mean()

vol_bbdc = get_security_risk(main_dataframe, 'BBDC4')
vol_itbu = get_security_risk(main_dataframe, 'ITBU4')
vol_mglu = get_security_risk(main_dataframe, 'MGLU3')
vol_petr = get_security_risk(main_dataframe, 'PETR4')
vol_vale = get_security_risk(main_dataframe, 'VALE3')

variacao_df = main_dataframe.pct_change()

plt.figure(figsize=(14, 7))
for c in variacao_df.columns.values:
    plt.plot(variacao_df.index, variacao_df[c], lw=3, alpha=0.8,label=c)
plt.legend(loc='upper right', fontsize=12)
plt.ylabel('daily returns')
plt.savefig('hist_retornos.png')

corr_df = variacao_df.corr()

f = plt.figure(figsize=(19, 15))
plt.matshow(corr_df, fignum=f.number)
plt.xticks(range(corr_df.shape[1]), corr_df.columns, fontsize=14, rotation=45)
plt.yticks(range(corr_df.shape[1]), corr_df.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)
plt.savefig('correlacao.png')

returns = variacao_df
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 25000
risk_free_rate = 0.03

display_simulated_ef_with_random(returns, mean_returns, cov_matrix, num_portfolios, risk_free_rate)