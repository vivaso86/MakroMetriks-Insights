import pandas as pd
import numpy as np
from itertools import combinations
import os
import io
import warnings
import importlib

import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.tsa.stattools as ts
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.excel_formatting import make_color_formats, write_formatted_table

#Fetch data functions
def download_and_convert_tickers(tickers_dict, start_date, end_date, fx_ticker, sector_name="tickers"):
    """
    Downloads historical ticker data and standardizes currency exposure to USD.
    Validates data integrity for subsequent OLS modeling.
    """
    all_tickers = list(tickers_dict.keys()) + [fx_ticker]
    data_raw = yf.download(all_tickers, start=start_date, end=end_date, auto_adjust=False)
    if data_raw is None or data_raw.empty:
        raise ValueError(f"Data ingestion failed: Verify if tickers {all_tickers} are valid and active in Yahoo Finance.")

    data = data_raw['Adj Close'].copy()
    
    data.rename(columns=tickers_dict, inplace=True)
    
    assets_to_convert = list(tickers_dict.values()) 
    new_currency = "USD"

    currency_part = fx_ticker[:-2]
    new_fx_name = f"{new_currency}{currency_part}"
    data.rename(columns={fx_ticker: new_fx_name}, inplace=True)
    
    for asset in assets_to_convert:
        base_name = asset[:-3]
        new_name = f"{base_name}{new_currency}"
        data[new_name] = data[asset] / data[new_fx_name]
    
    path = f"processed_data/comm_analysis/{sector_name}"
    sentiment_path = f"processed_data/sentiment"
    os.makedirs(path, exist_ok=True)
    
    file_path = f"{path}/dataset_{sector_name}.csv"
    sentiment_file_path = f"{sentiment_path}/dataset_{sector_name}.csv" 
    data.to_csv(file_path, index=True)
    data.to_csv(sentiment_file_path, index=True) #df is saved for sentiment module
    
    print(f"{sector_name}.csv generated at: {file_path}")
    print(f"{sector_name}.csv generated at: {sentiment_file_path}")
    
    return data

def download_comm (commodities, start_date, end_date, sector_name):
    """
    Downloads historical commodity data.
    Validates data integrity for subsequent OLS modeling.
    """

    data_comm= yf.download(list(commodities.keys()), start=start_date, end=end_date, auto_adjust=False)
    if data_comm is None or data_comm.empty:
        raise ValueError(f"Data ingestion failed: Verify if commodities {commodities} are valid and active in Yahoo Finance.")
    df_comm = data_comm['Adj Close'].copy()

    df_comm.rename(columns=commodities, inplace=True)

    path = f"processed_data/comm_analysis/{sector_name}"
    os.makedirs(path, exist_ok=True)
    df_comm.to_csv(f"{path}/{sector_name}_prices.csv", index=True)

    df_comm_shifted = df_comm.shift(1) #Data is shifted so it maches japanese datetime
    df_comm_shifted.to_csv(f"{path}/{sector_name}_shifted_prices.csv", index=True)

    
    print(f"{sector_name}_prices.csv generated at: {path}")

    return df_comm

#Graphs 
def plot_base(df, title, base, start_date, end_date, sector_name):
    """
    Normalizes and plots financial time series based on a starting value (Base 100/1000).
    Saves both the processed data and the high-resolution visualization.
    """
    if start_date == "":
        start_date = df.index.min()
    if end_date == "":
        end_date = df.index.max()
    
    df_plot = df.loc[start_date:end_date].ffill()
    if df_plot.empty:
        print("Error: Selected range doesnt contain any date")
        return
    
    df_norm = df_plot.copy()
    for col in df_norm.columns:
        first_val = df_norm[col].dropna().iloc[0]
        if first_val != 0:
            df_norm[col] = (df_norm[col] / first_val) * base
    
    df_norm.plot(figsize=(12, 6), linewidth=1.5)
    
    plt.title(title, fontweight='bold', fontsize=14)
    plt.axhline(base, color='black', linestyle='--', alpha=0.3)
    plt.xlabel("")
    plt.ylabel("")
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    data_path = f"processed_data/comm_analysis/{sector_name}/graph_data"
    os.makedirs(data_path, exist_ok=True)
    df_norm.to_csv(f"{data_path}/{sector_name}_base_{base}.csv")

    plot_path = f"plots/comm_analysis/{sector_name}"
    os.makedirs(plot_path, exist_ok=True)
    plot_path = f"{plot_path}/Base_100_{sector_name}.png"
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"{sector_name}_base_{base}.csv generated at: {data_path}/")
    print(f"Graph generated at: {plot_path}")
    return df_norm
    

def comm_cor(df, title, start_date, end_date, sector_name):
    """
    Computes and visualizes the correlation matrix based on daily returns.
    Uses percentage change to avoid spurious correlations and outputs a 
    annotated heatmap for risk exposure analysis.
    """
    if start_date == "":
        start_date = df.index.min()
    if end_date == "":
        end_date = df.index.max()

    df_plot = df.loc[start_date:end_date].ffill()
    if df_plot.empty:
        print("Error: Selected range doesnt contain any date")
        return
    
    returns = df_plot.pct_change()
    corr_matrix = returns.dropna().corr()

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f"{title} (Daily returns)")
    ax.set_xlabel("")
    ax.set_ylabel("")    
    plt.tight_layout()

    data_path = f"processed_data/comm_analysis/{sector_name}/graph_data"
    os.makedirs(data_path, exist_ok=True)
    csv_path = f"{data_path}/{sector_name}_corr_matrix.csv"
    corr_matrix.to_csv(csv_path, index=True)

    plot_path = f"plots/comm_analysis/{sector_name}"
    os.makedirs(plot_path, exist_ok=True)
    plot_path = f"{plot_path}/Daily_Corr_{sector_name}.png"
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"{sector_name}_corr_matrix.csv generated at: {data_path}/")
    print(f"Graph generated at: {plot_path}")


def plot_rolling_corr(df_main, df_assets, target, sector_name,window=60):
    """
    Computes and plots dynamic rolling correlations between a target asset and 
    sectoral drivers. Essential for identifying time-varying regime shifts and 
    the stability of statistical relationships. Recommended 60 day rolling window
    """
    combined = pd.concat([df_main[target], df_assets], axis=1).ffill()
    returns = combined.pct_change().dropna()
    
    rolling_results = pd.DataFrame(index=returns.index)

    plt.figure(figsize=(14, 7))  
    for col in df_assets.columns:
        rolling_corr = returns[target].rolling(window=window).corr(returns[col])
        rolling_results[f'corr_{target}_{col}'] = rolling_corr

        plot_data = rolling_corr.dropna()
        if not plot_data.empty:
            plt.plot(plot_data, label=f'Corr {target} vs {col}', linewidth=2)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.2, label='Strong Link')
    plt.title(f'Continuous Rolling Correlation ({window} Days) - {target}')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    
    data_path = f"processed_data/comm_analysis/{sector_name}/graph_data"
    os.makedirs(data_path, exist_ok=True)
    csv_path = f"{data_path}/{sector_name}_rolling_{window}.csv"
    rolling_results.to_csv(csv_path, index=True)

    plot_path = f"plots/comm_analysis/{sector_name}"
    os.makedirs(plot_path, exist_ok=True)
    title_clean = f"Rolling_Corr_{window}_{sector_name}"
    plot_path = f"{plot_path}/{title_clean}.png"
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"{sector_name}_rolling_{window}.csv generated at: {data_path}/")
    print(f"Graph generated at: {plot_path}")


#Checks
def check_adf(df, sector_name, mod_path = "comm_analysis"):
    """
    Performs the Augmented Dickey-Fuller (ADF) test to ensure stationarity.
    Automatically applies log-returns to price series while maintaining levels 
    for bounded indices, preventing spurious regression results in the OLS model.
    """
    results_list = []

    #Trends indices are kept in levels because they are bounded (0-100) and structurally stationary
    forced_levels = ['Internal_Index', 'External_Index', 'Panic_Index']
    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number) or col.lower() == 'date':
            continue

        if col in forced_levels:
            data_to_test = df[col]
            method = "LEVELS (Forced)"
        elif (df[col] <= 0).any():
            data_to_test = df[col]
            method = "LEVELS (Zero-detected)"
        else:
            data_to_test = np.log(df[col] / df[col].shift(1))
            method = "LOG-RETURNS"

        data_to_test = data_to_test.replace([np.inf, -np.inf], np.nan).dropna()

        if data_to_test.empty:
            continue

        result = ts.adfuller(data_to_test)
        
        status = "STATIONARY" if result[1] <= 0.05 else "NON-STATIONARY"
        
        print(f"--- ADF Test Results for: {col} ({method}) ---")
        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"p-value: {result[1]:.4f}")
        print(f"Result: {status}")
        print("-" * 40)

        results_list.append({
            'Variable': col,
            'Method': method,
            'ADF_Statistic': result[0],
            'p_value': result[1],
            'Critical_Value_5%': result[4]['5%'],
            'Is_Stationary': status
        })
    
    df_final_results = pd.DataFrame(results_list)
    if sector_name.lower() == "events":
        data_path = f"processed_data/{mod_path}/validation"
    elif sector_name.lower() == "final_ols":
        data_path = f"processed_data/{sector_name}/validation"
    else:
        data_path = f"processed_data/{mod_path}/{sector_name}/validation"
    os.makedirs(data_path, exist_ok=True)
    csv_path = f"{data_path}/{sector_name}_adf_test.csv"
    df_final_results.to_csv(csv_path, index=False)

    print(f"{sector_name}_adf_test.csv generated at: {data_path}/")
    
    return df_final_results

def check_vif(df_drivers, sector_name, mod_path = "comm_analysis"):
    """
    Evaluates multicollinearity among predictors using the Variance Inflation Factor (VIF).
    Ensures the structural integrity of the OLS model by identifying redundant drivers 
    that could inflate standard errors and bias coefficient significance.
    """
    df_numeric = df_drivers.select_dtypes(include=[np.number]).copy()
    if df_numeric.empty:
        print("Error: No numeric data to calculate VIF")
        return None
    
    X = df_numeric.ffill().pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    
    X_with_const = sm.add_constant(X)
    
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
    
    vif_report = vif_data[vif_data["Variable"] != "const"].sort_values(by="VIF", ascending=False)
    
    if sector_name.lower() == "events":
        data_path = f"processed_data/{mod_path}/validation"
    elif sector_name.lower() == "final_ols":
        data_path = f"processed_data/{sector_name}/validation"
    else:
        data_path = f"processed_data/{mod_path}/{sector_name}/validation"
    os.makedirs(data_path, exist_ok=True)
    csv_path = f"{data_path}/{sector_name}_vif_test.csv"
    vif_report.to_csv(csv_path, index=False)

    print(f"{sector_name}_vif_test.csv generated at: {data_path}/")

    print("\n--- Multicollinearity report (VIF) ---")
    print(vif_report)
    
    return vif_report

def check_ols(df_main, df_drivers, target_col, sector_name):
    """
    Executes a Multi-Factor Ordinary Least Squares (OLS) regression on standardized 
    returns. Computes relative feature importance, diagnostic metrics (Durbin-Watson, 
    Jarque-Bera), and visualizes the contribution of each macro driver to the target's variance.
    """
    df_main_num = df_main.select_dtypes(include=[np.number])
    df_drivers_num = df_drivers.select_dtypes(include=[np.number])
    Y = df_main_num[target_col].ffill().pct_change(fill_method=None)

    X_processed = pd.DataFrame(index=df_drivers.index)
    for col in df_drivers_num.columns:
        if 'Index' in col:
            X_processed[col] = df_drivers_num[col]
        else:
            X_processed[col] = df_drivers_num[col].ffill().pct_change(fill_method=None)  

    combined = pd.concat([Y, X_processed], axis=1).dropna()
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()

    Y_final = combined[target_col]
    X_final = combined.drop(columns=[target_col])
    
    X_std = (X_final - X_final.mean()) / X_final.std()
    X_std = sm.add_constant(X_std)  
    
    model = sm.OLS(Y_final, X_std).fit()
    
    coefs = model.params.drop('const')
    impact_pct = (np.abs(coefs) / np.abs(coefs).sum()) * 100

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(impact_pct)))
    impact_pct.sort_values().plot(kind='barh', color=colors)
    
    print(f"\n--- Regression summary (R-squared: {model.rsquared:.2f}) ---")
    for asset, pct in impact_pct.items():
        print(f"{asset}: {pct:.2f}% contribution to variance")
        
    plt.title(f'Determinants of {target_col} Returns (Relative Impact %)')
    plt.xlabel('Relative Weight (%)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    html_string = model.summary().tables[1].as_html()
    results_df = pd.read_html(io.StringIO(html_string), header=0, index_col=0)[0]
    
    coefs = model.params.drop('const')
    impact_pct = (np.abs(coefs) / np.abs(coefs).sum()) * 100
    results_df['Impact_Pct'] = impact_pct

    jb_test = sms.jarque_bera(model.resid)
    jb_pvalue = jb_test[1]
    
    metrics_df = pd.DataFrame([{
    'R-squared': model.rsquared,
    'Adj. R-squared': model.rsquared_adj,
    'F-statistic': model.fvalue,
    'Prob (F-stat)': model.f_pvalue,
    'Durbin-Watson': sms.durbin_watson(model.resid), 
    'Prob (Jarque-Bera)': jb_pvalue,    
    'AIC': model.aic,
    'BIC': model.bic,
    }])
    if sector_name.lower() == "events":
        data_path = "processed_data/sentiment/model_results"
        plot_path = f"plots/sentiment/{sector_name}"
    elif sector_name.lower() == "final_ols":
        data_path = f"processed_data/{sector_name}/model_results"
        plot_path = f"plots/{sector_name}"
    else:
        data_path = f"processed_data/comm_analysis/{sector_name}/model_results"
        plot_path = f"plots/comm_analysis/{sector_name}"
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    results_df.to_csv(f"{data_path}/{sector_name}_ols_coefficients.csv")
    metrics_df.to_csv(f"{data_path}/{sector_name}_ols_metrics.csv", index=False)
    plt.savefig(f"{plot_path}/OLS_Impact_{sector_name}.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()

    print(f"{sector_name}_ols_results.csv generated at: {data_path}/")
    print(f"Graph generated at: {plot_path}")
    print(model.summary())
    return model

def check_residuals(model, target_name, sector_name, num_errors):
    """
    Analyzes model prediction errors to validate OLS assumptions. 
    Performs outlier detection using a 2-sigma threshold and evaluates 
    the error distribution to identify potential non-linearity or structural breaks.
    """
    residuals = model.resid
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
    
    ax1.plot(residuals, color='purple', lw=1, alpha=0.7)
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
    
    std_res = residuals.std()
    ax1.axhline(2*std_res, color='red', linestyle=':', alpha=0.4, label='Anomaly Threshold (2σ)')
    ax1.axhline(-2*std_res, color='red', linestyle=':', alpha=0.4)
    
    ax1.set_title(f'Model Residuals Over Time - {target_name}', fontweight='bold')
    ax1.set_ylabel('Prediction Error')
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    
    sns.histplot(residuals, kde=True, ax=ax2, color='skyblue')
    ax2.set_title('Error Distribution (Residuals Analysis)')
    ax2.set_xlabel('Error Value')
    plt.tight_layout()
    
    if sector_name.lower() == "all_commodities":
        data_path = f"processed_data/comm_analysis/{sector_name}/model_results"
        plot_path = f"plots/comm_analysis/{sector_name}"
    elif sector_name.lower() == "final_ols":
        data_path = f"processed_data/{sector_name}/model_results"
        plot_path = f"plots/{sector_name}"

    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(f"{plot_path}/{sector_name}_residuals.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    top_errors = np.abs(residuals).sort_values(ascending=False).head(num_errors)
    print(f"\n--- Outlier Detection: Largest Residuals for {target_name} ---")
    print(top_errors)

    os.makedirs(data_path, exist_ok=True)
    residuals.to_csv(f"{data_path}/{sector_name}_ols_residuals.csv")
    top_errors.to_csv(f"{data_path}/{sector_name}_top_errors.csv", index=True)
    
    return top_errors

def check_granger(df, target_col, predictor_col, sector_name="final_ols",max_lag=5):
    """
    Performs the Granger Causality test to determine lead-lag relationships.
    Identifies if historical values of a predictor (e.g., Commodities) provide 
    statistically significant information to forecast a target asset's returns.
    """
    print(f"\n--- Checking Granger Causality: Does {predictor_col} 'cause' {target_col}? ---")
    all_results = {}

    for pred in predictor_col:
        print(f"\n  Predictor: {pred}")
        data = df[[target_col, pred]].dropna()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = ts.grangercausalitytests(data, maxlag=max_lag, verbose=False)

        significant_found = False
        rows = []

        for lag, test_output in results.items():
            p_value = test_output[0]['ssr_ftest'][1]
            f_stat  = test_output[0]['ssr_ftest'][0]
            status  = "YES" if p_value <= 0.05 else "NO"

            if p_value <= 0.05:
                significant_found = True

            print(f"Lag {lag}: p-value = {p_value:.4f} -> {'SIGNIFICANT' if status == 'YES' else 'not significant'}")

            rows.append({
                'Predictor': pred,
                'Target': target_col,
                'Lag': lag,
                'F_Statistic': round(f_stat, 4),
                'p_value': round(p_value, 4),
                'Significant': status,
            })

        if sector_name.lower() == "final_ols":
            save_dir = f"processed_data/{sector_name}/validation/{sector_name}_granger"
        else:
             save_dir = f"processed_data/comm_analysis/{sector_name}/validation/{sector_name}_granger"
        os.makedirs(save_dir, exist_ok=True)
        csv_path = f"{save_dir}/{pred}_granger.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  Exported: {csv_path}")

        if significant_found:
            print(f"CONCLUSION: {pred} has predictive power over {target_col}.")
        else:
            print(f"CONCLUSION: No predictive relationship found for {pred}.")

        all_results[pred] = results

    return all_results


def prepare_market_data(input_data):
    if isinstance(input_data, str):
        if not os.path.exists(input_data):
            raise FileNotFoundError(f"The file path was not found: {input_data}")
        df = pd.read_csv(input_data)
    else:
        df = input_data.copy()
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.drop_duplicates(subset='Date')
        df = df.set_index('Date')

    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except:
            raise KeyError("No 'Date' column found and Index is not convertible to Datetime.")
        
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    df.index.name = 'Date'
    return df


def generate_sector_report(sector_name, writer=None):
    output_dir = "reports/single_reports"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Folder '{output_dir}' created.")
    output_file = os.path.join(output_dir, f"{sector_name.upper()}_report.xlsx")
    sheet_name  = f'Dashboard_{sector_name}'

    path_model = f"processed_data/comm_analysis/{sector_name}/model_results"
    path_valid = f"processed_data/comm_analysis/{sector_name}/validation"
    path_plots = f"plots/comm_analysis/{sector_name}"

    _standalone = writer is None
    if _standalone:
        os.makedirs(output_dir, exist_ok=True)
        writer = pd.ExcelWriter(output_file, engine='xlsxwriter',
                                engine_kwargs={'options': {'nan_inf_to_errors': True}})
        workbook = writer.book
    else:
        workbook = writer.book

    sheet = workbook.add_worksheet(sheet_name)

    title_fmt = workbook.add_format({'bold': True, 'font_size': 26, 'font_color': '#333F48'})
    note_fmt = workbook.add_format({'font_size': 11, 'italic': True, 'font_color': '#5A5A5A', 'text_wrap': True, 'valign': 'top'})
    header_fmt = workbook.add_format({'bold': True, 'font_size': 22, 'font_color': '#1f4e78'})
    num_fmt = workbook.add_format({'bold': False, 'font_size': 16, 'font_color': '#000000', 'num_format': '#,##0.0000'})
    sci_fmt  = workbook.add_format({'bold': False, 'font_size': 16, 'font_color': '#000000', 'num_format': '0.00E+00'})
    col_header_fmt = workbook.add_format({'bold': True, 'font_size': 16, 'font_color': '#1f4e78', 'bg_color': '#D9E1F2', 'border': 1})
    row_index_fmt = workbook.add_format({'bold': True, 'font_size': 16, 'font_color': '#000000'})
    color_fmts = make_color_formats(workbook)  # formats imported from excel_formatting.py

    try:
        # Titles and notes
        sheet.write('A2', f"Sectorial Statistical Performance & Econometric Analysis ({sector_name})", title_fmt)
        sheet.write('B4', "Explanatory power of the model (0.067 = 6.7% of variance explained.)", note_fmt)
        sheet.write('D4', "Overall significance. If < 0.05, the model is valid.", note_fmt)
        sheet.write('E4', "Tests for autocorrelation. Values ~2.0 indicate a healthy time-series.", note_fmt)
        sheet.write('F4', "Residual normality. < 0.05 means non-normal errors (standard in high-frequency finance)", note_fmt)
        sheet.write('G4', "Model selection criteria. Lower is better when comparing different versions of this model.", note_fmt)
        sheet.write('E36', "Note: OLS Coefficients with P-values < 0.05 are statistically significant.", note_fmt)
        sheet.write('D73', "Series is stationary if p-value < 0.05 (Null hypothesis rejected).", note_fmt)
        sheet.write('C80', "VIF scale: 1 = No correlation | >5 = High multicollinearity risk.", note_fmt)

        images_config = {
            'G9': {'path': f"{path_plots}/Base_100_{sector_name}.png",       'scale': 0.90},
            'A9': {'path': f"{path_plots}/OLS_Impact_{sector_name}.png",     'scale': 1.00},
            'A43':{'path': f"{path_plots}/Daily_Corr_{sector_name}.png",     'scale': 0.85},
            'F43':{'path': f"{path_plots}/Rolling_Corr_60_{sector_name}.png",'scale': 0.85},
        }
        for cell, info in images_config.items():
            if os.path.exists(info['path']):
                sheet.insert_image(cell, info['path'], {
                    'x_scale': info['scale'],
                    'y_scale': info['scale'],
                    'object_position': 3,
                })
        sheet.set_column(0, 10, 20, num_fmt)

        tables_to_process = [
            (f"{path_model}/{sector_name}_ols_metrics.csv", 4, 0, 'OLS Results', True,  False),
            (f"{path_model}/{sector_name}_ols_coefficients.csv",  36, 0, 'OLS Coefficients', True,  False),
            (f"{path_valid}/{sector_name}_adf_test.csv", 73, 0, 'Stationarity Test (ADF)', False, True ),
            (f"{path_valid}/{sector_name}_vif_test.csv", 80, 0, 'Multicollinearity Test (VIF)', False, False),
        ]

        for path, s_row, s_col, s_title, has_index, is_adf in tables_to_process:
            if os.path.exists(path):
                df_temp = pd.read_csv(path, index_col=0 if has_index else None)
                write_formatted_table(
                    sheet, writer, df_temp, s_row, s_col, s_title,
                    header_fmt, col_header_fmt, row_index_fmt, sheet_name,
                    num_fmt, sci_fmt,
                    has_named_index=has_index,
                    color_fmts=color_fmts,
                    adf_mode=is_adf,
                )

    except Exception as e:
        import traceback
        print(f"Critical error while processing {sector_name}: {e}")
        traceback.print_exc()

    if _standalone:
        writer.close()
        print(f"File generated at: {os.path.abspath(output_file)}")