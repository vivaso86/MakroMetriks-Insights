# MakroMetriks Insights · Victor Valle Solar
# github.com/vivaso86/MakroMetriks-Insights

import pandas as pd
import src.analytics_engine as ana
import src.final_report as fin

def build_ols_dataset(data_fx_ready,df_agro_ready,df_ener_ready,df_metal_ready, data_trends_ready=None, shifts: dict = None,):
    """
    Assembles the OLS dataset by joining commodity and FX data.
    Optionally includes trend indices if data_trends_ready is provided.
    Returns assembled and cleaned DataFrame ready for OLS.
    """
    default_shifts = {
        'Corn':           5,
        'LNG_Asia':       2,
        'Internal_Index': 2,
        'External_Index': 5,
    }
    if shifts:
        default_shifts.update(shifts)

    df = data_fx_ready[['Nikkei_USD']].copy()

    df = df.join([
        df_agro_ready[['Corn']].shift(default_shifts['Corn']),
        df_ener_ready[['Brent_Crude']],
        df_ener_ready[['Natural_Gas']],
        df_ener_ready[['LNG_Asia']].shift(default_shifts['LNG_Asia']),
        df_metal_ready[['Copper']],
    ], how='left')

    if data_trends_ready is not None:
        trends = data_trends_ready[['Internal_Index', 'External_Index']].copy()
        trends['Internal_Index'] = trends['Internal_Index'].shift(default_shifts['Internal_Index'])
        trends['External_Index'] = trends['External_Index'].shift(default_shifts['External_Index'])
        df = df.join(trends, how='left')

    df = df.ffill().dropna()
    print(f"OLS dataset ready: {len(df)} rows | columns: {list(df.columns)}")
    return df


def run_ols_pipeline(sector_name, data_fx_ready, df_agro_ready, df_ener_ready, df_metal_ready, data_trends_ready=None,
    target_col='Mitsubishi_USD', max_lag=5, num_errors=7, shifts: dict = None, generate_report=True):
    """
    Runs the full OLS pipeline: dataset assembly, ADF, VIF, OLS, residuals,
    Granger causality, and optionally generates the Excel report.
    Returns ols_model, top_errors and granger_results
    """
    # --- 1. Build dataset ---
    ols_data = build_ols_dataset(
        data_fx_ready, df_agro_ready, df_ener_ready, df_metal_ready,
        data_trends_ready=data_trends_ready,
        shifts=shifts,
    )

    if len(ols_data) == 0:
        print("ERROR: 0 rows after assembly. Check that date ranges overlap across all files.")
        return None, None, None

    ana.check_adf(ols_data, sector_name)
    ana.check_vif(ols_data, sector_name)
    model = ana.check_ols(data_fx_ready, ols_data, target_col, sector_name)
    top_errors = ana.check_residuals(model, target_col, sector_name, num_errors)
    df_returns = ols_data.pct_change().dropna()
    df_returns[target_col] = data_fx_ready[target_col].pct_change(fill_method=None)
    df_returns = df_returns.dropna()

    granger_results = ana.check_granger(
        df_returns,
        target_col=target_col,
        predictor_col=ols_data,
        sector_name=sector_name,
        max_lag=max_lag,
    )

    if generate_report:
        fin.generate_final_report(sector_name)

    return model, top_errors, granger_results