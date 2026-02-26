# MakroMetriks Insights · Victor Valle Solar
# github.com/vivaso86/MakroMetriks-Insights

import pandas as pd
from pytrends.request import TrendReq
import importlib
import time
import random
import hashlib
import matplotlib.pyplot as plt
import yfinance as yf
import os
import numpy as np
import glob
from src.excel_formatting import make_color_formats, write_formatted_table

#Download volume and volatility
def vol_volatility_extractor(ticker, date_start, date_end):
    data = yf.download(ticker, start=date_start, end=date_end)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data.columns = [str(col).strip() for col in data.columns]

    col_precio = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    
    df = data.copy()
    df["Returns"] = np.log(df[col_precio] / df[col_precio].shift(1))
    df["Volatility"] = df["Returns"].rolling(window=21).std() * np.sqrt(252)

    df_market = df[[col_precio, 'Volume', 'Volatility']].dropna()
    
    df_market.columns = ['Adj Close', 'Volume', 'Volatility']

    path = f"processed_data/sentiment/events"
    os.makedirs(path, exist_ok=True)

    df_market.to_csv(f"{path}/vol_volat.csv")
    return df_market

#Download keywords data from google trends
def trend_extractor(date_list, pytrends, kw_list, geo, delIspartial, par_name):
    # Sort event dates to ensure chronological processing
    date_list = dict(sorted(date_list.items()))
    total_results = []
    print("\nInitializing data extraction (shifted datetime)")
    
    for date_str in date_list:
        try:
            # Define a 31-day window around the event (15 days before and after)
            actual_date = pd.to_datetime(date_str)
            start_date = (actual_date - pd.Timedelta(days=15)).strftime('%Y-%m-%d')
            end_date = (actual_date + pd.Timedelta(days=15)).strftime('%Y-%m-%d')
            
            date_range = f"{start_date} {end_date}"
            print(f"-> Analyzing event {date_str}...")

            event_kws = []

            # Request data for each keyword individually to maintain resolution
            for kw in kw_list:
                pytrends.build_payload([kw], cat=0, timeframe=date_range, geo=geo)
                df_temp = pytrends.interest_over_time()

                if not df_temp.empty:
                    if delIspartial == 1 and 'isPartial' in df_temp.columns:
                        df_temp = df_temp.drop(columns=['isPartial'])
                    
                    event_kws.append(df_temp)

                # Dynamic sleep to prevent IP blocking from Google's servers
                time.sleep(random.randint(15, 30)) 
            
            if event_kws:
                # Merge all keywords for the current event into a single DataFrame
                df_combined = pd.concat(event_kws, axis=1)
                df_combined = df_combined.loc[:,~df_combined.columns.duplicated()]
                df_combined['original_event'] = date_str
                total_results.append(df_combined)

        except Exception as e:
            if "429" in str(e):
                print(f"---Google Ban Detected (429)--- Sleeping for 5 minutes...")
                time.sleep(300)
            else:
                print(f"Error at {date_str}: {e}")
                time.sleep(5)
            
    if total_results:
        df_trends = pd.concat(total_results).reset_index()
        if 'date' in df_trends.columns:
            df_trends = df_trends.rename(columns={'date': 'Date'})

        path = f"processed_data/sentiment/events"
        os.makedirs(path, exist_ok=True)
        output_path = f"{path}/{par_name}.csv"
        df_trends.to_csv(output_path, index=False)

        print(f"\nExtraction complete. File saved at {output_path}")
        return df_trends
    
    print("Err. No data collected.")
    return None

#Individual trend data cleaning
def process_keyw_list(df, index_name):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    kw_cols = [c for c in df.columns if c != 'original_event']
    subset = df[kw_cols].fillna(0)
    
    subset_norm = (subset - subset.min()) / (subset.max() - subset.min())
    combined_series = subset_norm.mean(axis=1)
    index_smooth = combined_series.rolling(window=3, center=True, min_periods=1).mean()
        
    return index_smooth.rename(index_name)

#Clean and merge all trends
def clean_and_index_keyw(df_internal=None, df_external=None, df_panic=None):
    to_concat = []
    if df_internal is not None:
        idx_internal = process_keyw_list(df_internal, 'Internal_Index')
        to_concat.append(idx_internal)
        
    if df_external is not None:
        idx_external = process_keyw_list(df_external, 'External_Index')
        to_concat.append(idx_external)
        
    if df_panic is not None:
        idx_panic = process_keyw_list(df_panic, 'Panic_Index')
        to_concat.append(idx_panic)

    if not to_concat:
        print("Warning: No DataFrames provided to clean_and_index_keyw.")
        return pd.DataFrame()

    path = f"processed_data/sentiment"
    os.makedirs(path, exist_ok=True)

    final_df = pd.concat(to_concat, axis=1)
    final_df = final_df.fillna(0)
    final_df.to_csv(f"{path}/trends_merged.csv")
    
    return final_df  

#Process volume, volatility and trends data for graph representation
def vol_volatility_trends_graph(vol_df, trends_df, ols_date_keys):
    #Data Preprocessing and Indexing
    if 'Date' in vol_df.columns:
        vol_df['Date'] = pd.to_datetime(vol_df['Date'])
        vol_df.set_index('Date', inplace=True)
    vol_df.index = pd.to_datetime(vol_df.index)
    vol_df = vol_df[~vol_df.index.duplicated(keep='first')]

    date_col_t = 'Date' if 'Date' in trends_df.columns else 'Date'
    if date_col_t in trends_df.columns:
        trends_df[date_col_t] = pd.to_datetime(trends_df[date_col_t])
        trends_df.set_index(date_col_t, inplace=True)
    trends_df.index = pd.to_datetime(trends_df.index)
    trends_df = trends_df[~trends_df.index.duplicated(keep='first')]

    verdicts = []

    for event_date_str in ols_date_keys:
        event_dt = pd.to_datetime(event_date_str)
        # Define visualization and baseline windows
        start_date = event_dt - pd.Timedelta(days=15)
        end_date = event_dt + pd.Timedelta(days=15)
        baseline_start = event_dt - pd.Timedelta(days=31)
        baseline_end = event_dt - pd.Timedelta(days=1)

        m_seg = vol_df.loc[start_date:end_date].copy()
        t_seg = trends_df.loc[start_date:end_date].copy()

        if m_seg.empty:
            print(f"Not enough data for event: {event_date_str}")
            continue
        
        # Statistical Significance (Z-Score) Calculation
        baseline_data = trends_df.loc[baseline_start:baseline_end]
        
        val_int = trends_df.loc[event_dt, 'Internal_Index']
        val_ext = trends_df.loc[event_dt, 'External_Index']
        val_panic = trends_df.loc[event_dt, 'Panic_Index']
        
        z_int = (val_int - baseline_data['Internal_Index'].mean()) / (baseline_data['Internal_Index'].std() + 1e-6)
        z_ext = (val_ext - baseline_data['External_Index'].mean()) / (baseline_data['External_Index'].std() + 1e-6)
        z_panic = (val_panic - baseline_data['Panic_Index'].mean()) / (baseline_data['Panic_Index'].std() + 1e-6)
        
        #Classification Logic (Diagnostic Verdict)
        if z_int > 2.2:
            verdict = "INTERNAL (Corporate Event)"
        elif z_ext > 0.5 or (z_panic > 1.5 and z_ext > 0):
            verdict = "EXTERNAL (Market/Macro Driver)"
        elif z_int > 0.7: # Una noticia interna moderada
            verdict = "INTERNAL (Potential/Likely)"
        else:
            verdict = "Hybrid / Market Noise"
        
        verdicts.append({
            'Date': event_date_str,
            'Z_Internal': round(z_int, 4),
            'Z_External': round(z_ext, 4),
            'Z_Panic': round(z_panic, 4),
            'Verdict': verdict,
        })

        print(f"\n--- EVENT {event_date_str} DIAGNOSIS ---")
        print(f"Internal (Z-Score): {z_int:.2f}")
        print(f"External (Z-Score): {z_ext:.2f}")
        print(f"Panic/Surprise (Z-Score): {z_panic:.2f}")
        print(f"Dominant driver: {verdict}")

        #Visualization: Dual-Axis Plotting
        #Volume
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.bar(m_seg.index, m_seg['Volume'], color='silver', alpha=0.4, label='Volume')
        ax1.set_ylabel('Trading volume', color='gray')
        ax1.tick_params(axis='y', labelcolor='gray')

        ax2 = ax1.twinx()

        # Volatility (Smooth plotting for better visual identification)
        vol_smooth = m_seg['Volatility'].rolling(window=2).mean()
        ax2.plot(m_seg.index, vol_smooth, color='black', alpha=0.8, linewidth=4, label='Volatility')

        if not t_seg.empty:
            ax2.plot(t_seg.index, t_seg['Internal_Index'], color='blue', linewidth=2, label='Internal')
            ax2.plot(t_seg.index, t_seg['External_Index'], color='green', linewidth=2, label='External')
            ax2.plot(t_seg.index, t_seg['Panic_Index'], color='red', linewidth=2, label='Panic/Surprise')

        ax1.axvline(event_dt, color='red', linestyle='--', linewidth=2, label='Event day')

        ax1.set_title(f"Event {event_date_str} Analysis", fontsize=14)
        ax2.set_ylabel("Intensity Trends / Volatility")

        # Consolidated Legend placement
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))

        plt.xticks(rotation=45)
        plt.tight_layout()

        save_path = "plots/sentiment/events"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if verdicts:
            save_dir = f"processed_data/sentiment/events/verdicts"
            os.makedirs(save_dir, exist_ok=True)
            verdict_path = f"{save_dir}/{event_date_str}_event_verdicts.csv"
            pd.DataFrame([verdicts[-1]]).to_csv(verdict_path, index=False)
            print(f"\nVerdicts exported to: {verdict_path}")
        
        plt.savefig(f"{save_path}/{event_date_str}_event.png", dpi=300, bbox_inches='tight')
        plt.show()

#Download complete dataset for OLS regression
def download_trend_ols(kw_list, periods, par_name, geo='JP'):
    pytrends = TrendReq(hl='ja-JP', tz=540)
    all_blocks = []
    
    for block in periods:
        print(f"  -> Processing Block: {block}")
        block_kws = []
        
        for kw in kw_list:
            try:
                pytrends.build_payload([kw], timeframe=block, geo=geo)
                df_temp = pytrends.interest_over_time()
                
                if not df_temp.empty:
                    if 'isPartial' in df_temp.columns:
                        df_temp = df_temp.drop(columns=['isPartial'])
                    
                    block_kws.append(df_temp[[kw]])
                
                time.sleep(random.randint(10, 20))
                
            except Exception as e:
                print(f"Error downloading '{kw}': {e}")
                time.sleep(60)
        
        if block_kws:
            block_df = pd.concat(block_kws, axis=1)
            all_blocks.append(block_df)
            
    if all_blocks:
        df_final = pd.concat(all_blocks)
        
        df_final = df_final[~df_final.index.duplicated(keep='first')].sort_index()

        df_final = df_final.resample('D').asfreq().interpolate(method='linear')
        df_final = (df_final - df_final.mean()) / df_final.std()
        
        df_final = df_final.reset_index()
        df_final = df_final.rename(columns={'date': 'Date'})
        
        df_final['Date'] = pd.to_datetime(df_final['Date'])

        path = f"processed_data/sentiment/final_ols/trends"
        os.makedirs(path, exist_ok=True)
        output_path = f"{path}/{par_name}_ols.csv"
        df_final.to_csv(output_path, index=False)

        print(f"\nExtraction complete. File saved at {output_path}")
        
        return df_final

    print("Warning: No data collected for the specified periods.")
    return pd.DataFrame()

#Prepare trends dataset for OLS regression
def clean_trends_dataset(file_internal, file_external, file_panic):
    df_i = pd.read_csv(file_internal, index_col='Date', parse_dates=True)
    df_e = pd.read_csv(file_external, index_col='Date', parse_dates=True)
    df_p = pd.read_csv(file_panic, index_col='Date', parse_dates=True)
    
    df_i = df_i.drop(columns=[c for c in df_i.columns if 'Unnamed' in c])
    df_e = df_e.drop(columns=[c for c in df_e.columns if 'Unnamed' in c])
    df_p = df_p.drop(columns=[c for c in df_p.columns if 'Unnamed' in c])
    
    df_i = df_i.replace(0, np.nan).interpolate(method='linear').bfill()
    df_e = df_e.replace(0, np.nan).interpolate(method='linear').bfill()
    df_p = df_p.replace(0, np.nan).interpolate(method='linear').bfill()
    
    df_i['Internal_Index'] = df_i.mean(axis=1)
    df_e['External_Index'] = df_e.mean(axis=1)
    df_p['Panic_Index'] = df_p.mean(axis=1)
    
    df_final = pd.concat([df_i['Internal_Index'], df_e['External_Index'], df_p['Panic_Index']], axis=1)
    
    epsilon = 1e-6
    df_log = np.log((df_final + epsilon) / (df_final.shift(1) + epsilon))
    
    df_norm = (df_log - df_log.mean()) / df_log.std()
    df_norm = df_norm.shift(1)
    df_norm.dropna()

    path = f"processed_data/sentiment/final_ols"
    os.makedirs(path, exist_ok=True)
    output_path = f"{path}/trends_ols_df.csv"
    df_norm.to_csv(output_path)
    return df_norm

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

def generate_trends_report(sector_name="events", writer=None):
    output_dir = "reports/single_reports"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Folder '{output_dir}' created.")
    output_file = os.path.join(output_dir, "Trends_report.xlsx")
    sheet_name = 'Dashboard_trends'

    path_model = "processed_data/sentiment/model_results"
    path_valid = "processed_data/sentiment/validation"
    path_plots = f"plots/sentiment/{sector_name}"
    events_plot = sorted(glob.glob(f"{path_plots}/*_event.png"))
    events_ver = f"processed_data/sentiment/{sector_name}/verdicts"

    _standalone = writer is None
    if _standalone:
        os.makedirs(output_dir, exist_ok=True)
        writer = pd.ExcelWriter(output_file, engine='xlsxwriter',
                                engine_kwargs={'options': {'nan_inf_to_errors': True}})
        workbook = writer.book
    else:
        workbook = writer.book

    sheet = workbook.add_worksheet(sheet_name)
    title_fmt = workbook.add_format({'bold': True, 'font_size': 48, 'font_color': '#333F48'})
    note_fmt = workbook.add_format({'font_size': 11, 'italic': True, 'font_color': '#5A5A5A', 'text_wrap': True, 'valign': 'top'})
    header_fmt = workbook.add_format({'bold': True, 'font_size': 22, 'font_color': '#1f4e78'})
    num_fmt = workbook.add_format({'bold': False, 'font_size': 16, 'font_color': '#000000', 'num_format': '#,##0.0000'})
    sci_fmt = workbook.add_format({'bold': False, 'font_size': 16, 'font_color': '#000000', 'num_format': '0.00E+00'})
    col_header_fmt = workbook.add_format({'bold': True, 'font_size': 16, 'font_color': '#1f4e78', 'bg_color': '#D9E1F2', 'border': 1})
    row_index_fmt = workbook.add_format({'bold': True, 'font_size': 16, 'font_color': '#000000'})
    color_fmts = make_color_formats(workbook)

    try:
        sheet.write('A2', f"Sectorial Statistical Performance & Econometric Analysis ({sector_name})", title_fmt)
        sheet.write('A36', f"Historical Trends & Market Dynamics Overview", title_fmt)
        sheet.set_footer('&L&K909090&8Victor Valle Solar&R&K909090&8© MakroMetriks Insights')

        # Fit to page for PDF export
        sheet.set_landscape()
        sheet.fit_to_pages(1, 0)
        sheet.set_margins(left=0.25, right=0.25, top=0.75, bottom=0.75)
        sheet.set_paper(8)
        sheet.set_h_pagebreaks([35, 115])

        images_config = {'A8': {'path': f"{path_plots}/OLS_Impact_{sector_name}.png", 'scale': 1}}
        verdict_tables = []
        col_letters = ['A', 'G', 'M']  # 3 columnas: 0, 6, 12
        col_offsets  = [0, 6, 12]
        c = 38

        for i in range(0, len(events_plot), 3):
            for j in range(3):
                idx = i + j
                if idx >= len(events_plot):
                    break
                col_letter = col_letters[j]
                col_offset = col_offsets[j]
                date_str   = os.path.basename(events_plot[idx]).replace('_event.png', '')

                images_config[f'{col_letter}{c}'] = {'path': events_plot[idx], 'scale': 0.7}
                verdict_tables.append((
                    f"{events_ver}/{date_str}_event_verdicts.csv",
                    c + 21, col_offset,
                    f'Event Diagnosis · {date_str}',
                    False, False
                ))
            c += 26

        for cell, info in images_config.items():
            if os.path.exists(info['path']):
                sheet.insert_image(cell, info['path'], {
                    'x_scale': info['scale'],
                    'y_scale': info['scale'],
                    'object_position': 3,
                })
        sheet.set_column(0, 15, 20)

        tables_to_process = [
            (f"{path_model}/{sector_name}_ols_metrics.csv", 4, 0, 'OLS Results', True, False),
            (f"{path_model}/{sector_name}_ols_coefficients.csv", 4, 9, 'OLS Coefficients', True, False),
            (f"{path_valid}/{sector_name}_adf_test.csv", 11, 9, 'Stationarity Test (ADF)', False, True ),
            (f"{path_valid}/{sector_name}_vif_test.csv", 17, 9, 'Multicollinearity Test (VIF)',False, False),
        ] + verdict_tables

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