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
import src.analytics_engine as ana
import src.alternative_data as alt
importlib.reload(alt)
importlib.reload(ana)

def generate_final_report( report_name="final_ols", writer=None):
    """
    Automates the generation of a high-fidelity Excel Dashboard.
    """
    output_dir = "reports/single_reports"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Folder '{output_dir}' created.")
    final_report = 0
    output_file = os.path.join(output_dir, f"{report_name.upper()}_report.xlsx")
    sheet_name = f'Dashboard_{report_name}'

    path_comm_ols = f"processed_data/comm_analysis/all_commodities/model_results"

    if report_name == "final_ols":
        path_model = f"processed_data/{report_name}/model_results"
        path_valid = f"processed_data/{report_name}/validation"
        path_plots = f"plots/{report_name}"
        granger_dir = f"processed_data/final_ols/validation/{report_name}_granger"
        final_report = 1

    else: 
        path_model = f"processed_data/comm_analysis/{report_name}/model_results"
        path_valid = f"processed_data/comm_analysis/{report_name}/validation"
        path_plots = f"plots/comm_analysis/{report_name}"
        granger_dir = f"{path_valid}/{report_name}_granger"

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
    note_fmt  = workbook.add_format({'font_size': 11, 'italic': True, 'font_color': '#5A5A5A', 'text_wrap': True, 'valign': 'top'})
    big_header_fmt = workbook.add_format({'bold': True, 'font_size': 24, 'font_color': "#1a4266"})
    header_fmt  = workbook.add_format({'bold': True, 'font_size': 22, 'font_color': '#1f4e78'})
    num_fmt = workbook.add_format({'bold': False, 'font_size': 16, 'font_color': '#000000', 'num_format': '#,##0.0000'})
    sci_fmt = workbook.add_format({'bold': False, 'font_size': 16, 'font_color': '#000000', 'num_format': '0.00E+00'})
    col_header_fmt = workbook.add_format({'bold': True, 'font_size': 16, 'font_color': '#1f4e78', 'bg_color': '#D9E1F2', 'border': 1})
    row_index_fmt = workbook.add_format({'bold': True, 'font_size': 16, 'font_color': '#000000'})
    color_fmts = make_color_formats(workbook)  # formats imported from excel_formatting.py

    try:
        # Titles and notes
        sheet.write('A2', f"Sectorial Statistical Performance & Econometric Analysis ({report_name})", title_fmt)
        sheet.write('B3', "Explanatory power of the model (0.067 = 6.7% of variance explained.)", note_fmt)
        sheet.write('D3', "Overall significance. If < 0.05, the model is valid.", note_fmt)
        sheet.write('E3', "Tests for autocorrelation. Values ~2.0 indicate a healthy time-series.", note_fmt)
        sheet.write('F3', "Residual normality. < 0.05 means non-normal errors (standard in high-frequency finance)", note_fmt)
        sheet.write('G3', "Model selection criteria. Lower is better when comparing different versions of this model.", note_fmt)
        sheet.write('E36', "Note: OLS Coefficients with P-values < 0.05 are statistically significant.", note_fmt)
        sheet.write('D49', "Series is stationary if p-value < 0.05 (Null hypothesis rejected).", note_fmt)
        sheet.write('C60', "VIF scale: 1 = No correlation | >5 = High multicollinearity risk.", note_fmt)
        sheet.write('D71', "Note: Granger test validates that current time-shifts are optimal. Absence of significance in other lags proves the model's lead-lag setup is correctly calibrated", note_fmt)

        images_config = {
            'A9': {'path': f"{path_plots}/OLS_Impact_{report_name}.png",'scale': 1.00},
            'J5': {'path': f"{path_plots}/{report_name}_residuals.png",'scale': 0.90}
        }
        for cell, info in images_config.items():
            if os.path.exists(info['path']):
                sheet.insert_image(cell, info['path'], {
                    'x_scale': info['scale'],
                    'y_scale': info['scale'],
                    'object_position': 3,
                })
        sheet.set_column(0, 19, 20, num_fmt)

        tables_to_process = [
            (f"{path_model}/{report_name}_ols_metrics.csv", 4, 0, 'OLS Results', True,  False),
            (f"{path_model}/{report_name}_ols_coefficients.csv",  36, 0, 'OLS Coefficients', True,  False),
            (f"{path_comm_ols}/all_commodities_top_errors.csv",  49, 9, 'Residuals Top Errors (Before Trends)', True,  False),
            (f"{path_valid}/{report_name}_adf_test.csv", 49, 0, 'Stationarity Test (ADF)', False, True ),
            (f"{path_valid}/{report_name}_vif_test.csv", 60, 0, 'Multicollinearity Test (VIF)', False, False),
        ]
        if final_report == 1: 
            tables_to_process.append((f"{path_model}/{report_name}_top_errors.csv",  49, 12, 'Residuals Top Errors (After Trends)', True,  False))

        granger_files = sorted(glob.glob(f"{granger_dir}/*.csv"))

        tables_per_row = 2
        col_spacing = 7
        row_spacing = 9
        base_row = 72
        current_row = base_row
        current_col = 0
        tables_in_row = 0
        max_rows_in_current_row = 0

        sheet.write(base_row - 2, 0, "Granger Causality Tests", big_header_fmt)
        sheet.write(base_row - 1, 0, "Null hypothesis: predictor does NOT Granger-cause Mitsubishi returns.", note_fmt)
        sheet.set_footer('&L&K909090&8Victor Valle Solar&R&K909090&8© MakroMetriks Insights')

        # Fit to page for PDF export
        sheet.set_landscape()
        sheet.fit_to_pages(1, 0)
        sheet.set_margins(left=0.25, right=0.25, top=0.75, bottom=0.75)
        sheet.set_paper(8)
        sheet.set_h_pagebreaks([48, 89, 107])

        for i, gpath in enumerate(granger_files):
            df_g = pd.read_csv(gpath)
            predictor_name = df_g['Predictor'].iloc[0]

            table_height = 1 + 1 + len(df_g) + 2

            write_formatted_table(
                sheet, writer, df_g, current_row, current_col, f"Granger · {predictor_name}",
                header_fmt, col_header_fmt, row_index_fmt, sheet_name,
                num_fmt, sci_fmt,
                has_named_index=False,
                color_fmts=color_fmts,
                adf_mode=False,
            )

            max_rows_in_current_row = max(max_rows_in_current_row, table_height)
            tables_in_row += 1
            current_col   += col_spacing

            if tables_in_row >= tables_per_row:
                current_row += max_rows_in_current_row  # dynamic spacing
                current_col = 0
                tables_in_row = 0
                max_rows_in_current_row = 0

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
        print(f"Critical error while processing {report_name}: {e}")
        traceback.print_exc()

    if _standalone:
        writer.close()
        print(f"File generated at: {os.path.abspath(output_file)}")

def generate_master_report(sector_names, final_reports, trends_name = "events", output_name = "MASTER_report"):
    """
    Consolidates all dashboards into a single Excel workbook.
    Each report gets its own sheet tab.
    """
    output_dir  = "reports"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join("reports", f"{output_name}.xlsx")

    writer = pd.ExcelWriter(output_file, engine='xlsxwriter',
                             engine_kwargs={'options': {'nan_inf_to_errors': True}})

    print("Generating master report...")

    for report in final_reports:
        print(f"  → Sheet: {report}")
        generate_final_report(report, writer=writer)

    for sector in sector_names:
        print(f"  → Sheet: {sector}")
        ana.generate_sector_report(sector, writer=writer)

    print(f"  → Sheet: {trends_name}")
    alt.generate_trends_report(trends_name, writer=writer)

    writer.close()
    print(f"\nMaster report generated at: {os.path.abspath(output_file)}")