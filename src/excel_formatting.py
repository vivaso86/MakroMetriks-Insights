import numpy as np
import pandas as pd

def _rule_adj_r2(v):
    """Adj. R-squared: >0.02 green, <0.01 red, >0.50 red (data leakage)."""
    if v > 0.50: return 'red' # possible data leakage / spurious relationship
    elif v >= 0.02: return 'green'
    elif v < 0.01: return 'red'
    else: return None

def _rule_prob_f(v):
    """Prob (F-stat): <0.05 green, >0.10 red."""
    if v < 0.05: return 'green'
    elif v > 0.10: return 'red'
    else: return None 

def _rule_durbin_watson(v):
    """Durbin-Watson: 1.8-2.2 green, 1.5-1.8 or 2.2-2.5 orange, <1.5 or >2.5 red."""
    if 1.8 <= v <= 2.2: return 'green'
    elif 1.5 <= v < 1.8 or 2.2 < v <= 2.5: return 'orange'
    else: return 'red'

def _rule_prob_jb(v):
    """Prob (Jarque-Bera): >0.05 green (normal residuals), 0.001-0.05 orange, <0.001 red."""
    if v > 0.05: return 'green'
    elif 0.001 <= v <= 0.05: return 'orange'
    else: return 'red'

def _rule_adf_pvalue(v):
    """ADF p-value: <0.05 green (stationary), >0.05 red (unit root)."""
    return 'green' if v < 0.05 else 'red'

def _rule_adf_statistic(v, critical):
    """ADF Statistic: must be MORE negative than the 5% critical value."""
    return 'green' if v < critical else 'red'

def _rule_coef_pvalue(v):
    """P>|t|: <0.05 green (significant), >0.10 red, grey zone 0.05-0.10."""
    if v < 0.05: return 'green'
    elif v > 0.10: return 'red'
    else: return None

def _rule_vif(v):
    """VIF: <5 green (independent), 5-10 orange (moderate), >10 red (severe multicollinearity)."""
    if v < 5.0: return 'green'
    elif v <= 10.0: return 'orange'
    else: return 'red'

# Rule maps per table type
OLS_METRICS_RULES = {
    'Adj. R-squared': _rule_adj_r2,
    'Prob (F-stat)': _rule_prob_f,
    'Durbin-Watson': _rule_durbin_watson,
    'Prob (Jarque-Bera)': _rule_prob_jb,
}
OLS_COEF_RULES = {
    'P>|t|': _rule_coef_pvalue,
}
ADF_RULES = {
    'p_value': _rule_adf_pvalue,
}
VIF_RULES = {
    'VIF': _rule_vif,
}

#Color format

def make_color_formats(workbook, font_size=16):
    """
    Creates XlsxWriter red/green color formats.
    Call ONCE per workbook, right after the other add_format() calls.

    Returns:
        dict with keys: 'green', 'red', 'green_sci', 'red_sci'
    """
    return {
        'green': workbook.add_format({
            'font_size':  font_size,
            'font_color': '#1D6A38',
            'bg_color':   '#C6EFCE',
            'bold':       True,
            'num_format': '#,##0.000',
        }),
        'red': workbook.add_format({
            'font_size':  font_size,
            'font_color': '#9C0006',
            'bg_color':   '#FFC7CE',
            'bold':       True,
            'num_format': '#,##0.000',
        }),
        'green_sci': workbook.add_format({
            'font_size':  font_size,
            'font_color': '#1D6A38',
            'bg_color':   '#C6EFCE',
            'bold':       True,
            'num_format': '0.00E+00',
        }),
        'red_sci': workbook.add_format({
            'font_size':  font_size,
            'font_color': '#9C0006',
            'bg_color':   '#FFC7CE',
            'bold':       True,
            'num_format': '0.00E+00',
        }),
        'orange': workbook.add_format({
            'font_size':  font_size,
            'font_color': '#974706',
            'bg_color':   '#FFE699',
            'bold':       True,
            'num_format': '#,##0.000',
        }),
        'orange_sci': workbook.add_format({
            'font_size':  font_size,
            'font_color': '#974706',
            'bg_color':   '#FFE699',
            'bold':       True,
            'num_format': '0.00E+00',
        }),
        'strikeout': workbook.add_format({
        'font_size':    font_size,
        'font_color':   '#9C9C9C',
        'font_strikeout': True,
        'num_format':   '#,##0.000',
    }),
        'strikeout_sci': workbook.add_format({
        'font_size':    font_size,
        'font_color':   '#9C9C9C',
        'font_strikeout': True,
        'num_format':   '0.00E+00',
    }),
    }

# Helpers

def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans column names with characters that are problematic for XlsxWriter."""
    rename_map = {}
    for col in df.columns:
        if col.strip() == "[0.025":
            rename_map[col] = "CI_lower"
        elif col.strip() == "0.975]":
            rename_map[col] = "CI_upper"
        else:
            rename_map[col] = col.replace("[", "").replace("]", "").strip()
    return df.rename(columns=rename_map)

def resolve_fmt(color_key, value, sci_threshold, color_fmts, num_fmt, sci_fmt):
    """
    Combines color + scientific notation to return the final cell format.
    If color_key is None, returns num_fmt or sci_fmt with no color applied.
    """
    use_sci = (value != 0) and (abs(value) < sci_threshold)
    if color_key in ('green', 'red', 'orange'):
        sci_key = f'{color_key}_sci'
        return color_fmts[sci_key] if use_sci else color_fmts[color_key]
    else:
        return sci_fmt if use_sci else num_fmt

# Writing engine

def write_formatted_table(sheet, writer, df: pd.DataFrame, start_row: int, start_col: int, title: str, title_fmt, header_fmt,
    index_fmt, sheet_name: str, num_fmt, sci_fmt, sci_threshold: float = 0.0001, has_named_index: bool = None,
    color_fmts: dict = None, column_rules: dict = None, adf_mode: bool = False,
):
    """
    Structured writing engine for Excel dashboards.

    Writes cell by cell to:
    - Apply scientific notation to small values (< sci_threshold)
    - Apply red/green colors based on econometric rules
    - Prevent overwriting other tables in the dashboard
    """
    # Sanitize column names
    df = sanitize_column_names(df)
    df = clean_index_names(df) 

    # Detect index
    if has_named_index is None:
        has_named_index = bool(df.index.name)
    data_col_offset = 1 if has_named_index else 0

    sheet.write(start_row - 1, start_col, title, title_fmt)

    # Column headers
    if has_named_index:
        sheet.write(start_row, start_col, df.index.name or "", header_fmt)
    for col_num, col_name in enumerate(df.columns):
        sheet.write(start_row, start_col + data_col_offset + col_num, col_name, header_fmt)

    # Auto-detect active rules
    if column_rules is not None:
        active_rules = column_rules
    elif adf_mode:
        active_rules = ADF_RULES
    elif 'VIF' in df.columns:
        active_rules = VIF_RULES
    elif 'P>|t|' in df.columns:
        active_rules = OLS_COEF_RULES
    else:
        active_rules = OLS_METRICS_RULES

    # Numeric types: native Python + numpy scalars
    _num_types = (int, float, np.floating, np.integer)  #covers all numpy scalar types

    # Data rows
    for row_num, (idx_value, row) in enumerate(df.iterrows()):
        current_row = start_row + 1 + row_num

        if has_named_index:
            sheet.write(current_row, start_col, idx_value, index_fmt)

        # ADF: extract critical value from this row to compare ADF_Statistic
        adf_critical = None
        if adf_mode and 'Critical_Value_5%' in df.columns:
            adf_critical = row.get('Critical_Value_5%', None)

        for col_num, col_name in enumerate(df.columns):
            cell_value = row[col_name]
            current_col = start_col + data_col_offset + col_num

            # NaN / Inf guard (critical order: must run before write_number)
            if isinstance(cell_value, _num_types) and pd.isna(cell_value):
                sheet.write_blank(current_row, current_col, None, num_fmt)
                continue
            if isinstance(cell_value, _num_types) and not np.isfinite(cell_value):
                sheet.write_blank(current_row, current_col, None, num_fmt)
                continue

            # Strikeout: Impact_Pct crossed out if P>|t| > 0.05
            if col_name == 'Impact_Pct' and 'P>|t|' in df.columns and color_fmts:
                pval = row.get('P>|t|', None)
                if (pval is not None
                        and isinstance(pval, _num_types)
                        and np.isfinite(float(pval))
                        and float(pval) > 0.05
                        and isinstance(cell_value, _num_types)):
                    use_sci = (cell_value != 0) and (abs(cell_value) < sci_threshold)
                    fmt = color_fmts['strikeout_sci'] if use_sci else color_fmts['strikeout']
                    sheet.write_number(current_row, current_col, float(cell_value), fmt)
                    continue

            # Determine color
            color_key = None
            if color_fmts and isinstance(cell_value, _num_types):
                if adf_mode and col_name == 'ADF_Statistic' and adf_critical is not None:
                    color_key = _rule_adf_statistic(cell_value, adf_critical)
                elif col_name in ADF_RULES and adf_mode:
                    color_key = ADF_RULES[col_name](cell_value)
                elif col_name in active_rules:
                    color_key = active_rules[col_name](cell_value)

            # Write cell
            if isinstance(cell_value, bool):
                sheet.write(current_row, current_col, cell_value, index_fmt)
            elif isinstance(cell_value, _num_types):
                fmt = resolve_fmt(color_key, float(cell_value), sci_threshold,
                                  color_fmts or {}, num_fmt, sci_fmt)
                sheet.write_number(current_row, current_col, float(cell_value), fmt)
            else:
                sheet.write(current_row, current_col, str(cell_value), index_fmt)

_INDEX_RENAME = {
    'const':          'Intercept',
    'Internal_Index': 'Internal Index',
    'External_Index': 'External Index',
    'Panic_Index':    'Panic Index',
    }

def clean_index_names(df: pd.DataFrame) -> pd.DataFrame:
    """Renames DataFrame index values to cleaner display names."""
    return df.rename(index=_INDEX_RENAME)