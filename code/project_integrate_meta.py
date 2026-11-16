# ============================================================
# Project INTEGRATE — IPD Meta-Analysis in Python
# Replicates: Mun et al. (2022), Prevention Science
#
# Designed for JupyterLite / Jupyter Notebook
# Dataset: dat_all_PS.csv (stored in GitHub repo)
# ============================================================

# ── 0. Install / import libraries ───────────────────────────
# In JupyterLite, run this cell first:
#   import micropip
#   await micropip.install(['pandas','numpy','scipy','statsmodels'])

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SECTION 1 — Load the data
# ============================================================

DATA_URL = "https://raw.githubusercontent.com/drknortheim/Python_for_Addiction_Researchers/main/data/dat_all_PS.csv"

# Smart loader — works in both JupyterLite and regular Python
import sys, io

async def load_data_jupyterlite(url):
    from pyodide.http import pyfetch
    response = await pyfetch(url)
    content = await response.string()
    return pd.read_csv(io.StringIO(content))

def load_data_standard(url):
    return pd.read_csv(url)

# Detect environment and load accordingly
if "pyodide" in sys.modules or hasattr(__builtins__, '__IPYTHON__'):
    # Running in JupyterLite — use pyodide fetcher
    # NOTE: In JupyterLite, run this line in its own cell:
    #   df = await load_data_jupyterlite(DATA_URL)
    print("JupyterLite detected.")
    print("Run this in the next cell:")
    print("  df = await load_data_jupyterlite(DATA_URL)")
else:
    # Running in standard Python / Jupyter Notebook
    df = load_data_standard(DATA_URL)

print("Dataset loaded successfully")
print(f"Shape: {df.shape}")          # (rows, columns)
print(f"Columns: {list(df.columns)}")
print()
print(df.head())


# ============================================================
# SECTION 2 — Explore the data
# ============================================================

print("\n── Time points ──────────────────────────────────────")
print(df['time'].value_counts().sort_index())

print("\n── Studies ──────────────────────────────────────────")
print(df['study'].unique())

print("\n── Outcome variable (y = drinks/week) ───────────────")
print(df['y'].describe())
zero_pct = (df['y'] == 0).mean() * 100
print(f"Zero drinkers: {zero_pct:.1f}%")


# ============================================================
# SECTION 3 — Split data by follow-up time point
# ============================================================

df_1mo  = df[df['time'] == 1].copy()   # 1–3 months
df_6mo  = df[df['time'] == 6].copy()   # 6 months
df_12mo = df[df['time'] == 12].copy()  # 9–12 months

print(f"\n1–3 months:  {len(df_1mo):,} participants")
print(f"6 months:    {len(df_6mo):,} participants")
print(f"9–12 months: {len(df_12mo):,} participants")


# ============================================================
# SECTION 4 — Helper: check zero inflation
# ============================================================

def is_zero_inflated(y_series, threshold=0.10):
    """
    Returns True if the observed zero rate exceeds the Poisson
    expected zero rate by more than the threshold.
    Used to decide between Negative Binomial and MZIP models.
    """
    y = y_series.dropna()
    mean_y = y.mean()
    zero_rate_observed    = (y == 0).mean()
    zero_rate_theoretical = np.exp(-mean_y)   # Poisson expected zeros
    return zero_rate_observed > zero_rate_theoretical + threshold


# ============================================================
# SECTION 5 — Helper: fit model for one study
# ============================================================

INTERVENTION_VARS = ['int_GMI', 'int_MI_PF', 'int_PF']
COVARIATE_VARS    = ['m0', 'gender', 'year_new', 'race_new']
PROBLEM_STUDIES   = ['2', '7.1']   # gender caused estimation issues in R code


def fit_study(study_df, study_id):
    """
    Fits a Negative Binomial regression for one study.
    Returns a DataFrame of intervention coefficient estimates.

    Note: Python's statsmodels does not have a built-in MZIP
    (Marginalized Zero-Inflated Poisson) model matching the R
    'mcount' package. We use Negative Binomial for all studies
    here, which is the same model used when zero inflation is
    below the threshold in the original R code.
    """

    # Build variable list — drop gender for problem studies
    covariates = COVARIATE_VARS.copy()
    if study_id in PROBLEM_STUDIES:
        covariates = [v for v in covariates if v != 'gender']

    # Keep only intervention vars present in this study (not all NA)
    int_vars_present = [
        v for v in INTERVENTION_VARS
        if v in study_df.columns and study_df[v].notna().any()
    ]

    all_vars = int_vars_present + covariates

    # Drop columns that are entirely missing
    all_vars = [v for v in all_vars if study_df[v].notna().any()]

    # Drop rows with any missing values
    model_df = study_df[all_vars + ['y']].dropna()

    if len(model_df) < 20:
        print(f"  Study {study_id}: skipped (too few rows after dropna)")
        return None

    # Check zero inflation (informational)
    inflated = is_zero_inflated(model_df['y'])
    model_type = "MZIP-candidate (using NB)" if inflated else "NB"
    print(f"  Study {study_id}: {model_type}, n={len(model_df)}")

    # Build formula
    formula = "y ~ " + " + ".join(all_vars)

    try:
        model  = smf.negativebinomial(formula, data=model_df).fit(disp=False)
        result = model.summary2().tables[1]

        # Keep only intervention coefficients
        int_rows = result[result.index.str.startswith('int_')]
        return int_rows[['Coef.', 'Std.Err.']].rename(
            columns={'Coef.': 'estimate', 'Std.Err.': 'se'}
        )
    except Exception as e:
        print(f"  Study {study_id}: model failed — {e}")
        return None


# ============================================================
# SECTION 6 — Helper: run meta-analysis (random effects)
# ============================================================

def random_effects_meta(estimates, ses):
    """
    Random-effects meta-analysis using the DerSimonian-Laird method.
    Mirrors the rma() call in the R code.

    Returns: pooled estimate, 95% CI, p-value, I² heterogeneity
    """
    estimates = np.array(estimates)
    ses       = np.array(ses)
    variances = ses ** 2
    weights   = 1 / variances

    # Fixed-effect pooled estimate
    fe_estimate = np.sum(weights * estimates) / np.sum(weights)

    # Cochran's Q and tau² (between-study variance)
    Q  = np.sum(weights * (estimates - fe_estimate) ** 2)
    k  = len(estimates)
    df = k - 1
    c  = np.sum(weights) - np.sum(weights ** 2) / np.sum(weights)
    tau2 = max(0, (Q - df) / c)   # DerSimonian-Laird

    # Random-effects weights and pooled estimate
    re_weights  = 1 / (variances + tau2)
    re_estimate = np.sum(re_weights * estimates) / np.sum(re_weights)
    re_se       = np.sqrt(1 / np.sum(re_weights))

    # 95% CI and p-value
    ci_lower = re_estimate - 1.96 * re_se
    ci_upper = re_estimate + 1.96 * re_se
    z        = re_estimate / re_se
    p_value  = 2 * (1 - stats.norm.cdf(abs(z)))

    # I² — proportion of variance due to heterogeneity
    i2 = max(0, (Q - df) / Q * 100) if Q > 0 else 0

    return {
        'estimate': re_estimate,
        'se':       re_se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'z':        z,
        'p_value':  p_value,
        'tau2':     tau2,
        'Q':        Q,
        'I2':       i2,
        'k':        k
    }


# ============================================================
# SECTION 7 — Run the full meta-analysis for one time point
# ============================================================

def run_meta_analysis(subset_df, label):
    """
    Step 1: fit model in each study.
    Step 2: pool results with random-effects meta-analysis.
    Prints a summary table of results.
    """
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    studies = subset_df['study'].unique()
    all_results = []

    for study_id in sorted(studies):
        study_data = subset_df[subset_df['study'] == study_id].copy()
        result = fit_study(study_data, study_id)
        if result is not None:
            result = result.copy()
            result['study'] = study_id
            all_results.append(result)

    if not all_results:
        print("No results to pool.")
        return None

    combined = pd.concat(all_results)

    print(f"\n── Pooled Meta-Analysis Results ─────────────────────")
    print(f"{'Intervention':<15} {'Est':>7} {'SE':>7} {'95% CI':>18} {'p':>8} {'I²':>7} {'k':>4}")
    print("-" * 65)

    meta_summary = {}
    for int_var in INTERVENTION_VARS:
        int_data = combined[combined.index.str.startswith(int_var)]
        if len(int_data) < 2:
            continue

        meta = random_effects_meta(
            int_data['estimate'].values,
            int_data['se'].values
        )
        meta_summary[int_var] = meta

        ci_str = f"[{meta['ci_lower']:+.3f}, {meta['ci_upper']:+.3f}]"
        print(
            f"{int_var:<15} "
            f"{meta['estimate']:>+7.3f} "
            f"{meta['se']:>7.3f} "
            f"{ci_str:>18} "
            f"{meta['p_value']:>8.4f} "
            f"{meta['I2']:>6.1f}% "
            f"{meta['k']:>4}"
        )

    return meta_summary


# ============================================================
# SECTION 8 — Run all three time points
# ============================================================

results_1mo  = run_meta_analysis(df_1mo,  "1–3 Months Follow-Up")
results_6mo  = run_meta_analysis(df_6mo,  "6 Months Follow-Up")
results_12mo = run_meta_analysis(df_12mo, "9–12 Months Follow-Up")


# ============================================================
# SECTION 9 — Export results to CSV (for GitHub / reporting)
# ============================================================

rows = []
for label, results in [('1-3mo', results_1mo),
                        ('6mo',   results_6mo),
                        ('12mo',  results_12mo)]:
    if results is None:
        continue
    for int_var, meta in results.items():
        rows.append({
            'timepoint':    label,
            'intervention': int_var,
            'estimate':     round(meta['estimate'], 4),
            'se':           round(meta['se'], 4),
            'ci_lower':     round(meta['ci_lower'], 4),
            'ci_upper':     round(meta['ci_upper'], 4),
            'p_value':      round(meta['p_value'], 4),
            'I2':           round(meta['I2'], 1),
            'k_studies':    meta['k']
        })

results_df = pd.DataFrame(rows)
results_df.to_csv('meta_analysis_results.csv', index=False)
print("\nResults saved to meta_analysis_results.csv")
print(results_df.to_string(index=False))
