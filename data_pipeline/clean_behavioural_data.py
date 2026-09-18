import numpy as np
import pandas as pd

RAW_PATH = "data/behavioural_dataset_raw.csv"
CLEAN_PATH = "data/behavioural_dataset_clean.csv"

#step 1:Loading and Inspecting
def load_raw_data(path:str)->pd.DataFrame:
    df=pd.read_csv(path)
    print(f"Load {len(df)} rows, {len(df.columns)} cloumns from {path}")
    print()
    print("Missing Values per column:")
    print(df.isna().sum().to_string())
    print()
    print(f"Fully duplicated rows:{df.duplicated().sum()}")
    print(f"Duplicated user_ids:{df['user_id'].duplicated().sum()}")

    return df


#sep 2: Removing duplicates
def remove_duplicates(df: pd.DataFrame)->pd.DataFrame:
    n_before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    n_removed=n_before - len(df)

    print(f"Removed {n_removed} duplicate rows ({n_before} -> {len(df)})")
    return df

#Step 3: Verifying the time_to_report_sec missingness pattern
def verify_missingness_pattern(df:pd.DataFrame)->None:
    is_missing = df["time_to_report_sec"].isna()
    did_click = df["clicked_link"] == 1
 
    matches_exactly = (is_missing == did_click).all()
 
    print(f"NaN count in time_to_report_sec: {is_missing.sum()}")
    print(f"Users who clicked: {did_click.sum()}")
    print(f"Every NaN matches a clicker exactly: {matches_exactly}")
 
    if matches_exactly:
        print("-> All missingness here is structural. No imputation needed;")
        print("   NaN correctly means 'not applicable' for this feature.")
    else:
        mismatch_count = (is_missing != did_click).sum()
        print(f"-> {mismatch_count} rows don't match the expected pattern.")
        print("   Some missingness here is NOT purely structural - would need")
        print("   separate handling for the non-structural portion.")
#step 4: Impute missing values in the remaining feature columns
def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    median_impute_cols = [
        "time_to_first_click_sec", "hover_time_sec", "mean_mouse_speed_px_s",
        "slow_movement_ratio", "hover_count", "reopen_count",
    ]
    for col in median_impute_cols:
        n_missing = df[col].isna().sum()
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"{col}: imputed {n_missing} values with median ({median_val:.2f})")
 
    mode_val = df["sender_checked"].mode()[0]
    n_missing = df["sender_checked"].isna().sum()
    df["sender_checked"] = df["sender_checked"].fillna(mode_val)
    print(f"sender_checked: imputed {n_missing} values with mode ({mode_val})")
 
    return df

# Step 5: Detect and cap outliers (log-space IQR, for skewed data)
def cap_outliers(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
 
    for col in cols:
        log_vals = np.log1p(df[col])
 
        q1 = log_vals.quantile(0.25)
        q3 = log_vals.quantile(0.75)
        iqr = q3 - q1
        lower_log = q1 - 1.5 * iqr
        upper_log = q3 + 1.5 * iqr
 
        lower_bound = max(np.expm1(lower_log), 0)  # never allow a negative bound
        upper_bound = np.expm1(upper_log)
 
        n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
 
        print(f"{col}: capped {n_outliers} outliers to range [{lower_bound:.1f}, {upper_bound:.1f}]")
 
    return df

# Step 6: Final validation and save
def validate_and_save(df: pd.DataFrame, path: str) -> None:
    problems = []
 
    # Every column except time_to_report_sec should now be fully populated
    for col in df.columns:
        if col == "time_to_report_sec":
            continue
        if df[col].isna().any():
            problems.append(f"{col} still has {df[col].isna().sum()} missing values")
 
    # Counts can't be negative
    for col in ["hover_count", "reopen_count"]:
        if (df[col] < 0).any():
            problems.append(f"{col} has negative values")
 
    # Binary columns must only contain 0 or 1
    for col in ["sender_checked", "clicked_link"]:
        if not df[col].isin([0, 1]).all():
            problems.append(f"{col} has values outside {{0, 1}}")
 
    if problems:
        print("VALIDATION FAILED:")
        for p in problems:
            print(f"  - {p}")
        raise ValueError("Cleaned dataset failed validation - see issues above")
 
    print("Validation passed - no unexpected missing values or impossible entries.")
    print()
    print(f"Final cleaned dataset: {len(df)} rows, {len(df.columns)} columns")
 
    df.to_csv(path, index=False)
    print(f"Saved to {path}")


if __name__ == "__main__":
    raw_df = load_raw_data(RAW_PATH)
    print()
    df=remove_duplicates(raw_df)
    print()
    verify_missingness_pattern(df)
    print()
    df = impute_missing_values(df)
    print()
    print("Remaining missing values per column:")
    print(df.isna().sum().to_string())
    outlier_cols = ["time_to_first_click_sec", "hover_time_sec", "mean_mouse_speed_px_s"]
    df = cap_outliers(df, outlier_cols)
    print()
 
    validate_and_save(df, CLEAN_PATH)