import csv
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import requests

def download_csv_from_url(url: str, save_path: str) -> None:
    """
    Download a CSV file from the given URL and save it to disk.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded CSV to {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading CSV: {e}")
        
if __name__ == "__main__":
    # 1. Download the latest CSV from GitHub
    csv_url = "https://raw.githubusercontent.com/Tony-Long-414/final-report/refs/heads/main/DCWP_Charges_20250618.csv"
    save_location = "DCWP_Charges_20250618.csv"
    download_csv_from_url(csv_url, save_location)

    # 2. Now load and process
    df = load_and_clean(save_location)
    biz = prepare_business_dataset(df)
    train_and_eval_logistic(biz)
    train_and_eval_rf(biz)
    print("\n6-Month Forecast:")
    print(forecast_monthly(df, 6))

def load_and_clean(path: str) -> pd.DataFrame:
    """
    Load the DCWP charges CSV, normalize columns, parse dates,
    drop invalid rows, and fill basic missing values.
    Uses the Python engine & QUOTE_NONE to avoid EOF-inside-string errors.
    """
    read_args = dict(
        encoding='utf-8',
        engine='python',
        quoting=csv.QUOTE_NONE,       # do not interpret quotes
        on_bad_lines='skip',          # skip any malformed line
        sep=','                       # explicit separator
    )
    try:
        df = pd.read_csv(path, **read_args)
    except UnicodeDecodeError:
        # fallback to latin1 if needed
        read_args['encoding'] = 'latin1'
        df = pd.read_csv(path, **read_args)

    # normalize column names
    df.columns = (df.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(' ', '_')
                    .str.replace('/', '_'))

    # parse violation_date
    df['violation_date'] = pd.to_datetime(df['violation_date'], errors='coerce')
    df = df.dropna(subset=['violation_date'])

    # fill yes/no fields
    for col in ['cure_eligible', 'cured']:
        if col in df.columns:
            df[col] = df[col].fillna('No')

    # fill outcome counts
    for col in ['guilty', 'not_guilty', 'dismissed']:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    return df

def prepare_business_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per business, build:
      - total_violations
      - main_category (most frequent category)
      - past_violations & repeat_offender
    """
    # ensure the cleaned column name is used
    df['business_category'] = df['business_category'].fillna('Unknown')

    # group by the normalized ID
    grp = df.groupby('business_unique_id')

    # safe modal function
    def get_main_category(x):
        vc = x.value_counts()
        return vc.idxmax() if not vc.empty else 'Unknown'

    # build business‐level table
    bd = pd.DataFrame({
        'business_id':       grp['business_unique_id'].first(),
        'total_violations':  grp.size(),
        'main_category':     grp['business_category'].agg(get_main_category)
    }).reset_index(drop=True)

    # create target and a past‐count feature if you still want it
    bd['past_violations'] = (bd['total_violations'] - 1).clip(lower=0)
    bd['repeat_offender'] = (bd['total_violations'] > 1).astype(int)

    return bd




def train_and_eval_logistic(biz_df: pd.DataFrame):
    """
    Train & evaluate a logistic regression using only the business's main_category.
    Prints classification report, ROC AUC, and top 5 coefficients.
    """
    # Prepare features & target
    X_lr = pd.get_dummies(biz_df[['main_category']], drop_first=True)
    y_lr = biz_df['repeat_offender']

    # Stratified train/test split
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
        X_lr, y_lr,
        test_size=0.2,
        stratify=y_lr,
        random_state=42
    )

    # Train logistic regression
    log_model = LogisticRegression(
        solver='liblinear',
        class_weight='balanced',
        max_iter=500
    )
    log_model.fit(X_train_lr, y_train_lr)

    # Predict & evaluate
    y_pred_lr = log_model.predict(X_test_lr)
    print("=== Logistic Regression Classification Report ===")
    print(classification_report(y_test_lr, y_pred_lr))

    auc_lr = roc_auc_score(y_test_lr, log_model.predict_proba(X_test_lr)[:, 1])
    print(f"Logistic ROC AUC: {auc_lr:.3f}")

    # Inspect top coefficients
    coef = pd.Series(log_model.coef_[0], index=X_lr.columns).abs().sort_values(ascending=False)
    print("\n=== Top 5 Coefficients ===")
    print(coef.head(5))

def train_and_eval_rf(biz_df: pd.DataFrame):
    """
    Train & evaluate a Random Forest classifier using only the business's main_category.
    Prints classification report, ROC AUC, and top 5 feature importances.
    """
    # Prepare features & target
    X = pd.get_dummies(biz_df[['main_category']], drop_first=True)
    y = biz_df['repeat_offender']

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Initialize & train the Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = rf.predict(X_test)
    print("=== Random Forest Classification Report ===")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f"Random Forest ROC AUC: {auc:.3f}")

    # Feature importances
    imps = pd.Series(rf.feature_importances_, index=X.columns)
    print("\n=== Top 5 Features ===")
    print(imps.sort_values(ascending=False).head(5))

def forecast_monthly(df: pd.DataFrame, steps: int = 6) -> pd.Series:
    """
    Resample df by month on the appropriate date column, fit a
    Seasonal ARIMA (1,1,1)x(1,1,1,12), and return the next `steps` months.
    Automatically finds a column containing 'date' if 'violation_date' is missing.
    """
    # 1) Determine which column to use as our date index
    if 'violation_date' in df.columns:
        date_col = 'violation_date'
    else:
        # try to find any date-like column
        candidates = [c for c in df.columns if 'date' in c]
        if not candidates:
            raise KeyError("No date column found in DataFrame (looked for 'violation_date' or any 'date' column).")
        date_col = candidates[0]
        print(f"⚠️ Warning: using '{date_col}' as the date column.")

    # 2) Ensure it's datetime and drop missing
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])

    # 3) Build monthly count series
    series = df.set_index(date_col).resample('M').size()

    # 4) Fit SARIMA
    model = sm.tsa.SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fit = model.fit(disp=False)

    # 5) Forecast
    fc = fit.forecast(steps=steps)

    # 6) Reindex with proper month ends
    dates = pd.date_range(
        start=series.index[-1] + pd.offsets.MonthEnd(1),
        periods=steps,
        freq='M'
    )
    fc.index = dates

    # 7) Print & return
    print("=== 6-Month Forecast of Violations ===")
    print(fc.round())
    return fc

if __name__ == "__main__":
    df = load_and_clean("DCWP_Charges_20250618.csv")
    biz = prepare_business_dataset(df)
    train_and_eval_logistic(biz)
    train_and_eval_rf(biz)
    print("\n6-Month Forecast:")
    print(forecast_monthly(df,6))
