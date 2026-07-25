import nbformat as nbf
import os

cells_content = [
"""# Cell 1: Import Libraries
!pip install -q optuna xgboost lightgbm catboost prophet statsmodels plotly scikit-learn nbformat torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, median_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import time
import os
import joblib
import json
import warnings
warnings.filterwarnings('ignore')
""",
"""# Cell 2: Load Dataset
file_path = 'dataset1_nonghan_water_quality.csv'
if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
else:
    df_raw = pd.read_csv(file_path)
    print(f"Dataset loaded successfully with shape: {df_raw.shape}")
    display(df_raw.head())
""",
"""# Cell 3: Data Cleaning
if 'df_raw' in locals():
    df = df_raw.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    df = df.drop_duplicates()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_monthly = df.groupby(pd.Grouper(key='date', freq='ME'))[numeric_cols].mean().reset_index()
    df_monthly = df_monthly.ffill().bfill()
    Q1 = df_monthly['WQI_al_score'].quantile(0.25)
    Q3 = df_monthly['WQI_al_score'].quantile(0.75)
    IQR = Q3 - Q1
    df_monthly['WQI_al_score'] = np.clip(df_monthly['WQI_al_score'], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    print(f"Data Cleaning Complete. Shape: {df_monthly.shape}")
    display(df_monthly.head())
""",
"""# Cell 4: EDA
if 'df_monthly' in locals():
    fig = px.line(df_monthly, x='date', y='WQI_al_score', title='Water Quality Index (WQI) Over Time', markers=True)
    fig.update_layout(template='plotly_white')
    fig.show()
    cols_to_plot = ['DO_mg_L', 'BOD_mg_L', 'temp_c', 'pH']
    cols = [c for c in cols_to_plot if c in df_monthly.columns]
    if len(cols) > 0:
        fig2 = make_subplots(rows=len(cols), cols=1, shared_xaxes=True, subplot_titles=cols)
        for i, col in enumerate(cols, 1):
            fig2.add_trace(go.Scatter(x=df_monthly['date'], y=df_monthly[col], mode='lines', name=col), row=i, col=1)
        fig2.update_layout(height=800, title_text="Key Environmental Features Over Time", template='plotly_white')
        fig2.show()
""",
"""# Cell 5: Feature Engineering
if 'df_monthly' in locals():
    df_feat = df_monthly.copy()
    for lag in [1, 2, 3, 6, 12]:
        df_feat[f'lag_{lag}'] = df_feat['WQI_al_score'].shift(lag)
    for w in [3, 6, 12]:
        df_feat[f'rolling_mean_{w}'] = df_feat['WQI_al_score'].rolling(window=w).mean()
    for w in [3, 6]:
        df_feat[f'rolling_std_{w}'] = df_feat['WQI_al_score'].rolling(window=w).std()
    df_feat['month'] = df_feat['date'].dt.month
    df_feat['quarter'] = df_feat['date'].dt.quarter
    df_feat['year'] = df_feat['date'].dt.year
    def get_season(m):
        if m in [2,3,4,5]: return 1
        elif m in [6,7,8,9,10]: return 2
        else: return 3
    df_feat['season'] = df_feat['month'].apply(get_season)
    df_feat['summer'] = (df_feat['season'] == 1).astype(int)
    df_feat['rainy'] = (df_feat['season'] == 2).astype(int)
    df_feat['winter'] = (df_feat['season'] == 3).astype(int)
    # 🚨 กำจัด Data Leakage: ตัดคอลัมน์เคมีที่เป็นส่วนผสมสูตร WQI ทิ้ง!
    leakage_cols = ['DO_mg_L', 'BOD_mg_L', 'pH', 'temp_c', 'EC_uS_cm', 'COD_mg_L', 'NH3_mg_L', 'NO3_mg_L', 'TP_mg_L', 'TCB_MPN_100mL', 'FCB_MPN_100mL']
    cols_to_drop = [c for c in leakage_cols if c in df_feat.columns]
    df_feat = df_feat.drop(columns=cols_to_drop)
    
    df_feat = df_feat.dropna().reset_index(drop=True)
    print(f"Feature Engineering Complete (Anti-Leakage Applied). Shape: {df_feat.shape}")
""",
"""# Cell 6: Walk-Forward Validation Setup
if 'df_feat' in locals():
    dates = df_feat['date']
    y = df_feat['WQI_al_score']
    X = df_feat.drop(columns=['date', 'WQI_al_score'])
    
    # Define Walk-Forward Cross Validator (3 Folds)
    tscv = TimeSeriesSplit(n_splits=3)
    
    class DataWrapper:
        def __init__(self):
            self.X, self.y = X, y
            self.y_stat = y.values
            self.df_prophet = pd.DataFrame({'ds': dates, 'y': y})
            self.scaler = MinMaxScaler()
            self.scaler.fit(X)
            self.X_dl = torch.tensor(self.scaler.transform(X), dtype=torch.float32).unsqueeze(1)
            self.y_dl = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    data = DataWrapper()
    print(f"Total samples: {len(X)}")
    print(f"Walk-Forward Folds: {tscv.n_splits}")
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"Fold {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
""",
"""# Cell 7: Train Every Model (Walk-Forward)
models_to_run = {'SARIMA': True, 'ETS': True, 'Prophet': True, 'XGBoost': True, 'RandomForest': True, 'LSTM': True, 'LightGBM': True, 'CatBoost': True, 'GRU': True}
print("Training Models using Walk-Forward Validation...")

def smape(A, F): return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F) + 1e-8))

oof_metrics = {m: {'RMSE': [], 'MAE': [], 'MAPE': [], 'SMAPE': [], 'R2': []} for m in models_to_run}
train_times = {m: [] for m in models_to_run}

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, model_type):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) if model_type=='LSTM' else nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

for fold, (train_idx, test_idx) in enumerate(tscv.split(data.X)):
    print(f"Processing Fold {fold+1}...")
    X_t, y_t = data.X.iloc[train_idx], data.y.iloc[train_idx]
    X_v, y_v = data.X.iloc[test_idx], data.y.iloc[test_idx]
    y_t_stat, y_v_stat = data.y_stat[train_idx], data.y_stat[test_idx]
    
    def record_metrics(m_name, preds, t_time):
        train_times[m_name].append(t_time)
        oof_metrics[m_name]['RMSE'].append(np.sqrt(mean_squared_error(y_v_stat, preds)))
        oof_metrics[m_name]['MAE'].append(mean_absolute_error(y_v_stat, preds))
        oof_metrics[m_name]['MAPE'].append(np.mean(np.abs((y_v_stat - preds) / y_v_stat)) * 100)
        oof_metrics[m_name]['SMAPE'].append(smape(y_v_stat, preds))
        oof_metrics[m_name]['R2'].append(r2_score(y_v_stat, preds))

    if models_to_run['SARIMA']:
        t0 = time.time()
        try:
            m = SARIMAX(y_t_stat, order=(1,1,1)).fit(disp=False)
            record_metrics('SARIMA', m.forecast(steps=len(y_v_stat)), time.time()-t0)
        except: pass

    if models_to_run['ETS']:
        t0 = time.time()
        try:
            m = ExponentialSmoothing(y_t_stat, trend='add').fit()
            record_metrics('ETS', m.forecast(steps=len(y_v_stat)), time.time()-t0)
        except: pass

    if models_to_run['Prophet']:
        t0 = time.time()
        try:
            m = Prophet(yearly_seasonality=True).fit(data.df_prophet.iloc[train_idx])
            fut = pd.DataFrame({'ds': data.df_prophet['ds'].iloc[test_idx]})
            record_metrics('Prophet', m.predict(fut)['yhat'].values, time.time()-t0)
        except: pass

    if models_to_run['XGBoost']:
        t0 = time.time()
        m = xgb.XGBRegressor(n_estimators=100, random_state=42).fit(X_t, y_t)
        record_metrics('XGBoost', m.predict(X_v), time.time()-t0)

    if models_to_run['RandomForest']:
        t0 = time.time()
        m = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_t, y_t)
        record_metrics('RandomForest', m.predict(X_v), time.time()-t0)

    if models_to_run['LightGBM']:
        t0 = time.time()
        m = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1).fit(X_t, y_t)
        record_metrics('LightGBM', m.predict(X_v), time.time()-t0)

    if models_to_run['CatBoost']:
        t0 = time.time()
        m = cb.CatBoostRegressor(n_estimators=100, random_state=42, verbose=False).fit(X_t, y_t)
        record_metrics('CatBoost', m.predict(X_v), time.time()-t0)

    def train_dl_fold(mtype):
        m = SimpleRNN(data.X_dl.shape[2], 32, 1, mtype)
        opt = optim.Adam(m.parameters(), lr=0.01)
        t0 = time.time()
        for _ in range(100):
            opt.zero_grad()
            nn.MSELoss()(m(data.X_dl[train_idx]), data.y_dl[train_idx]).backward()
            opt.step()
        m.eval()
        with torch.no_grad(): return m(data.X_dl[test_idx]).numpy().flatten(), time.time()-t0

    if models_to_run['LSTM']:
        preds, tt = train_dl_fold('LSTM')
        record_metrics('LSTM', preds, tt)
        
    if models_to_run['GRU']:
        preds, tt = train_dl_fold('GRU')
        record_metrics('GRU', preds, tt)

print("✅ Walk-Forward Evaluation Complete.")
""",
"""# Cell 8: Compare Models and Select Best Model
eval_results = []
for m in models_to_run:
    if len(oof_metrics[m]['RMSE']) > 0:
        eval_results.append({
            'Model': m,
            'Mean MAE': np.mean(oof_metrics[m]['MAE']),
            'Mean RMSE': np.mean(oof_metrics[m]['RMSE']),
            'Mean MAPE': np.mean(oof_metrics[m]['MAPE']),
            'Mean SMAPE': np.mean(oof_metrics[m]['SMAPE']),
            'Mean R²': np.mean(oof_metrics[m]['R2']),
            'Mean Train Time (s)': np.mean(train_times[m])
        })

comparison_df = pd.DataFrame(eval_results).sort_values('Mean RMSE').reset_index(drop=True)
best_model_name = comparison_df.iloc[0]['Model']

print(f"🏆 Best Model based on Walk-Forward Evaluation: {best_model_name}")
display(comparison_df)
""",
"""# Cell 9: Hyperparameter Tuning
print(f"--- 9. Hyperparameter Tuning for {best_model_name} ---")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def evaluate_cv(model_fn, params):
    rmses, maes, mapes, r2s = [], [], [], []
    for train_idx, val_idx in tscv.split(data.X):
        X_t, y_t = data.X.iloc[train_idx], data.y.iloc[train_idx]
        X_v, y_v = data.X.iloc[val_idx], data.y.iloc[val_idx]
        preds = model_fn(params, X_t, y_t, X_v, train_idx, val_idx)
        if preds is not None:
            rmses.append(np.sqrt(mean_squared_error(y_v, preds)))
            maes.append(mean_absolute_error(y_v, preds))
            mapes.append(np.mean(np.abs((y_v.values - preds) / y_v.values)) * 100)
            r2s.append(r2_score(y_v, preds))
    if not rmses: return 9999, {}
    return np.mean(rmses), {'MAE': {'Mean': np.mean(maes)}, 'RMSE': {'Mean': np.mean(rmses)}, 'MAPE': {'Mean': np.mean(mapes)}, 'R2': {'Mean': np.mean(r2s)}}

def objective(trial):
    try:
        if best_model_name == 'XGBoost':
            p = {'n_estimators': trial.suggest_int('n_estimators', 50, 300), 'max_depth': trial.suggest_int('max_depth', 2, 8), 'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True), 'subsample': trial.suggest_float('subsample', 0.5, 1.0), 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)}
            fn = lambda p,Xt,yt,Xv,ti,vi: xgb.XGBRegressor(**p, random_state=42, objective='reg:squarederror').fit(Xt, yt).predict(Xv)
        elif best_model_name == 'RandomForest':
            p = {'n_estimators': trial.suggest_int('n_estimators', 50, 300), 'max_depth': trial.suggest_int('max_depth', 2, 10), 'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)}
            fn = lambda p,Xt,yt,Xv,ti,vi: RandomForestRegressor(**p, random_state=42).fit(Xt, yt).predict(Xv)
        elif best_model_name == 'LightGBM':
            p = {'num_leaves': trial.suggest_int('num_leaves', 10, 100), 'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True), 'max_depth': trial.suggest_int('max_depth', -1, 10)}
            fn = lambda p,Xt,yt,Xv,ti,vi: lgb.LGBMRegressor(**p, random_state=42, verbose=-1).fit(Xt, yt).predict(Xv)
        elif best_model_name == 'CatBoost':
            p = {'depth': trial.suggest_int('depth', 2, 8), 'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True), 'iterations': trial.suggest_int('iterations', 50, 300)}
            fn = lambda p,Xt,yt,Xv,ti,vi: cb.CatBoostRegressor(**p, random_state=42, verbose=False).fit(Xt, yt).predict(Xv)
        elif best_model_name == 'Prophet':
            p = {'cps': trial.suggest_float('cps', 0.001, 0.5, log=True), 'sps': trial.suggest_float('sps', 0.01, 10.0, log=True)}
            def fn(p,Xt,yt,Xv,ti,vi):
                m = Prophet(yearly_seasonality=True, changepoint_prior_scale=p['cps'], seasonality_prior_scale=p['sps'])
                return m.fit(pd.DataFrame({'ds': data.df_prophet['ds'].iloc[ti], 'y': yt.values})).predict(pd.DataFrame({'ds': data.df_prophet['ds'].iloc[vi]}))['yhat'].values
        elif best_model_name == 'SARIMA':
            p = {'order': (trial.suggest_int('p',0,2), trial.suggest_int('d',0,1), trial.suggest_int('q',0,2)), 's_order': (1,1,1,12) if len(yt)>=24 else (0,0,0,0)}
            fn = lambda p,Xt,yt,Xv,ti,vi: SARIMAX(yt.values, order=p['order'], seasonal_order=p['s_order'], enforce_stationarity=False, enforce_invertibility=False).fit(disp=False).forecast(len(Xv))
        elif best_model_name == 'ETS':
            p = {'trend': trial.suggest_categorical('trend', ['add', None]), 'seasonal': 'add' if len(yt)>=24 else None, 'sp': 12 if len(yt)>=24 else None}
            fn = lambda p,Xt,yt,Xv,ti,vi: ExponentialSmoothing(yt.values, trend=p['trend'], seasonal=p['seasonal'], seasonal_periods=p['sp']).fit().forecast(len(Xv))
        elif best_model_name in ['LSTM', 'GRU']:
            p = {'hidden_units': trial.suggest_int('hidden_units', 16, 64), 'layers': trial.suggest_int('layers', 1, 3), 'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True), 'epochs': trial.suggest_int('epochs', 50, 150)}
            def fn(p,Xt,yt,Xv,ti,vi):
                m = SimpleRNN(Xt.shape[1], p['hidden_units'], p['layers'], best_model_name)
                opt = optim.Adam(m.parameters(), lr=p['lr'])
                xt_t = data.X_dl[ti]
                yt_t = data.y_dl[ti]
                xv_t = data.X_dl[vi]
                for _ in range(p['epochs']):
                    opt.zero_grad()
                    nn.MSELoss()(m(xt_t), yt_t).backward()
                    opt.step()
                m.eval()
                with torch.no_grad(): return m(xv_t).numpy().flatten()
        else: return 9999
        
        rmse, metrics = evaluate_cv(fn, p)
        trial.set_user_attr('metrics', metrics)
        return rmse
    except: return 9999

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)
best_params = study.best_params
tuned_metrics = study.best_trial.user_attrs['metrics']

print("\\n✅ Best Parameters:", best_params)
""",
"""# Cell 10: Retrain the Best Model Using the Entire Dataset
print(f"Retraining {best_model_name} on ALL data...")
X_all, y_all, y_all_arr, dates_all = data.X, data.y, data.y_stat, data.df_prophet['ds']
final_model = None

if best_model_name == 'XGBoost':
    final_model = xgb.XGBRegressor(**best_params, random_state=42, objective='reg:squarederror').fit(X_all, y_all)
elif best_model_name == 'RandomForest':
    final_model = RandomForestRegressor(**best_params, random_state=42).fit(X_all, y_all)
elif best_model_name == 'LightGBM':
    final_model = lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1).fit(X_all, y_all)
elif best_model_name == 'CatBoost':
    final_model = cb.CatBoostRegressor(**best_params, random_state=42, verbose=False).fit(X_all, y_all)
elif best_model_name == 'Prophet':
    final_model = Prophet(yearly_seasonality=True, changepoint_prior_scale=best_params.get('cps', 0.05), seasonality_prior_scale=best_params.get('sps', 10.0))
    final_model.fit(pd.DataFrame({'ds': dates_all, 'y': y_all}))
elif best_model_name == 'SARIMA':
    final_model = SARIMAX(y_all_arr, order=best_params['order'], seasonal_order=best_params['s_order'], enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
elif best_model_name == 'ETS':
    final_model = ExponentialSmoothing(y_all_arr, trend=best_params['trend'], seasonal=best_params['seasonal'], seasonal_periods=best_params['sp']).fit()
elif best_model_name in ['LSTM', 'GRU']:
    final_model = SimpleRNN(X_all.shape[1], best_params['hidden_units'], best_params['layers'], best_model_name)
    opt = optim.Adam(final_model.parameters(), lr=best_params['lr'])
    for _ in range(best_params.get('epochs', 100)):
        opt.zero_grad()
        nn.MSELoss()(final_model(data.X_dl), data.y_dl).backward()
        opt.step()

print("✅ Final Model Retrained on 100% Data!")
""",
"""# Cell 11: Forecast Next 12 Months
horizon = 12
dates_all = data.df_prophet['ds']
future_dates = pd.date_range(start=dates_all.iloc[-1] + pd.DateOffset(months=1), periods=horizon, freq='ME')
forecast = []
base_rmse = tuned_metrics['RMSE']['Mean']

if 'df_feat' in locals():
    non_exo = ['date', 'WQI_al_score', 'month', 'quarter', 'year', 'season', 'summer', 'rainy', 'winter']
    exo_cols = [c for c in df_feat.columns if c not in non_exo and not c.startswith('lag_') and not c.startswith('rolling_')]
    monthly_exo_avg = df_feat.groupby('month')[exo_cols].mean()

if best_model_name in ['XGBoost', 'RandomForest', 'LightGBM', 'CatBoost', 'LSTM', 'GRU']:
    curr_feat = data.X.iloc[-1].copy()
    hist_preds = list(data.y.iloc[-12:].values)
    
    for i in range(horizon):
        m = future_dates[i].month
        curr_feat['month'] = m
        curr_feat['quarter'] = future_dates[i].quarter
        curr_feat['year'] = future_dates[i].year
        s = 1 if m in [2,3,4,5] else 2 if m in [6,7,8,9,10] else 3
        curr_feat['season'] = s
        curr_feat['summer'], curr_feat['rainy'], curr_feat['winter'] = int(s==1), int(s==2), int(s==3)
        
        for col in exo_cols:
            if col in curr_feat: curr_feat[col] = monthly_exo_avg.loc[m, col]
            
        for lag in [12, 6, 3, 2, 1]:
            if f'lag_{lag}' in curr_feat and len(hist_preds) >= lag:
                curr_feat[f'lag_{lag}'] = hist_preds[-lag]
                
        for w in [3, 6, 12]:
            if f'rolling_mean_{w}' in curr_feat and len(hist_preds) >= w:
                curr_feat[f'rolling_mean_{w}'] = np.mean(hist_preds[-w:])
            if f'rolling_std_{w}' in curr_feat and len(hist_preds) >= w:
                curr_feat[f'rolling_std_{w}'] = np.std(hist_preds[-w:])
                
        if best_model_name in ['LSTM', 'GRU']:
            final_model.eval()
            with torch.no_grad():
                X_t = torch.tensor(data.scaler.transform(pd.DataFrame([curr_feat])), dtype=torch.float32).unsqueeze(1)
                pred = final_model(X_t).numpy()[0,0]
        else:
            pred = final_model.predict(pd.DataFrame([curr_feat]))[0]
            
        forecast.append(pred)
        hist_preds.append(pred)

elif best_model_name == 'Prophet':
    forecast = final_model.predict(final_model.make_future_dataframe(periods=horizon, freq='ME'))['yhat'].iloc[-horizon:].values

elif best_model_name in ['SARIMA', 'ETS']:
    forecast = final_model.forecast(steps=horizon)

forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': np.round(forecast,2)})
display(forecast_df)
""",
"""# Cell 12: Visualization
fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(x=data.df_prophet['ds'], y=data.y, mode='lines+markers', name='Historical WQI', line=dict(color='blue')))
conn_d, conn_y = [data.df_prophet['ds'].iloc[-1], forecast_df['Date'].iloc[0]], [data.y.iloc[-1], forecast_df['Forecast'].iloc[0]]
fig_fc.add_trace(go.Scatter(x=conn_d, y=conn_y, mode='lines', line=dict(dash='dash', color='red'), showlegend=False))
fig_fc.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines+markers', name='Forecast WQI', line=dict(color='red', dash='dash')))
fig_fc.update_layout(
    title=f"WQI 12-Month Forecast ({best_model_name})", 
    template="plotly_white",
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="lightgray",
        borderwidth=1
    )
)
fig_fc.show()

if hasattr(final_model, 'feature_importances_'):
    fi = pd.DataFrame({'Feature': data.X.columns, 'Importance': final_model.feature_importances_}).sort_values('Importance').tail(15)
    fig_fi = px.bar(fi, x='Importance', y='Feature', orientation='h', title='Top 15 Feature Importances')
    fig_fi.show()
""",
"""# Cell 13: Export
os.makedirs('output', exist_ok=True)
forecast_df.to_csv('output/forecast.csv', index=False)
forecast_df.to_json('output/forecast.json', orient='records', date_format='iso')
comparison_df.to_csv('output/comparison.csv', index=False)

if best_model_name in ['LSTM', 'GRU']:
    torch.save(final_model.state_dict(), 'output/best_model.pth')
elif best_model_name == 'Prophet':
    from prophet.serialize import model_to_json
    with open('output/best_model.json', 'w') as f: json.dump(model_to_json(final_model), f)
else:
    joblib.dump(final_model, 'output/best_model.pkl')

print("✅ Pipeline execution complete. Files exported to output/ folder.")
"""
]

nb = nbf.v4.new_notebook()
for code in cells_content:
    cell = nbf.v4.new_code_cell(code)
    nb.cells.append(cell)

with open('train_models.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Created train_models.ipynb with 13 cells (Walk-Forward).")
