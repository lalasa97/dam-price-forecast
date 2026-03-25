import numpy as np
import pandas as pd

def date_features(df):
    df = df.copy()

    df['dow'] = df['datetime'].dt.dayofweek 
    df['is_sunday'] = (df['dow'] == 6).astype(int)
    df['dow_sin']   = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos']   = np.cos(2 * np.pi * df['dow'] / 7)

    df['day'] = df['datetime'].dt.day
    df['is_month_start'] = (df['day'] <= 3).astype(int)
    df['is_month_end'] = (df['day'] >= 28).astype(int)

    df = df.drop(columns=['day'])

    return df

def hour_block_features(df): 
    df = df.copy()
    df['hour'] = df['datetime'].dt.hour 
    df['is_peak_hours'] = df['hour'].between(19, 22).astype(int)
    df['is_valley_hour'] = df['hour'].between(8, 13).astype(int)

    df['date'] = df['datetime'].dt.date
    # yesterday's hourly average
    hourly_avg = (
        df.groupby(['date', 'hour'])['price']
        .mean()
        .groupby(level=1)
        .shift(1)
    )
    df['yday_same_hour_avg'] = pd.Series(list(zip(df['date'], df['hour']))).map(hourly_avg)
    

    df['block'] = df['hour'] * 4 + df['datetime'].dt.minute  // 15 
    df['sin_block'] = np.sin(2 * np.pi * df['block'] / 96)
    df['cos_block'] = np.cos(2 * np.pi * df['block'] / 96)

    df = df.drop(columns= ['date'])

    return df

def spike_features(df):
    df = df.copy()
    df['is_spike_prev'] = (df['price'].shift(96)>=10000).astype(int)
    df['spike_duration'] = df['is_spike_prev'].rolling(96).sum()
    df['spike_persistence'] = df['is_spike_prev'].rolling(96*3).mean()
    df['lag96_if_spike'] = df['price'].shift(96) * (df['price'].shift(96) == 10000)
    df['lag96_if_normal'] = df['price'].shift(96) * (df['price'].shift(96) < 10000)

    df['spike_magnitude'] = np.maximum(df['price'].shift(96) - 10000, 0)

     # plateau continuation signal
    df['plateau_flag'] = (
        df['price'].shift(96).rolling(4).std() < 50
        ).astype(int)

    return df

def volatility_features(df):
    df = df.copy()
    
    # daily amplitude feature - captures voltality of yesterday
    df['prevday_range'] = (
        df['price'].shift(96).rolling(96).max() -
        df['price'].shift(96).rolling(96).min()
    )
    # peak deviation feature
    df['prevday_peak_dev'] = (
        df['price'].shift(96).rolling(96).max() -
        df['price'].shift(96).rolling(96).mean()
    )
    df['price_prevday_dev'] = (
        df['price'].shift(96) -
        df['price'].shift(96).rolling(96*7).mean()
    )
    df['price_prevday_norm'] = (
        df['price'].shift(96) /
        df['price'].shift(96).rolling(96*7).mean()
    )
    df['vol_1d'] = df['price'].shift(96).rolling(96).std()
    df['vol_3d'] = df['price'].shift(96).rolling(96*3).std()
    df['vol_7d'] = df['price'].shift(96).rolling(96*7).std()
    df['vol_change'] = df['vol_1d'] - df['vol_3d']
    df['is_high_vol'] = (df['vol_1d'] > df['vol_7d']).astype(int) # more spikes likely

    df['is_high_vol'] = (df['vol_1d'] > df['vol_7d']).astype(int)
    df['prevday_peak'] = df['price'].shift(96).rolling(96).max()
    df['distance_to_peak'] = df['prevday_peak'] - df['price'].shift(96)

    return df

def lag_shape_features(df):
    df = df.copy()
    lags = [4*i for i in range(24)]
    lags.extend([96*i for i in range(1, 7)])
  
    for i in lags:
        df[f'price_lag_{i+96}'] = df['price'].shift(i+96)

    df['delta_1d'] = df['price'].shift(96) - df['price'].shift(192)
    df['delta_7d'] = df['price'].shift(96) - df['price'].shift(672)
    
    df['roll_mean_96']  = df['price'].shift(96).rolling(96).mean()   # 24hr
    df['roll_mean_672'] = df['price'].shift(96).rolling(672).mean()  # 1wk
    df['price_trend_3h'] = df['price'].shift(96) - df['price'].shift(96+3*4)
    df['price_trend_6h'] = df['price'].shift(96) - df['price'].shift(96+6*4)

    df['prevday_slope'] = df['price'].shift(96) - df['price'].shift(96+4)
    df['prevday_acceleration'] = df['price'].shift(96) - 2*df['price'].shift(96+4) + df['price'].shift(96+8)
    
    return df

def buy_sell_features(df):
    df = df.copy()
    # Imbalance lag features
    df['diff_buy_sell'] = df['buy_mw']-df['sell_mw']
    df['imbalance_lag_96'] = df['diff_buy_sell'].shift(96)
    df['ratio_lag_96'] = np.log1p(df['buy_mw'].shift(96) / (df['sell_mw'].shift(96) + 1))
    
    # Drop the original buy/sell columns as they are "future" data during prediction
    df = df.drop(columns=['buy_mw', 'sell_mw', 'diff_buy_sell'])
    return df

def midnight_lag_features(df):
    """
    For any row on 2023-10-03 (any time):
    Take midnight → 2023-10-03 00:00
    Then:
    lag_1 → 2023-10-02 23:45
    lag_4 → 2023-10-02 23:00
    lag_8 → 2023-10-02 22:00
    And this should be same for all rows of that day
    """
    df = df.copy()
    df['midnight'] = df['datetime'].dt.floor('D')
    for i in range(96):
        df[f'price_prevday_{i}'] = (
            df['midnight'] - pd.Timedelta(minutes=15*(i+1))
        ).map(df.set_index('datetime')['price'])


    df = df.drop(columns=['midnight'])

    return df 

# Combine all


def build_features(df):
    df = df.copy()

    df = date_features(df)
    df = hour_block_features(df)
    df = spike_features(df)
    df = volatility_features(df)
    df = lag_shape_features(df)
    df = buy_sell_features(df)
    df = midnight_lag_features(df)

    nan_min_time = df.loc[df.isna().any(axis=1), 'datetime'].min()
    nan_max_time = df.loc[df.isna().any(axis=1), 'datetime'].max()
    print(f'Min nan date time: {nan_min_time}')
    print(f'Max nan date time: {nan_max_time}')

    df = df.dropna().reset_index(drop=True)
    return df