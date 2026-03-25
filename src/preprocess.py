import pandas as pd


def preprocess(df):
    df = df.copy()

    # Rename ------
    df = df.rename(columns={
        'Delivery Date': 'date',
        'Time Period': 'time_period',
        'Cleared Buy (MW)': 'buy_mw',
        'Cleared Sell (MW)': 'sell_mw',
        'Price (Rs./MWh)' : 'price'
    })

    # Extract datetime 
    df['start_time'] = df['time_period'].str.split('-').str[0].str.strip()
    
    df['datetime'] = pd.to_datetime(
            df['date'].str.strip() + ' ' + df['start_time'],
            format='%d/%m/%Y %H:%M'
        )
    df = df.drop(columns=['date', 'time_period', 'start_time'])

    # sort + deduplicate
    df = df.sort_values('datetime').drop_duplicates('datetime')

    # Convert numeric columns
    for col in ['buy_mw', 'sell_mw', 'price']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Set index for frequency handling
    df = df.set_index('datetime')

    # enforce frequency (important for rolling features)
    df = df.asfreq('15min')
    
    # handle missing
    # handling missing values - not needed but if needed -
    for col in ['price','buy_mw', 'sell_mw']:
        df[col] = df[col].interpolate()

    df = df.reset_index()

    # Sanity checks
    assert df['datetime'].is_monotonic_increasing
    assert df['datetime'].is_unique
    
    return df