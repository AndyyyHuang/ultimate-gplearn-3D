import copy
from copy import deepcopy
import numpy as np
import pandas as pd
import numba as nb
from numba import jit, prange

import talib as ta

from functions import _Function


def rolling_window(a, window, axis=0):
    """
    返回2D array的滑窗array的array
    """
    if axis == 0:
        shape = (a.shape[0] - window + 1, window, a.shape[-1])
        strides = (a.strides[0],) + a.strides
        a_rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    elif axis == 1:
        shape = (a.shape[-1] - window + 1,) + (a.shape[0], window)
        strides = (a.strides[-1],) + a.strides
        a_rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return a_rolling


@jit(nopython=True,nogil=True,parallel=True)
def calc_zscore_2d(series,rolling_window=180):
    res=series.copy()#初始填充原始值，不是nan
    symbol_num=len(series[0,:])
    for i in prange(rolling_window,len(series)):
        temp=series[i+1-rolling_window:i+1,:]
        # s_mean=np.nanmean(temp,axis=0)
        # s_std=np.nanstd(temp,axis=0)
        for j in prange(symbol_num):
            s_mean=np.nanmean(temp[:,j])
            s_std=np.nanstd(temp[:,j])
            res[i,j] = (series[i,j]-s_mean)/max(s_std,10e-9)
    return res


def rolling_nanmean(A, window=None):
    ret = pd.DataFrame(A)
    factor_table = copy.deepcopy(ret)
    for col in ret.columns:
        current_data = copy.deepcopy(ret[col])
        current_data.dropna(inplace=True)
        current_factor = current_data.rolling(window).mean().values
        number = 0
        for index, data in enumerate(ret[col]):

            if ret[col][index] != ret[col][index]:
                factor_table[col][index] = np.nan
            else:
                factor_table[col][index] = current_factor[number]
                number += 1
    factor = factor_table.to_numpy(dtype=np.double)
    return factor


def rolling_max(A, window=None):
    # ret = np.full(A.shape, np.nan)
    # A_rolling = rolling_window(A, window=window, axis=0)
    # Atmp = np.stack(map(lambda x:np.max(x, axis=0), A_rolling))
    # ret[window-1:,:] = Atmp
    ret = pd.DataFrame(A)
    factor = ret.rolling(window).max()
    factor = factor.to_numpy(dtype=np.double)
    return factor


def rolling_nanstd(A, window=None):
    ret = pd.DataFrame(A)
    factor_table = copy.deepcopy(ret)
    for col in ret.columns:
        current_data = copy.deepcopy(ret[col])
        current_data.dropna(inplace=True)
        current_factor = current_data.rolling(window).std().values
        number = 0
        for index, data in enumerate(ret[col]):

            if ret[col][index] != ret[col][index]:
                factor_table[col][index] = np.nan
            else:
                factor_table[col][index] = current_factor[number]
                number += 1
    factor = factor_table.to_numpy(dtype=np.double)
    return factor


"""
def _ts_corr(X: pd.DataFrame, t):
    return (pd.Series(X.iloc[:, 0]).rolling(t).corr(pd.Series(X.iloc[:, 1]))).to_numpy(dtype=np.double)
"""
def _ts_delay(x1, t):
    return pd.DataFrame(x1).shift(t).values

def _ts_delta(x1, t):
    return x1 - _ts_delay(x1, t)



def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))


def _ts_std(x1, t):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_nanstd(x1, t)


def _ts_mean(x1, t):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_nanmean(x1, t)


def _ts_max(x1, t):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_max(x1, t)

def _ts_normalize_180(x1):
    with np.errstate(over='ignore', under='ignore'):
        return calc_zscore_2d(x1, 180)

"""---------------------------------------------Ta-lib Integration---------------------------------------------"""



"""------Overlap Studies Functions------"""

def _BBANDS(x1: np.ndarray, t) -> np.ndarray:
    """
    Calculate the Bollinger Bands for each column of the input 2D array.

    Parameters:
    - x1: 2D Numpy array (represents closing prices).
    - t: Time period for calculating Bollinger Bands.

    Returns:
    - 2D Numpy array with the upper Bollinger Band values. (Can be adjusted for middle and lower bands)
    """
    try:
        factor_table = copy.deepcopy(x1)
        
        for col in range(x1.shape[1]):
            upper, middle, lower = ta.BBANDS(x1[:, col], timeperiod=t)
            factor_table[:, col] = (upper - middle) / (middle - lower)
        
        return factor_table
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_bbands = _Function(function=_BBANDS, name='ts_bbands', arity=0, isRandom=(True, [7, 14, 21, 28]), need_param=['close'])


def _DEMA(x1, t):
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            factor_table = copy.deepcopy(x1)
            for col in range(x1.shape[1]):
                factor_table[:, col] = ta.DEMA(x1[:, col], t)
            return factor_table
    except:
        return np.full(x1.shape, np.nan)
ts_dema = _Function(function=_DEMA, name='ts_dema', arity=1, isRandom=(True, [7, 14, 21, 28]))

def _HT_TRENDMODE(x1: np.ndarray) -> np.ndarray:
    """
    Calculate the Hilbert Transform - Trend vs Cycle Mode for each column of the input 2D array.

    Parameters:
    - x1: 2D Numpy array (represents closing prices).

    Returns:
    - 2D Numpy array with the Trend vs Cycle Mode values.
    """
    try:
        factor_table = copy.deepcopy(x1)
        
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.HT_TRENDMODE(x1[:, col])
        
        return factor_table
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_ht_trendmode = _Function(function=_HT_TRENDMODE, name='ts_ht_trendmode', arity=0, isRandom=(False, []), need_param=['close'])


def _KAMA(x1, t):
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            factor_table = copy.deepcopy(x1)
            for col in range(x1.shape[1]):
                factor_table[:, col] = ta.KAMA(x1[:, col], t)
            return factor_table
    except:
        return np.full(x1.shape, np.nan)
ts_kama = _Function(function=_KAMA, name='ts_kama', arity=1, isRandom=(True, [7, 14, 21, 28]))

def _MIDPOINT(x1, t):
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            factor_table = copy.deepcopy(x1)
            for col in range(x1.shape[1]):
                factor_table[:, col] = ta.MIDPOINT(x1[:, col], t)
            return factor_table
    except:
        return np.full(x1.shape, np.nan)
ts_midpoint = _Function(function=_MIDPOINT, name='ts_midpoint', arity=1, isRandom=(True, [7, 14, 21, 28]))

def _MIDPRICE(x1: np.ndarray, x2: np.ndarray, t) -> np.ndarray:
    try:
        factor_table = copy.deepcopy(x1)
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.MIDPRICE(x1[:, col], x2[:, col], timeperiod=t)
        return factor_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_midprice = _Function(function=_MIDPRICE, name='ts_midprice', arity=0, isRandom=(True, [7, 14, 21, 28]), need_param=['high', 'low'])

def _SAR(x1: np.ndarray, x2: np.ndarray, acceleration=0.02, maximum=0.2) -> np.ndarray:
    try:
        factor_table = copy.deepcopy(x1)
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.SAR(x1[:, col], x2[:, col], acceleration=acceleration, maximum=maximum)
        return factor_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_sar = _Function(function=_SAR, name='ts_sar', arity=0, isRandom=(False, []), need_param=['high', 'low'])


def _SMA(x1: np.ndarray, t) -> np.ndarray:
    """
    Calculate the SMA for each column of the input 2D array.

    Parameters:
    - x1: 2D Numpy array (represents closing prices).
    - t: Period for the moving average.

    Returns:
    - 2D Numpy array with the SMA values.
    """
    try:
        factor_table = copy.deepcopy(x1)
        
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.SMA(x1[:, col], timeperiod=t)
        
        return factor_table
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_sma = _Function(function=_SMA, name='ts_sma', arity=1, isRandom=(True, [7, 14, 21, 28]))


def _TEMA(x1: np.ndarray, t) -> np.ndarray:
    try:
        factor_table = copy.deepcopy(x1)
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.TEMA(x1[:, col], timeperiod=t)
        return factor_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_tema = _Function(function=_TEMA, name='ts_tema', arity=1, isRandom=(True, [7, 14, 21, 28]))

def _TRIMA(x1: np.ndarray, t) -> np.ndarray:
    try:
        factor_table = copy.deepcopy(x1)
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.TRIMA(x1[:, col], timeperiod=t)
        return factor_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_trima = _Function(function=_TRIMA, name='ts_trima', arity=1, isRandom=(True, [7, 14, 21, 28]))






"""-------Momentum Indicator Functions-------"""


def _ADX(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, t=14) -> np.ndarray:
    try:
        factor_table = copy.deepcopy(x1)
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.ADX(x1[:, col], x2[:, col], x3[:, col], timeperiod=t)
        return factor_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_adx = _Function(function=_ADX, name='ts_adx', arity=0, isRandom=(True, [7, 14, 21, 28]), need_param=['high', 'low', 'close'])

def _ADXR(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, t) -> np.ndarray:
    try:
        factor_table = copy.deepcopy(x1)
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.ADXR(x1[:, col], x2[:, col], x3[:, col], timeperiod=t)
        return factor_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_adxr = _Function(function=_ADXR, name='ts_adxr', arity=0, isRandom=(True, [7, 14, 21, 28]), need_param=['high', 'low', 'close'])

def _APO(x1: np.ndarray, t) -> np.ndarray:
    try:
        factor_table = copy.deepcopy(x1)
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.APO(x1[:, col], fastperiod=t, slowperiod=2*(t+1))
        return factor_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_apo = _Function(function=_APO, name='ts_apo', arity=0, isRandom=(True, [7, 14, 21, 28]), need_param=['close'])

def _STOCHRSI(x1: np.ndarray, t, fastk_period=5, fastd_period=3, fastd_matype=0) -> np.ndarray:
    try:
        factor_table = copy.deepcopy(x1)
        for col in range(x1.shape[1]):
            fastk, fastd = ta.STOCHRSI(x1[:, col], timeperiod=t, fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)
            factor_table[:, col] = fastk/fastd  # You can also choose to return fastd or both
        return factor_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_stochrsi = _Function(function=_STOCHRSI, name='ts_stochrsi', arity=0, isRandom=(True, [7, 14, 21, 28]), need_param=['close'])


def _AROONOSC(x1: np.ndarray, x2: np.ndarray, t) -> np.ndarray:
    try:
        factor_table = copy.deepcopy(x1)
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.AROONOSC(x1[:, col], x2[:, col], timeperiod=t)
        return factor_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_aroonosc = _Function(function=_AROONOSC, name='ts_aroonosc', arity=0, isRandom=(True, [7, 14, 21, 28]), need_param=['high', 'low'])

def _BOP(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, x4: np.ndarray) -> np.ndarray:
    """
    Calculate the Balance of Power (BOP) for each column of the input 2D arrays.

    Parameters:
    - x1: 2D Numpy array (represents open prices).
    - x2: 2D Numpy array (represents high prices).
    - x3: 2D Numpy array (represents low prices).
    - x4: 2D Numpy array (represents close prices).

    Returns:
    - 2D Numpy array with the BOP values.
    """
    try:
        factor_table = copy.deepcopy(x1)
        
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.BOP(x1[:, col], x2[:, col], x3[:, col], x4[:, col])
        
        return factor_table
            
    except Exception as e:
        # Ideally, you'd log the exception here for debugging
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

# Assuming the `_Function` class/factory is defined somewhere:
ts_bop = _Function(function=_BOP, name='ts_bop', arity=0, isRandom=(False, []), need_param=['open', 'high', 'low', 'close'])

def _CCI(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, t: int = 14) -> np.ndarray:
    """
    Calculate the CCI for each column of the input 2D arrays.

    Parameters:
    - x1: 2D Numpy array (represents high prices).
    - x2: 2D Numpy array (represents low prices).
    - x3: 2D Numpy array (represents close prices).
    - t: Time period for calculating CCI.

    Returns:
    - 2D Numpy array with the CCI values.
    """
    try:
        factor_table = copy.deepcopy(x1)
        
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.CCI(x1[:, col], x2[:, col], x3[:, col], timeperiod=t)
        
        return factor_table
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_cci = _Function(function=_CCI, name='ts_cci', arity=0, isRandom=(True, [7, 14, 21]), need_param=['high', 'low', 'close'])

def _CMO(x1: np.ndarray, t) -> np.ndarray:
    try:
        factor_table = copy.deepcopy(x1)
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.CMO(x1[:, col], timeperiod=t)
        return factor_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_cmo = _Function(function=_CMO, name='ts_cmo', arity=0, isRandom=(True, [7, 14, 21, 28]), need_param=['close'])

def _DX(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, t) -> np.ndarray:
    try:
        factor_table = copy.deepcopy(x1)
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.DX(x1[:, col], x2[:, col], x3[:, col], timeperiod=t)
        return factor_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_dx = _Function(function=_DX, name='ts_dx', arity=0, isRandom=(True, [7, 14, 21, 28]), need_param=['high', 'low', 'close'])

def _MACD(x1: np.ndarray, t) -> np.ndarray:
    """
    Calculate the MACD for each column of the input 2D array.

    Parameters:
    - x1: 2D Numpy array (represents closing prices).

    Returns:
    - 2D Numpy array with the MACD values.
    """
    try:
        factor_table = copy.deepcopy(x1)
        
        for col in range(x1.shape[1]):
            macd_line, signal_line, hist = ta.MACD(x1[:, col], fastperiod=t, slowperiod=2*(t+1), signalperiod=t-4)
            factor_table[:, col] = macd_line
        
        return factor_table
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_macd = _Function(function=_MACD, name='ts_macd', arity=0, isRandom=(True, [7, 14, 21, 28]), need_param=['close'])

def _MFI(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, x4: np.ndarray, t) -> np.ndarray:
    try:
        factor_table = copy.deepcopy(x1)
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.MFI(x1[:, col], x2[:, col], x3[:, col], x4[:, col], timeperiod=t)
        return factor_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_mfi = _Function(function=_MFI, name='ts_mfi', arity=0, isRandom=(True, [7, 14, 21, 28]), need_param=['high', 'low', 'close', 'volume'])

def _MINUS_DI(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, t) -> np.ndarray:
    try:
        factor_table = copy.deepcopy(x1)
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.MINUS_DI(x1[:, col], x2[:, col], x3[:, col], timeperiod=t)
        return factor_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_minus_di = _Function(function=_MINUS_DI, name='ts_minus_di', arity=0, isRandom=(True, [7, 14, 21, 28]), need_param=['high', 'low', 'close'])

def _MINUS_DM(x1: np.ndarray, x2: np.ndarray, t) -> np.ndarray:
    try:
        factor_table = copy.deepcopy(x1)
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.MINUS_DM(x1[:, col], x2[:, col], timeperiod=t)
        return factor_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_minus_dm = _Function(function=_MINUS_DM, name='ts_minus_dm', arity=0, isRandom=(True, [7, 14, 21, 28]), need_param=['high', 'low'])

def _MOM(x1: np.ndarray, t) -> np.ndarray:
    """
    Calculate the Momentum for each column of the input 2D array.

    Parameters:
    - x1: 2D Numpy array (represents closing prices).
    - t: Time period for calculating Momentum.

    Returns:
    - 2D Numpy array with the Momentum values.
    """
    try:
        factor_table = copy.deepcopy(x1)
        
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.MOM(x1[:, col], timeperiod=t)
        
        return factor_table
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_mom = _Function(function=_MOM, name='ts_mom', arity=0, isRandom=(True, [7, 14, 21, 28]), need_param=['close'])

def _PLUS_DI(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, t) -> np.ndarray:
    try:
        factor_table = copy.deepcopy(x1)
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.PLUS_DI(x1[:, col], x2[:, col], x3[:, col], timeperiod=t)
        return factor_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_plus_di = _Function(function=_PLUS_DI, name='ts_plus_di', arity=0, isRandom=(True, [7, 14, 21, 28]), need_param=['high', 'low', 'close'])

def _PLUS_DM(x1: np.ndarray, x2: np.ndarray, t) -> np.ndarray:
    try:
        factor_table = copy.deepcopy(x1)
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.PLUS_DM(x1[:, col], x2[:, col], timeperiod=t)
        return factor_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_plus_dm = _Function(function=_PLUS_DM, name='ts_plus_dm', arity=0, isRandom=(True, [7, 14, 21, 28]), need_param=['high', 'low'])


def _PPO(x1: np.ndarray, t) -> np.ndarray:
    try:
        factor_table = copy.deepcopy(x1)
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.PPO(x1[:, col], fastperiod=t, slowperiod=2*(t+1))
        return factor_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_ppo = _Function(function=_PPO, name='ts_ppo', arity=0, isRandom=(True, [7, 14, 21, 28]), need_param=['close'])

def _ROC(x1: np.ndarray, t: int = 10) -> np.ndarray:
    """
    Calculate the Rate of Change for each column of the input 2D array.

    Parameters:
    - x1: 2D Numpy array (represents closing prices).
    - t: Time period for calculating ROC.

    Returns:
    - 2D Numpy array with the ROC values.
    """
    try:
        factor_table = copy.deepcopy(x1)
        
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.ROC(x1[:, col], timeperiod=t)
        
        return factor_table
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_roc = _Function(function=_ROC, name='ts_roc', arity=0, isRandom=(True, [7, 14, 21, 28]), need_param=['close'])


def _RSI(x1: np.ndarray, t) -> np.ndarray:
    """
    Calculate the RSI for each column of the input 2D array.

    Parameters:
    - x1: 2D Numpy array (represents closing prices).

    Returns:
    - 2D Numpy array with the RSI values.
    """
    try:
        factor_table = copy.deepcopy(x1)
        
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.RSI(x1[:, col], timeperiod=t)
        
        return factor_table
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_rsi = _Function(function=_RSI, name='ts_rsi', arity=0, isRandom=(True, [7, 14, 21, 28]), need_param=['close'])

def _STOCH(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
    """
    Calculate the Stochastic Oscillator for each column of the input 2D arrays.

    Parameters:
    - x1: 2D Numpy array (represents high prices).
    - x2: 2D Numpy array (represents low prices).
    - x3: 2D Numpy array (represents close prices).

    Returns:
    - 2D Numpy array with the Stochastic Oscillator K values. (Can be adjusted for D values)
    """
    try:
        factor_table = copy.deepcopy(x1)
        
        for col in range(x1.shape[1]):
            slowk, slowd = ta.STOCH(x1[:, col], x2[:, col], x3[:, col])
            factor_table[:, col] = slowk / slowd
        
        return factor_table
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_stoch = _Function(function=_STOCH, name='ts_stoch', arity=0, isRandom=(False, []), need_param=['high', 'low', 'close'])



def _TRIX(x1: np.ndarray, t) -> np.ndarray:
    try:
        factor_table = copy.deepcopy(x1)
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.TRIX(x1[:, col], timeperiod=t)
        return factor_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_trix = _Function(function=_TRIX, name='ts_trix', arity=0, isRandom=(True, [7, 14, 21, 28]), need_param=['close'])


def _ULTOSC(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, t) -> np.ndarray:
    try:
        factor_table = copy.deepcopy(x1)
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.ULTOSC(x1[:, col], x2[:, col], x3[:, col], timeperiod1=t, timeperiod2=2*t, timeperiod3=4*t)
        return factor_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_ultosc = _Function(function=_ULTOSC, name='ts_ultosc', arity=0, isRandom=(True, [7, 14, 21, 28]), need_param=['high', 'low', 'close'])


def _WILLR(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, t: int = 14) -> np.ndarray:
    """
    Calculate the Williams %R for each column of the input 2D arrays.

    Parameters:
    - x1: 2D Numpy array (represents high prices).
    - x2: 2D Numpy array (represents low prices).
    - x3: 2D Numpy array (represents close prices).
    - t: Time period for calculating Williams %R.

    Returns:
    - 2D Numpy array with the Williams %R values.
    """
    try:
        factor_table = copy.deepcopy(x1)
        
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.WILLR(x1[:, col], x2[:, col], x3[:, col], timeperiod=t)
        
        return factor_table
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_willr = _Function(function=_WILLR, name='ts_willr', arity=0, isRandom=(True, [7, 14, 21, 28]), need_param=['high', 'low', 'close'])


"""------Volume Indicator Functions------"""

def _AD(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, x4: np.ndarray) -> np.ndarray:
    try:
        factor_table = copy.deepcopy(x1)
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.AD(x1[:, col], x2[:, col], x3[:, col], x4[:, col])
        return factor_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_ad = _Function(function=_AD, name='ts_ad', arity=0, need_param=['high', 'low', 'close', 'volume'])

def _ADOSC(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, x4: np.ndarray, fastperiod=3, slowperiod=10) -> np.ndarray:
    try:
        factor_table = copy.deepcopy(x1)
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.ADOSC(x1[:, col], x2[:, col], x3[:, col], x4[:, col], fastperiod=fastperiod, slowperiod=slowperiod)
        return factor_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_adosc = _Function(function=_ADOSC, name='ts_adosc', arity=0, need_param=['high', 'low', 'close', 'volume'])

def _OBV(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Calculate the OBV for each column of the input 2D array.

    Parameters:
    - x1: 2D Numpy array (represents closing prices).
    - x2: 2D Numpy array (represents volume).

    Returns:
    - 2D Numpy array with the OBV values.
    """
    try:
        factor_table = copy.deepcopy(x1)
        
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.OBV(x1[:, col], x2[:, col])
        
        return factor_table
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_obv = _Function(function=_OBV, name='ts_obv', arity=0, isRandom=(False, []), need_param=['close', 'volume'])

"""------Volatility Indicator Functions------"""

def _NATR(x1, x2, x3, t):
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            factor_table = copy.deepcopy(x1)
            for col in range(x1.shape[1]):
                factor_table[:, col] = ta.NATR(x1[:, col], x2[:, col], x3[:, col], t)
            return factor_table
    except:
        return np.full(x1.shape, np.nan)
ts_natr = _Function(function=_NATR, name='ts_natr', arity=0, isRandom=(True, [7, 14, 21, 28]), need_param=['high', 'low', 'close'])

def _ATR(x1, x2, x3, t):
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            factor_table = copy.deepcopy(x1)
            for col in range(x1.shape[1]):
                factor_table[:, col] = ta.ATR(x1[:, col], x2[:, col], x3[:, col], t)
            return factor_table
    except:
        return np.full(x1.shape, np.nan)
ts_atr = _Function(function=_ATR, name='ts_atr', arity=0, isRandom=(True, [7, 14, 21, 28]), need_param=['high', 'low', 'close'])

def _TRANGE(x1, x2, x3):
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            factor_table = copy.deepcopy(x1)
            for col in range(x1.shape[1]):
                factor_table[:, col] = ta.TRANGE(x1[:, col], x2[:, col], x3[:, col])
            return factor_table
    except:
        return np.full(x1.shape, np.nan)
ts_trange = _Function(function=_TRANGE, name='ts_trange', arity=0, need_param=['high', 'low', 'close'])



"""-----Price Transform Functions-----"""

def _AVGPRICE(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, x4: np.ndarray) -> np.ndarray:
    """
    Calculate the Average Price for each column of the input 2D arrays.

    Parameters:
    - x1: 2D Numpy array (represents opening prices).
    - x2: 2D Numpy array (represents high prices).
    - x3: 2D Numpy array (represents low prices).
    - x4: 2D Numpy array (represents closing prices).

    Returns:
    - 2D Numpy array with the Average Price values.
    """
    try:
        factor_table = copy.deepcopy(x1)
        
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.AVGPRICE(x1[:, col], x2[:, col], x3[:, col], x4[:, col])
        
        return factor_table
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_avgprice = _Function(function=_AVGPRICE, name='ts_avgprice', arity=0, isRandom=(False, []), need_param=['open', 'high', 'low', 'close'])

def _MEDPRICE(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Calculate the Median Price for each column of the input 2D arrays.

    Parameters:
    - x1: 2D Numpy array (represents high prices).
    - x2: 2D Numpy array (represents low prices).

    Returns:
    - 2D Numpy array with the Median Price values.
    """
    try:
        factor_table = copy.deepcopy(x1)
        
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.MEDPRICE(x1[:, col], x2[:, col])
        
        return factor_table
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_medprice = _Function(function=_MEDPRICE, name='ts_medprice', arity=0, isRandom=(False, []), need_param=['high', 'low'])

def _TYPPRICE(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
    """
    Calculate the Typical Price for each column of the input 2D arrays.

    Parameters:
    - x1: 2D Numpy array (represents high prices).
    - x2: 2D Numpy array (represents low prices).
    - x3: 2D Numpy array (represents closing prices).

    Returns:
    - 2D Numpy array with the Typical Price values.
    """
    try:
        factor_table = copy.deepcopy(x1)
        
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.TYPPRICE(x1[:, col], x2[:, col], x3[:, col])
        
        return factor_table
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_typprice = _Function(function=_TYPPRICE, name='ts_typprice', arity=0, isRandom=(False, []), need_param=['high', 'low', 'close'])


def _WCLPRICE(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
    """
    Calculate the Weighted Close Price for each column of the input 2D arrays.

    Parameters:
    - x1: 2D Numpy array (represents high prices).
    - x2: 2D Numpy array (represents low prices).
    - x3: 2D Numpy array (represents closing prices).

    Returns:
    - 2D Numpy array with the Weighted Close Price values.
    """
    try:
        factor_table = copy.deepcopy(x1)
        
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.WCLPRICE(x1[:, col], x2[:, col], x3[:, col])
        
        return factor_table
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_wclprice = _Function(function=_WCLPRICE, name='ts_wclprice', arity=0, isRandom=(False, []), need_param=['high', 'low', 'close'])









"""------Cycle indicators-------"""

def _HT_DCPERIOD(x1: np.ndarray) -> np.ndarray:
    """
    Calculate the Hilbert Transform - Dominant Cycle Period for each column of the input 2D array.

    Parameters:
    - x1: 2D Numpy array (represents closing prices).

    Returns:
    - 2D Numpy array with the Dominant Cycle Period values.
    """
    try:
        factor_table = copy.deepcopy(x1)
        
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.HT_DCPERIOD(x1[:, col])
        
        return factor_table
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_ht_dcperiod = _Function(function=_HT_DCPERIOD, name='ts_ht_dcperiod', arity=0, isRandom=(False, []), need_param=['close'])


def _HT_DCPHASE(x1: np.ndarray) -> np.ndarray:
    """
    Calculate the Hilbert Transform - Dominant Cycle Phase for each column of the input 2D array.

    Parameters:
    - x1: 2D Numpy array (represents closing prices).

    Returns:
    - 2D Numpy array with the Dominant Cycle Phase values.
    """
    try:
        factor_table = copy.deepcopy(x1)
        
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.HT_DCPHASE(x1[:, col])
        
        return factor_table
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_ht_dcphase = _Function(function=_HT_DCPHASE, name='ts_ht_dcphase', arity=0, isRandom=(False, []), need_param=['close'])

def _HT_PHASOR(x1: np.ndarray) -> np.ndarray:
    """
    Calculate the Hilbert Transform - Phasor Components for each column of the input 2D array.

    Parameters:
    - x1: 2D Numpy array (represents closing prices).

    Returns:
    - 2D Numpy array with the Phasor Components values.
    """
    try:
        factor_table = copy.deepcopy(x1)
        
        for col in range(x1.shape[1]):
            inPhase, quadrature = ta.HT_PHASOR(x1[:, col])
            factor_table[:, col] = inPhase  # Using inPhase component. Adjust if needed.
        
        return factor_table
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_ht_phasor = _Function(function=_HT_PHASOR, name='ts_ht_phasor', arity=0, isRandom=(False, []), need_param=['close'])


def _HT_SINE(x1: np.ndarray) -> np.ndarray:
    """
    Calculate the Hilbert Transform - SineWave for each column of the input 2D array.

    Parameters:
    - x1: 2D Numpy array (represents closing prices).

    Returns:
    - 2D Numpy array with the SineWave values.
    """
    try:
        factor_table = copy.deepcopy(x1)
        
        for col in range(x1.shape[1]):
            sine, leadsine = ta.HT_SINE(x1[:, col])
            factor_table[:, col] = sine  # Using sine component. Adjust if needed.
        
        return factor_table
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_ht_sine = _Function(function=_HT_SINE, name='ts_ht_sine', arity=0, isRandom=(False, []), need_param=['close'])




"""-----Statistic Functions------"""

def _BETA(x1, x2, t):
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            factor_table = copy.deepcopy(x1)
            for col in range(x1.shape[1]):
                factor_table[:, col] = ta.BETA(x1[:, col], x2[:, col], t)
            return factor_table
    except:
        return np.full(x1.shape, np.nan)
ts_beta = _Function(function=_BETA, name='ts_beta', arity=2, isRandom=(True, [7, 14, 21, 28]))

def _CORREL(x1, x2, t):
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            factor_table = copy.deepcopy(x1)
            for col in range(x1.shape[1]):
                factor_table[:, col] = ta.CORREL(x1[:, col], x2[:, col], t)
            return factor_table
    except:
        return np.full(x1.shape, np.nan)
ts_correl = _Function(function=_CORREL, name='ts_correl', arity=2, isRandom=(True, [7, 14, 21, 28]))

def _LINEARREG(x1: np.ndarray, t) -> np.ndarray:
    try:
        factor_table = copy.deepcopy(x1)
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.LINEARREG(x1[:, col], timeperiod=t)
        return factor_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_linearreg = _Function(function=_LINEARREG, name='ts_linearreg', arity=1, isRandom=(True, [7, 14, 21, 28]))

def _LINEARREG_ANGLE(x1, t):
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            factor_table = copy.deepcopy(x1)
            for col in range(x1.shape[1]):
                factor_table[:, col] = ta.LINEARREG_ANGLE(x1[:, col], t)
            return factor_table
    except:
        return np.full(x1.shape, np.nan)
ts_linearreg_angle = _Function(function=_LINEARREG_ANGLE, name='ts_linearreg_angle', arity=1, isRandom=(True, [7, 14, 21, 28]))

def _LINEARREG_INTERCEPT(x1: np.ndarray, t) -> np.ndarray:
    try:
        factor_table = copy.deepcopy(x1)
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.LINEARREG_INTERCEPT(x1[:, col], timeperiod=t)
        return factor_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_linearreg_intercept = _Function(function=_LINEARREG_INTERCEPT, name='ts_linearreg_intercept', arity=1, isRandom=(True, [7, 14, 21, 28]))

def _LINEARREG_SLOPE(x1, t):
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            factor_table = copy.deepcopy(x1)
            for col in range(x1.shape[1]):
                factor_table[:, col] = ta.LINEARREG_SLOPE(x1[:, col], t)
            return factor_table
    except:
        return np.full(x1.shape, np.nan)
ts_linearreg_slope = _Function(function=_LINEARREG_SLOPE, name='ts_linearreg_slope', arity=1, isRandom=(True, [7, 14, 21, 28]))

def _TSF(x1: np.ndarray, t) -> np.ndarray:
    try:
        factor_table = copy.deepcopy(x1)
        for col in range(x1.shape[1]):
            factor_table[:, col] = ta.TSF(x1[:, col], timeperiod=t)
        return factor_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(x1.shape, np.nan)

ts_tsf = _Function(function=_TSF, name='ts_tsf', arity=1, isRandom=(True, [7, 14, 21, 28]))




def _cal_rolling_ic(y_pred, y):
    with np.errstate(divide='ignore', invalid='ignore'):
        
        no_future_beta = []
        for i in range(len(y_pred)):
            no_future_beta.append(spearmanr(y_pred[i], y[i])[0])
        no_future_beta = pd.Series(no_future_beta).shift(1).fillna(0)
        no_future_beta_rolling = no_future_beta.rolling(rolling_window_2).mean()
        weighted_factor_table = y_pred * no_future_beta_rolling
        
        return weighted_factor_table.to_numpy(dtype=np.double)

def _alpha_pool_2(x1, x2, close):
    try:
        x1 = deepcopy(x1)
        x2 = deepcopy(x2)
        if (x1 is None) or (x2 is None):
            return np.full(x1.shape, np.nan)
        
        rolling_window_1 = 180
        rolling_window_2 = 90
        y = (pd.DataFrame(close).diff(1).shift(-1) / pd.DataFrame(close)).to_numpy(dtype=np.double)
    
        
        with np.errstate(divide='ignore', invalid='ignore'):
            # normalized y_pred
            x1 = calc_zscore_2d(x1, rolling_window_1)
            x2 = calc_zscore_2d(x2, rolling_window_1)
            x1 = _cal_rolling_ic(x1, y)
            x2 = _cal_rolling_ic(x2, y)
            x = x1 + x2
            return x
    except:
        return np.full(x1.shape, np.nan)
alpha_pool_2 = _Function(function=_alpha_pool_2, name='alpha_pool_2', arity=2, need_param=['close'])

def _alpha_pool_3(x1, x2, x3, close):
    try:
        x1 = deepcopy(x1)
        x2 = deepcopy(x2)
        x3 = deepcopy(x3)
        if (x1 is None) or (x2 is None) or (x3 is None):
            return np.full(x1.shape, np.nan)
        
        rolling_window_1 = 180
        rolling_window_2 = 90
        y = (pd.DataFrame(close).diff(1).shift(-1) / pd.DataFrame(close)).to_numpy(dtype=np.double)
    
        
        with np.errstate(divide='ignore', invalid='ignore'):
            # normalized y_pred
            x1 = calc_zscore_2d(x1, rolling_window_1)
            x2 = calc_zscore_2d(x2, rolling_window_1)
            x3 = calc_zscore_2d(x3, rolling_window_1)
            x1 = _cal_rolling_ic(x1, y)
            x2 = _cal_rolling_ic(x2, y)
            x3 = _cal_rolling_ic(x3, y)
            x = x1 + x2 + x3
            return x
    except:
        return np.full(x1.shape, np.nan)
alpha_pool_3 = _Function(function=_alpha_pool_3, name='alpha_pool_3', arity=3, need_param=['close'])



dynamic_ts_std = _Function(function=_ts_std, name='dynamic_ts_std', arity=1, isRandom=(True,[7, 14, 21, 28]))
dynamic_ts_mean = _Function(function=_ts_mean, name='dynamic_ts_mean', arity=1, isRandom=(True,[7, 14, 21, 28]))
dynamic_ts_max = _Function(function=_ts_max, name='dynamic_ts_max', arity=1, isRandom=(True,[7, 14, 21, 28]))
ts_delay = _Function(function=_ts_delay, name='ts_delay', arity=1, isRandom=(True, [1, 3, 5, 7, 12, 14]))
ts_delta = _Function(function=_ts_delta, name='ts_delta', arity=1, isRandom=(True, [1, 3, 5, 7, 12, 14]))
ts_normalize_180 = _Function(function=_ts_normalize_180, name='ts_normalize_180', arity=1)


_extra_function_map = {
    "dynamic_ts_std":dynamic_ts_std,
    "dynamic_ts_mean":dynamic_ts_mean,
    "dynamic_ts_max":dynamic_ts_max,
    "ts_delay": ts_delay,
    "ts_delta": ts_delta,
    "ts_normalize_180":ts_normalize_180,
    "ts_bbands": ts_bbands,
    "ts_dema": ts_dema,
    "ts_ht_trendmode": ts_ht_trendmode,
    "ts_kama": ts_kama,
    "ts_midpoint": ts_midpoint,
    "ts_midprice": ts_midprice,
    "ts_sar": ts_sar,
    "ts_sma": ts_sma,
    "ts_tema": ts_tema,
    "ts_trima": ts_trima,
    "ts_adx": ts_adx,
    "ts_adxr": ts_adxr,
    "ts_apo": ts_apo,
    "ts_stochrsi": ts_stochrsi,
    "ts_aroonosc": ts_aroonosc,
    "ts_bop": ts_bop,
    "ts_cci": ts_cci,
    "ts_cmo": ts_cmo,
    "ts_dx": ts_dx,
    "ts_macd": ts_macd,
    "ts_mfi": ts_mfi,
    "ts_minus_di": ts_minus_di,
    "ts_minus_dm": ts_minus_dm,
    "ts_mom": ts_mom,
    "ts_plus_di": ts_plus_di,
    "ts_plus_dm": ts_plus_dm,
    "ts_ppo": ts_ppo,
    "ts_roc": ts_roc,
    "ts_rsi": ts_rsi,
    "ts_stoch": ts_stoch,
    "ts_trix": ts_trix,
    "ts_ultosc": ts_ultosc,
    "ts_willr": ts_willr,
    "ts_ad": ts_ad,
    "ts_adosc": ts_adosc,
    "ts_obv": ts_obv,
    "ts_natr": ts_natr,
    "ts_atr": ts_atr,
    "ts_trange": ts_trange,
    "ts_avgprice": ts_avgprice,
    "ts_medprice": ts_medprice,
    "ts_typprice": ts_typprice,
    "ts_wclprice": ts_wclprice,
    "ts_ht_dcperiod": ts_ht_dcperiod,
    "ts_ht_dcphase": ts_ht_dcphase,
    "ts_ht_phasor": ts_ht_phasor,
    "ts_ht_sine": ts_ht_sine,
    "ts_beta": ts_beta,
    "ts_correl": ts_correl,
    "ts_linearreg": ts_linearreg,
    "ts_linearreg_angle": ts_linearreg_angle,
    "ts_linearreg_intercept": ts_linearreg_intercept,
    "ts_linearreg_slope": ts_linearreg_slope,
    "ts_tsf": ts_tsf
}
