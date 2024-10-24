"""Metrics to evaluate the fitness of a program.

The :mod:`gplearn.fitness` module contains some metric with which to evaluate
the computer programs created by the :mod:`gplearn.genetic` module.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numbers
import copy
import numpy as np
from joblib import wrap_non_picklable_objects
from scipy.stats import rankdata, spearmanr
import pandas as pd
# import empyrical
__all__ = ['make_fitness']

from numba import jit,prange
import numba as nb
from copy import  deepcopy

@jit(nopython=True,nogil=True,parallel=True)
def calc_zscore_2d(series,rolling_window):
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



class _Fitness(object):

    """A metric to measure the fitness of a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting floating point score quantifying the quality of the program's
    representation of the true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    """

    def __init__(self, function, greater_is_better):
        self.function = function
        self.greater_is_better = greater_is_better
        self.sign = 1 if greater_is_better else -1

    def __call__(self, *args):
        return self.function(*args)


def make_fitness(*, function, greater_is_better, wrap=True):
    """Make a fitness measure, a metric scoring the quality of a program's fit.

    This factory function creates a fitness measure object which measures the
    quality of a program's fit and thus its likelihood to undergo genetic
    operations into the next generation. The resulting object is able to be
    called with NumPy vectorized arguments and return a resulting floating
    point score quantifying the quality of the program's representation of the
    true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom metrics is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    """
    if not isinstance(greater_is_better, bool):
        raise ValueError('greater_is_better must be bool, got %s'
                         % type(greater_is_better))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))
    if function.__code__.co_argcount != 3:
        raise ValueError('function requires 3 arguments (y, y_pred, w),'
                         ' got %d.' % function.__code__.co_argcount)
    if not isinstance(function(np.array([1, 1]),
                      np.array([2, 2]),
                      np.array([1, 1])), numbers.Number):
        raise ValueError('function must return a numeric.')

    if wrap:
        return _Fitness(function=wrap_non_picklable_objects(function),
                        greater_is_better=greater_is_better)
    return _Fitness(function=function,
                    greater_is_better=greater_is_better)




def _weighted_pearson_3D(y, y_pred, w, rolling_window_1=180, rolling_window_2=90, fee=0.0003):
    """Calculate the weighted Pearson correlation coefficient."""
    # y: array - like, shape = [n_samples] -> [n_dates, n_stocks]
    y = y[np.where(w == 1)]
    y_pred = y_pred[np.where(w == 1)]
    with np.errstate(divide='ignore', invalid='ignore'):
        n_dates,n_stocks = y.shape
        total_IC = 0.0
        iter_number = 0
        for current_date in range(n_dates):
            # 首先需要把两边的nan的值全部同时删掉，相当于取交集
            y_pred_cur_date = copy.deepcopy(y_pred[current_date,:])
            y_current_date = copy.deepcopy(y[current_date,:])
            for i in range(len(y_current_date)):
                if y_current_date[i] != y_current_date[i] or y_pred_cur_date[i] != y_pred_cur_date[i]:
                    y_current_date[i] = np.nan
                    y_pred_cur_date[i] = np.nan

            if np.sum(np.isnan(y_current_date)) == len(y_current_date) or np.sum(np.isnan(y_pred_cur_date)) == len(y_pred_cur_date):
                continue
            y_pred_demean =y_pred_cur_date - np.nanmean(y_pred_cur_date)
            y_demean = y_current_date - np.nanmean(y_current_date)
            corr =np.nanmean(np.nansum(y_pred_demean * y_demean) /
                (np.sqrt(np.nansum(np.square(y_pred_demean))) *
                np.sqrt(np.nansum(np.square(y_demean)))))
            if corr != corr:
                continue
            total_IC += corr
            iter_number+=1
        if iter_number>0:
            total_IC = total_IC/iter_number
        else:
            total_IC = 0.0
    if np.isfinite(total_IC):
        return np.abs(total_IC)
    return 0.


def _Alert_weighted_pearson_3D(y, y_pred, w, rolling_window_1=180, rolling_window_2=90, fee=0.0003):
    """Calculate the weighted Pearson correlation coefficient."""
    # y: array - like, shape = [n_samples] -> [n_dates, n_stocks]
    y = y[np.where(w == 1)]
    y_pred = y_pred[np.where(w == 1)]
    with np.errstate(divide='ignore', invalid='ignore'):
        n_dates,n_stocks = y.shape
        total_IC = 0.0
        iter_number = 0
        for current_date in range(n_dates):
            # 首先需要把两边的nan的值全部同时删掉，相当于取交集
            y_pred_cur_date = copy.deepcopy(y_pred[current_date,:])
            y_current_date = copy.deepcopy(y[current_date,:])
            for i in range(len(y_current_date)):
                if y_current_date[i] != y_current_date[i] or y_pred_cur_date[i] != y_pred_cur_date[i]:
                    y_current_date[i] = np.nan
                    y_pred_cur_date[i] = np.nan

            if np.sum(np.isnan(y_current_date)) == len(y_current_date) or np.sum(np.isnan(y_pred_cur_date)) == len(y_pred_cur_date):
                continue
            y_pred_demean =y_pred_cur_date - np.nanmean(y_pred_cur_date)
            y_demean = y_current_date - np.nanmean(y_current_date)
            corr =np.nanmean(np.nansum(y_pred_demean * y_demean) /
                (np.sqrt(np.nansum(np.square(y_pred_demean))) *
                np.sqrt(np.nansum(np.square(y_demean)))))
            if corr != corr:
                continue
            total_IC += corr
            iter_number+=1
        if iter_number>0:
            total_IC = total_IC/iter_number
        else:
            total_IC = 0.0
    if np.isfinite(total_IC):
        return total_IC
    return 0.




def _weighted_spearman_3D(y, y_pred, w, rolling_window_1=180, rolling_window_2=90, fee=0.0003):
    """Calculate the weighted Pearson correlation coefficient."""
    # y: array - like, shape = [n_samples] -> [n_dates, n_stocks]
    y = y[np.where(w==1)]
    y_pred = y_pred[np.where(w==1)]
    with np.errstate(divide='ignore', invalid='ignore'):
        n_dates,n_stocks = y.shape
        total_IC = 0.0
        iter_number = 0
        for current_date in range(n_dates):
            # 首先需要把两边的nan的值全部同时删掉，相当于取交集
            y_pred_cur_date = copy.deepcopy(y_pred[current_date,:])
            y_current_date = copy.deepcopy(y[current_date,:])
            y_pred_cur_date = np.apply_along_axis(rankdata, 0, y_pred_cur_date)
            y_current_date = np.apply_along_axis(rankdata, 0, y_current_date)

            for i in range(len(y_current_date)):
                if y_current_date[i] != y_current_date[i] or y_pred_cur_date[i] != y_pred_cur_date[i]:
                    y_current_date[i] = np.nan
                    y_pred_cur_date[i] = np.nan

            if np.sum(np.isnan(y_current_date)) == len(y_current_date) or np.sum(np.isnan(y_pred_cur_date)) == len(y_pred_cur_date):
                continue
            y_pred_demean =y_pred_cur_date - np.nanmean(y_pred_cur_date)
            y_demean = y_current_date - np.nanmean(y_current_date)
            corr =np.nanmean(np.nansum(y_pred_demean * y_demean) /
                (np.sqrt(np.nansum(np.square(y_pred_demean))) *
                np.sqrt(np.nansum(np.square(y_demean)))))
            if corr != corr:
                continue
            total_IC += corr
            iter_number+=1
        if iter_number>0:
            total_IC = total_IC/iter_number
        else:
            total_IC = 0.0
    if np.isfinite(total_IC):
        return np.abs(total_IC)
    return 0.


def _Alert_weighted_spearman_3D(y, y_pred, w, rolling_window_1=180, rolling_window_2=90, fee=0.0003):
    """Calculate the weighted Pearson correlation coefficient."""
    # y: array - like, shape = [n_samples] -> [n_dates, n_stocks]
    y = y[np.where(w==1)]
    y_pred = y_pred[np.where(w==1)]
    with np.errstate(divide='ignore', invalid='ignore'):
        n_dates,n_stocks = y.shape
        total_IC = 0.0
        iter_number = 0
        for current_date in range(n_dates):
            # 首先需要把两边的nan的值全部同时删掉，相当于取交集
            y_pred_cur_date = copy.deepcopy(y_pred[current_date,:])
            y_current_date = copy.deepcopy(y[current_date,:])
            y_pred_cur_date = np.apply_along_axis(rankdata, 0, y_pred_cur_date)
            y_current_date = np.apply_along_axis(rankdata, 0, y_current_date)

            for i in range(len(y_current_date)):
                if y_current_date[i] != y_current_date[i] or y_pred_cur_date[i] != y_pred_cur_date[i]:
                    y_current_date[i] = np.nan
                    y_pred_cur_date[i] = np.nan

            if np.sum(np.isnan(y_current_date)) == len(y_current_date) or np.sum(np.isnan(y_pred_cur_date)) == len(y_pred_cur_date):
                continue
            y_pred_demean =y_pred_cur_date - np.nanmean(y_pred_cur_date)
            y_demean = y_current_date - np.nanmean(y_current_date)
            corr =np.nanmean(np.nansum(y_pred_demean * y_demean) /
                (np.sqrt(np.nansum(np.square(y_pred_demean))) *
                np.sqrt(np.nansum(np.square(y_demean)))))
            if corr != corr:
                continue
            total_IC += corr
            iter_number+=1
        if iter_number>0:
            total_IC = total_IC/iter_number
        else:
            total_IC = 0.0
    if np.isfinite(total_IC):
        return total_IC
    return 0.






def _weighted_Information_Ratio_3D(y,y_pred,w, rolling_window_1=180, rolling_window_2=90, fee=0.0003):

    y = y[np.where(w==1)]
    y_pred = y_pred[np.where(w==1)]
    with np.errstate(divide='ignore', invalid='ignore'):
        n_dates,n_stocks = y.shape
        IC_list = []
        for current_date in range(n_dates):
            # 首先需要把两边的nan的值全部同时删掉，相当于取交集
            y_pred_cur_date = copy.deepcopy(y_pred[current_date,:])
            y_current_date = copy.deepcopy(y[current_date,:])
            for i in range(len(y_current_date)):
                if y_current_date[i] != y_current_date[i] or y_pred_cur_date[i] != y_pred_cur_date[i]:
                    y_current_date[i] = np.nan
                    y_pred_cur_date[i] = np.nan

            if np.sum(np.isnan(y_current_date)) == len(y_current_date) or np.sum(np.isnan(y_pred_cur_date)) == len(y_pred_cur_date):
                continue
            y_pred_demean =y_pred_cur_date - np.nanmean(y_pred_cur_date)
            y_demean = y_current_date - np.nanmean(y_current_date)
            corr =np.nanmean(np.nansum(y_pred_demean * y_demean) /
                (np.sqrt(np.nansum(np.square(y_pred_demean))) *
                np.sqrt(np.nansum(np.square(y_demean)))))
            if corr != corr:
                continue
            IC_list.append(corr)

        if len(IC_list)>0:
            IR = np.nanmean(IC_list)/np.nanstd(IC_list)
        else:
            IR = 0.0
    if np.isfinite(IR):
        return np.abs(IR)
    return 0.


def _Alert_weighted_Information_Ratio_3D(y,y_pred,w, rolling_window_1=180, rolling_window_2=90, fee=0.0003):
    
    y = y[np.where(w==1)]
    y_pred = y_pred[np.where(w==1)]
    with np.errstate(divide='ignore', invalid='ignore'):
        n_dates,n_stocks = y.shape
        IC_list = []
        for current_date in range(n_dates):
            # 首先需要把两边的nan的值全部同时删掉，相当于取交集
            y_pred_cur_date = copy.deepcopy(y_pred[current_date,:])
            y_current_date = copy.deepcopy(y[current_date,:])
            for i in range(len(y_current_date)):
                if y_current_date[i] != y_current_date[i] or y_pred_cur_date[i] != y_pred_cur_date[i]:
                    y_current_date[i] = np.nan
                    y_pred_cur_date[i] = np.nan

            if np.sum(np.isnan(y_current_date)) == len(y_current_date) or np.sum(np.isnan(y_pred_cur_date)) == len(y_pred_cur_date):
                continue
            y_pred_demean =y_pred_cur_date - np.nanmean(y_pred_cur_date)
            y_demean = y_current_date - np.nanmean(y_current_date)
            corr =np.nanmean(np.nansum(y_pred_demean * y_demean) /
                (np.sqrt(np.nansum(np.square(y_pred_demean))) *
                np.sqrt(np.nansum(np.square(y_demean)))))
            if corr != corr:
                continue
            IC_list.append(corr)

        if len(IC_list)>0:
            IR = np.nanmean(IC_list)/np.nanstd(IC_list)
        else:
            IR = 0.0
    if np.isfinite(IR):
        return np.abs(IR)
    return 0.



def _bt_sharpe_old_version(y, y_pred, w, rolling_window_1=180, rolling_window_2=90, fee=0.0003):
    """Calculate the weighted Pearson correlation coefficient."""
    # y: array - like, shape = [n_samples] -> [n_dates, n_stocks]
    
    if y_pred is None:
        return -1.

    y = y[np.where(w == 1)]
    y_pred = y_pred[np.where(w == 1)]

    # normalized y_pred
    y_pred = calc_zscore_2d(y_pred, rolling_window_1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        n_dates,n_stocks = y.shape
        """
        for current_date in range(n_dates):
            # 首先需要把两边的nan的值全部同时删掉，相当于取交集
            y_pred_cur_date = copy.deepcopy(y_pred[current_date,:])
            y_current_date = copy.deepcopy(y[current_date,:])
            for i in range(len(y_current_date)):
                if y_current_date[i] != y_current_date[i] or y_pred_cur_date[i] != y_pred_cur_date[i]:
                    y_current_date[i] = np.nan
                    y_pred_cur_date[i] = np.nan

            if np.sum(np.isnan(y_current_date)) == len(y_current_date) or np.sum(np.isnan(y_pred_cur_date)) == len(y_pred_cur_date):
                continue
        """   

        no_future_beta = []
        origin_signals_df = np.array([0]*y_pred.shape[1])
        for i in range(len(y_pred)):
            no_future_beta.append(spearmanr(y_pred[i], y[i])[0])
            signals = np.select(condlist=[y_pred[i] > np.quantile(y_pred[i], 0.8), y_pred[i] < np.quantile(y_pred[i], 0.2)],
                                choicelist=[1, -1], default=0)
            origin_signals_df = np.vstack((origin_signals_df, signals))
        origin_signals_df = origin_signals_df[1:, :]
        no_future_beta = pd.Series(no_future_beta).shift(1).fillna(0)
        y_notna = y[no_future_beta.rolling(rolling_window_2).mean().notna()]
        origin_signals_df_notna = origin_signals_df[no_future_beta.rolling(rolling_window_2).mean().notna()]
        no_future_beta_rolling = no_future_beta.rolling(rolling_window_2).mean().dropna()
        signal_matrix = np.diag(np.sign(no_future_beta_rolling)) @ origin_signals_df_notna

        hedge_ret = np.nanmean(np.select([signal_matrix==1, signal_matrix==-1], [y_notna, -1 * y_notna], default=np.nan), axis=1)
        turnover_rate = np.concatenate([np.abs(signal_matrix[0, :].reshape((1, -1))), np.abs(np.diff(signal_matrix, axis=0))], axis=0).sum(axis=1) / np.abs(signal_matrix[0, :]).sum()
        fee_cost = turnover_rate * fee
        equity = 1+(hedge_ret - fee_cost).cumsum()
        sharpe = np.nanmean(np.diff(equity)) / np.nanstd(np.diff(equity))

    if sharpe is not None:
        if np.isfinite(sharpe):
            return sharpe, equity, hedge_ret, turnover_rate
        
    return -1.


import numba as nb
@nb.njit
def _start_loop_for_fitness(n, m, trade_p_matrix, signal_matrix, taker_fee):
    
    position = np.array([0.0] * m, dtype=np.float64)
    cash = 1000.0
    order_size = cash * 2
    res = np.zeros((n, m+1))
    res.fill(np.nan)
    
    for i in range(n):
        # 获取当前各标的最新价格(open)
        present_price = trade_p_matrix[i, :]
        # 计算当前仓位比重
        tmp_weight = position * present_price / order_size
        # 计算目标仓位比重
        allocated_weight = signal_matrix[i]
        if np.nansum(np.abs(allocated_weight)) == 0:
            allocated_weight = np.array([0.0] * m, dtype=np.float64)
        else:
            allocated_weight = np.array([allocated_weight[i] / np.nansum(np.abs(allocated_weight)) for i in range(m)])

        # 计算要达到allocated_weight 需要交易的实际仓位
        trade_weight = allocated_weight - tmp_weight
        orders_v = trade_weight * order_size / present_price

        trade = -1 * orders_v * present_price * np.where(trade_weight > 0, 1+taker_fee, 1-taker_fee)
        position_change = orders_v

        tmp_position = position + position_change
        tmp_cash = cash + np.sum(trade)

        position = tmp_position
        cash = tmp_cash
        res[i] = np.append(position, cash)

    return res

    
def _bt_sharpe(y, y_pred, w, rolling_window_1=180, rolling_window_2=90, fee=0.0003):
    """Calculate the weighted Pearson correlation coefficient."""
    # y: array - like, shape = [n_samples] -> [n_dates, n_stocks]
    
    if y_pred is None:
        return -1.

    # normalized y_pred and align the y
    y_pred_normalized = y_pred[np.where(w == 1)]
    y_normalized = y[np.where(w == 1)]
    
    y_pred_normalized = calc_zscore_2d(y_pred_normalized, rolling_window_1)[rolling_window_1-1:]
    y_normalized = y_normalized[rolling_window_1-1:]

    
    with np.errstate(divide='ignore', invalid='ignore'):
        no_future_beta = []
        origin_signals_df = np.array([0]*y_pred_normalized.shape[1])
        for i in range(len(y_pred_normalized)):
            no_future_beta.append(spearmanr(y_pred_normalized[i], y_normalized[i])[0])
            signals = np.select(condlist=[y_pred_normalized[i] > np.quantile(y_pred_normalized[i], 0.8), y_pred_normalized[i] < np.quantile(y_pred_normalized[i], 0.2)],
                                choicelist=[1, -1], default=0)
            origin_signals_df = np.vstack((origin_signals_df, signals))
        origin_signals_df = origin_signals_df[1:, :]
        no_future_beta = pd.Series(no_future_beta).shift(1).fillna(0)
        y_notna = y_normalized[no_future_beta.rolling(rolling_window_2, min_periods=rolling_window_2).mean().notna()]
        origin_signals_df_notna = origin_signals_df[no_future_beta.rolling(rolling_window_2, min_periods=rolling_window_2).mean().notna()]
        no_future_beta_rolling = no_future_beta.rolling(rolling_window_2, min_periods=rolling_window_2).mean().dropna()
        signal_matrix = np.diag(np.sign(no_future_beta_rolling)) @ origin_signals_df_notna
    
        present_price = np.concatenate([np.array([1] * y_notna.shape[1]).reshape((1, -1)), 1+y_notna], axis=0).cumprod(axis=0)[:-1]
        n, m = signal_matrix.shape
        res = _start_loop_for_fitness(n, m, present_price, signal_matrix, fee)
        res[:, :-1] = res[:, :-1] * present_price
        equity = res.sum(axis=1)
        sharpe = np.nanmean(np.diff(equity)) / np.nanstd(np.diff(equity))

    if sharpe is not None:
        if np.isfinite(sharpe):
            return sharpe
        
    return -1.

def _bt_pnl(y, y_pred, w, rolling_window_1=180, rolling_window_2=90, fee=0.0003):
    """Calculate the weighted Pearson correlation coefficient."""
    # y: array - like, shape = [n_samples] -> [n_dates, n_stocks]
    
    if y_pred is None:
        return -1.

    # normalized y_pred and align the y
    y_pred_normalized = y_pred[np.where(w == 1)]
    y_normalized = y[np.where(w == 1)]
    
    y_pred_normalized = calc_zscore_2d(y_pred_normalized, rolling_window_1)[rolling_window_1-1:]
    y_normalized = y_normalized[rolling_window_1-1:]

    
    with np.errstate(divide='ignore', invalid='ignore'):
        no_future_beta = []
        origin_signals_df = np.array([0]*y_pred_normalized.shape[1])
        for i in range(len(y_pred_normalized)):
            no_future_beta.append(spearmanr(y_pred_normalized[i], y_normalized[i])[0])
            signals = np.select(condlist=[y_pred_normalized[i] > np.quantile(y_pred_normalized[i], 0.8), y_pred_normalized[i] < np.quantile(y_pred_normalized[i], 0.2)],
                                choicelist=[1, -1], default=0)
            origin_signals_df = np.vstack((origin_signals_df, signals))
        origin_signals_df = origin_signals_df[1:, :]
        no_future_beta = pd.Series(no_future_beta).shift(1).fillna(0)
        y_notna = y_normalized[no_future_beta.rolling(rolling_window_2, min_periods=rolling_window_2).mean().notna()]
        origin_signals_df_notna = origin_signals_df[no_future_beta.rolling(rolling_window_2, min_periods=rolling_window_2).mean().notna()]
        no_future_beta_rolling = no_future_beta.rolling(rolling_window_2, min_periods=rolling_window_2).mean().dropna()
        signal_matrix = np.diag(np.sign(no_future_beta_rolling)) @ origin_signals_df_notna
    
        present_price = np.concatenate([np.array([1] * y_notna.shape[1]).reshape((1, -1)), 1+y_notna], axis=0).cumprod(axis=0)[:-1]
        n, m = signal_matrix.shape
        res = _start_loop_for_fitness(n, m, present_price, signal_matrix, fee)
        res[:, :-1] = res[:, :-1] * present_price
        equity = res.sum(axis=1)
        pnl = (equity[-1] - equity[0]) / equity[0]
        
    if pnl is not None:
        if np.isfinite(pnl):
            return pnl
        
    return -1.


# def _mean_absolute_error(y, y_pred, w):
#     """Calculate the mean absolute error."""
#     return np.average(np.abs(y_pred - y), weights=w)
#
#
# def _mean_square_error(y, y_pred, w):
#     """Calculate the mean square error."""
#     return np.average(((y_pred - y) ** 2), weights=w)
#
#
# def _root_mean_square_error(y, y_pred, w):
#     """Calculate the root mean square error."""
#     return np.sqrt(np.average(((y_pred - y) ** 2), weights=w))
#
#
# def _log_loss(y, y_pred, w):
#     """Calculate the log loss."""
#     eps = 1e-15
#     inv_y_pred = np.clip(1 - y_pred, eps, 1 - eps)
#     y_pred = np.clip(y_pred, eps, 1 - eps)
#     score = y * np.log(y_pred) + (1 - y) * np.log(inv_y_pred)
#     return np.average(-score, weights=w)


# weighted_pearson = _Fitness(function=_weighted_pearson,
#                             greater_is_better=True)
weighted_pearson_3d = _Fitness(function=_weighted_pearson_3D,greater_is_better=True)
alert_weighted_pearson_3d = _Fitness(function=_Alert_weighted_pearson_3D,greater_is_better=True)
# weighted_spearman = _Fitness(function=_weighted_spearman,
#                              greater_is_better=True)

weighted_spearman_3d = _Fitness(function=_weighted_spearman_3D,greater_is_better=True)
alert_weighted_spearman_3d = _Fitness(function=_Alert_weighted_spearman_3D,greater_is_better=True)
# mean_absolute_error = _Fitness(function=_mean_absolute_error,
#                                greater_is_better=False)
weighted_information_ratio = _Fitness(function=_weighted_Information_Ratio_3D,greater_is_better=True)
alert_weighted_information_ratio = _Fitness(function=_Alert_weighted_Information_Ratio_3D,greater_is_better=True)

bt_sharpe = _Fitness(function=_bt_sharpe, greater_is_better=True)
bt_pnl = _Fitness(function=_bt_pnl, greater_is_better=True)
# mean_square_error = _Fitness(function=_mean_square_error,
#                              greater_is_better=False)
# root_mean_square_error = _Fitness(function=_root_mean_square_error,
#                                   greater_is_better=False)
# log_loss = _Fitness(function=_log_loss,
#                     greater_is_better=False)

_fitness_map = {
    # 'pearson': weighted_pearson,
    #             'spearman': weighted_spearman,
    #             'mean absolute error': mean_absolute_error,
    #             'mse': mean_square_error,
    #             'rmse': root_mean_square_error,
    #             'log loss': log_loss

}

_extra_map = {
    "pearson_3d":weighted_pearson_3d,
    "spearman_3d":weighted_spearman_3d,
    "IR":weighted_information_ratio,
    "sharpe":bt_sharpe,
    "pnl":bt_pnl,
    # alert开头的函数都不是用在gplearn里面的计算用的，而是最后show_program出指标的时候看IC符号的。
    # 之所有要这么干是因为，gpelarn里面IC越高越好，不管正负，但我们看的时候还是关注IC方向的。
    "alert_spearman":alert_weighted_spearman_3d,
    "alert_pearson":alert_weighted_pearson_3d,
    "alert_information_ratio":alert_weighted_information_ratio,
}
_fitness_map = dict(_fitness_map, **_extra_map)


