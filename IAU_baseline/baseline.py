import matplotlib
matplotlib.rcParams['backend'] = "QtAgg"  # has to be called before importing plt
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.figure

import warnings
import numpy as np
from math import pi
from scipy.optimize import curve_fit
from inspect import signature
import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Tuple, Sequence
from scipy.interpolate import CubicSpline

from IAU_baseline import fit_functions as fct
from IAU_baseline import ctrl_plots as cp

# =============================================================================
# outlier identification outer loop
# =============================================================================
def find_base(func: fct, x_val: list, y_val: list, y_err: list, flag: Sequence[int],
            direction: str='p', verbose: bool=True, plot: bool=False, ctrl_plots: bool=False, limit: float=0.1,
            stop_rel: bool=True, post_prc_filter: bool=True):
    """
    function to identify outliers and return the baseline. 

    Parameters
    ----------
    func : fct
        fitfunction to use from ol_fir_functions (e.g. fct.higher) 
    
    x_val : list
        list of x values (e.g. time)
    
    y_val : list 
        list of y values (e.g. mole fraction)

    y_err : list
        list of absolute error values of y values (e.g. stdev of mole fraction)
    
    flag : list
        list containg outlier identification flags from a previous run, will be created and initialized
        with flag = len(y_val)*[0] if flag is None

    direction : str
        identifying outliers above or below the baseline or both 
        'p' - above baseline, 'n' below baseline, 'pn' above and below

    plot : bool
        Toggle plot output of baseline

    ctrl_plots : bool
        Toggle additional control plot output 

    limit : float
        stop criterion for outlier search. if the residual changes by less than the limit value,
        no outliers will be identified. Defaluts to 0.1

    stop_rel : bool
        if True, stop iterative outlier detection accorindg to relative change of the
        st.dev. as prescribed by limit, if False, the value supplied as limit will be used as absolute criterion

    post_prc_filter : bool
        if True data identified as outliers with large error bars will be unflagged (3*precision criterion) 

    Returns
    -------
    """

    if stop_rel and (limit >= 1 or 0 > limit):
        raise ValueError('ERROR: To interrupt iteration based on a relative change, '
                         'value of limit has to be betwee 0 and 1. Nothing will be done.\n')

    # date may be read as strings from ASCII data, check if X is not a string
    if isinstance(x_val[0], str):
        raise ValueError('ERROR: X values are strings not numbers or datetime. Nothing will be done.\n')

    if all(np.isnan(x) for x in y_val):
        raise ValueError('ERROR: No valid data in y_val. Nothing will be done.\n')

    # modify x axis
    if isinstance(x_val[0], datetime.datetime):
        print('Found X axis as datetime, converting to fractional year using datetime_to_fractionalyear from toolpac.conv.times')
        x_datetime = x_val
        x_val = datetime_to_fractionalyear(x_datetime, method='exact')

    # check data for possible inconsistencies known to cause issues:
    # warnings are not affected by verbose True/False
    warning = False

    if len(x_val) != len(y_val):
        raise ValueError(f'ERROR: x_val and y_val have different length {len(x_val)} and {len(y_val)}. Nothing will be done.\n')

    if any(np.isnan(x) for x in x_val):
        warnings.warn('WARNING: x_val axis contains NaN.')
        warning = True

    if y_err is not None:
        if len(y_err) != len(y_val):
            raise ValueError('ERROR: y_err and y_val have different length. Nothing will be done.\n')
        if any(np.isnan(y_val) != np.isnan(y_err)):
            warnings.warn('WARNING: Different occurence of NaN found in y_val and y_err. This may cause issues.\n')
            warning = True
        if len([x for x in y_err if x == 0]) > 0:
            warnings.warn('WARNING: y_err contains 0. values. This may cause issues and will create error messages.\n '
                          'y_err will be ignored.\n')
            y_err=None
            warning = True

    # data seems ok if warning is false

    # if no flag list provided create it
    if flag is None:
        prefl = False
        if verbose:
            print('List \'flag\' initialized. All elements set to 0.\n')
        flag = len(y_val)*[0]
    else:
        prefl = True
        flag_in =  flag.copy()
        if verbose:
            print('Pre-flag data found.')

    if verbose:
        if stop_rel:
            print('Limit set to ', 100 * limit, '%.\n')
        else:
            print('Absolute stop criterion set to ', limit, '.\n')

    if isinstance(x_val[0], float):
        if x_val[0] > 1000.:
            x_val -= np.floor(x_val[0])
            if verbose:
                print('X axis first entry seems to contain a large value. Using offset for fitting.')

    # initialize plot if option set True
    cf_fit = None
    cf_ax1 = None
    if plot:
        cf_fit = plt.figure()
        cf_ax1 = cf_fit.add_subplot(111)
        # cf_ax1.set_title(subst)
        # these axis labels do not really make sense
        # plt.xlabel('X_delta')
        # plt.ylabel('Y_val')
        cf_ax1.set_xlim(np.nanmin(x_val) - 0.1, np.nanmax(x_val) + 0.1)
        cf_ax1.set_ylim(np.nanmin(y_val) * 0.9, np.nanmax(y_val) * 1.1)
        cf_ax1.scatter(x_val, y_val, color='lightgray')

    no_final = False

    # first iteration is set to 1 because flag 0 indicates baseline data
    iteration = 1
    # make sure that while loop is executed at least once
    # for the relative stop criterion limit has to be < 1
    change_rel = 1
    # set arbitrary values
    sd_new = 10 * limit
    count_ol = 100

    if stop_rel:
        compare = change_rel
    else:
        compare = sd_new

    while compare > limit and sum(~np.isnan([x for i, x in enumerate(y_val) if flag[i]==0])) > 10 and count_ol > 0:
        tmp = find_ol_next_iteration(func, x_val, y_val, y_err, flag, iteration,
                                     direction=direction, verbose=verbose, plot=plot, ctrl_plots=ctrl_plots,
                                     limit=limit, cf_ax1=cf_ax1, stop_rel=stop_rel)

        change_rel, flag, residual, sd_new, count_ol, popt1 = tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5]
        baseline = func(np.array(x_val), *popt1)

        if stop_rel:
            compare = change_rel
        else:
            compare = sd_new

        iteration = iteration + 1

        # if no outliers were found in last step and stop criterion is set to absolute and post prc filtering is active,
        # check if there are points identified as outliers but close to residual (using x times mean meaurement error)
        if count_ol == 0 and post_prc_filter and not stop_rel:
            outliers = [y if flag[i] != 0 else np.nan for (i, y) in enumerate(y_val)]
            unfilter_flag = 99
            tmp_flag = unfilter_within_prc(outliers, y_err, baseline,
                                           offset=limit, factor=3, flag_value=unfilter_flag,
                                           verbose=verbose, direction=direction)

            flag = [0 if abs(y) == unfilter_flag else x for (x, y) in zip(flag, tmp_flag)]

        if sum(~np.isnan([x for i, x in enumerate(y_val) if flag[i]==0])) < 10:
            if verbose:
                print('too few data points left without limit of change reached')
            no_final = True

    if plot:
        fl_y_val = [y_val[i] if flag[i] != 0 else None for i in range(len(y_val))]
        # needed for points with flag 0 not to be overplotted
        cf_ax1.scatter(x_val, fl_y_val, c=flag)
        cf_ax1.scatter(x_val, baseline, c='dimgray', zorder=1, s=8, label='baseline')

        if not no_final:
            tmp_x_val, tmp_y_val, tmp_y_err = get_no_nan(x_val, y_val, y_err, flagged=True)
            popt1 = fit_data(func, tmp_x_val, tmp_y_val, tmp_y_err)
            yfit = func(np.array(x_val), *popt1)
            cf_ax1.plot(x_val, yfit, color='red', label='no flagging', linewidth=2)
            if prefl:
                tmp_x_val, tmp_y_val, tmp_y_err = get_no_nan(x_val, y_val, y_err, flag=flag_in, flagged=False)
                popt1 = fit_data(func, tmp_x_val, tmp_y_val, tmp_y_err)
                yfit = func(np.array(x_val), *popt1)
                cf_ax1.plot(x_val, yfit, color='green', label='preflagged', linewidth=2)
            tmp_x_val, tmp_y_val, tmp_y_err = get_no_nan(x_val, y_val, y_err, flag=flag, flagged=False)
            popt1 = fit_data(func, tmp_x_val, tmp_y_val, tmp_y_err)
            yfit = func(np.array(x_val), *popt1)
            cf_ax1.plot(x_val, yfit, color='black', label='baseline final', linewidth=2)

        cf_fit.legend()
    else:
        if not no_final:
            tmp_x_val, tmp_y_val, tmp_y_err = get_no_nan(x_val, y_val, y_err, flag=flag, flagged=False)
            popt1 = fit_data(func, tmp_x_val, tmp_y_val, tmp_y_err)
            yfit = func(np.array(x_val), *popt1)

    if verbose:
        print('Fit result: ', popt1)

    if warning:
        print('\n There have been warnings about data issues. Check screen output at beginning of function run.')

    return flag, residual, warning, popt1, baseline


# =============================================================================
# outlier identification: data wrapper (based on work by Sascha Alber)
# =============================================================================
def ol_wrapper(func: fct, x_val: Sequence[float], y_val: Sequence[float], y_err: Sequence[float], flag: Sequence[int],
               plot: bool=True, direction='p', verbose: bool=True, limit: float=0.1, stop_rel: bool=True,
               sub_time: int=None, sub_len: int=0, n_sliding: int=3, post_prc_filter: bool=True, test: bool=False):
    """
    Parameters
    ----------
    func : fct
        fitfunction to use from ol_fir_functions (e.g. fct.higher) 
    
    x_val : list
        list of x values (e.g. time)
    
    y_val : list 
        list of y values (e.g. mole fraction)

    y_err : list
        list of absolute error values of y values (e.g. stdev of mole fraction)
    
    flag : list
        list containg outlier identification flags from a previous run, will be created and initialized
        with flag = len(y_val)*[0] if flag is None

    direction : str
        identifying outliers above or below the baseline or both 
        'p' - above baseline, 'n' below baseline, 'pn' above and below

    plot : bool
        Toggle plot output of baseline

    ctrl_plots : bool
        Toggle additional control plot output 

    limit : float
        stop criterion for outlier search. if the residual changes by less than the limit value,
        no outliers will be identified. Defaluts to 0.1

    stop_rel : bool
        if True, stop iterative outlier detection accorindg to relative change of the
        st.dev. as prescribed by limit, if False, the value supplied as limit will be used as absolute criterion
    
    sub_time : int
        a time interval defining how the timeseries is cut into subsets
        sub_time has to be a relativedelta object. E. g. sub_time = relativedelta(months=6)
        
    sub_len : int
        a number of elements defining how the timeseries is cut into subsets
        only sub_time OR sub_len may be provided
    
    n_sliding : int
        the number of intervals to be used for a sliding interval calculation
        default is 3, i.e. that 1 interval after and 1 before will be included to detect outliers
        but only results from the central interval will be used
        for the first/last interval x subsequent/preceding ones will be used for outlier detection
        n_sliding should be an odd number
        
    post_prc_filter : bool
        if True data identified as outliers with large error bars will be unflagged (3*precision criterion)
        (effective for abssolute stop criterion only)
        
    test : bool
        True will result in no fitting performed but data series cutting point reported

    find_ol can cope with d_mxr and flag being None
    """

    if len(x_val) != len(y_val):
        raise ValueError('ERROR: time and mxr have different length. Nothing will be done.')
    if not y_err is None and len(y_err)!=len(y_val):
        raise ValueError('ERROR: d_mxr and mxr have different length. Nothing will be done.')
    if isinstance(x_val[0], str):
        raise ValueError('ERROR: time values are strings not numbers or datetime. Nothing will be done.\n')

    time_cut = False
    len_cut = False

    if sub_time is not None:
        time_cut = True

    if sub_len is not None:
        len_cut = True
        # if length of intervals is longer than length of dataset then warapping doe not make any sense
        if sub_len > len(x_val):
            warnings.warn('WARNING: parameter sub_len is larger than length of data array. '
                          'Consider to use outlier detection on full dataset instead of wrapping.\n')
            sub_len = len(x_val)-1

    if not time_cut and not len_cut:  # both False
        print('ERROR: Parameters \'sub_time\' and \'sub_len\' both not provided.')
    elif time_cut and len_cut:         # both True
        if verbose:
            print('Parameters \'sub_time\' and \'sub_len\' both provided. \'sub_time\' will be used.')
        len_cut=False

    if verbose:
        if time_cut:
             print('Cutting data by time.')
        elif len_cut:
             print('Cutting data by number of entries.')

    if len_cut and sub_len<=0:
        raise ValueError('ERROR: Parameter  \'sub_len\' must be > 0.')

    if not time_cut and not len_cut:
        raise ValueError('ERROR: Could not determine time or length cutting.')


    first_x = x_val[0]
    last_x  = x_val[-1]
    if isinstance(x_val[0], datetime.datetime):
        x_frac = datetime_to_fractionalyear(x_val, method='exact')
    else:
        x_frac = x_val

    if time_cut:
        if not all(isinstance(x, datetime.date) for x in x_val) :
            raise ValueError('ERROR: Time data supplied is not timestamp. '
                             'Use sub_len to cut by the number of entries or convert in put data.')
        if not isinstance(sub_time, relativedelta):
            raise ValueError('Parameter sub_time has to be a relativedelta object. E. g. sub_time = relativedelta(months=6)')
            # timedelta does not have attributes months ands years
        if not first_x + n_sliding * sub_time < last_x:
            raise ValueError('Time interval is too short. Try shorter sub_time or try using complete dataset.')

    if not flag is None:
        flag_in = flag.copy()

    # initialization
    flag_dummy = 100
    new_flag = len(x_val)*[flag_dummy]
    # 100 is an unlikely value, allows for checking if all data points have been properly evaluated
    # using np.nan would make spotting misidentification more difficult
    new_baseline = len(x_val)*[np.nan]

    # find data sections to do a sliding average
    # initialization
    add_win = int(np.floor(n_sliding/2))
    interval = [0]
    i=0
    start_idx = [0]

    if time_cut:
        '''
            find the index j of time value for which x_val is closest 
            to the start of the timeseries plus one time interval of length sub_time
            this is the end of the first subset of data to be analysed
            take the absolute value of the difference between each data point in x_val and the 
            theoretical endpoint in time (first_x + sub_time))
            locate the mininmum value of the difference running over all indices of x_val using (range(len(x_val))
        '''
        end_idx = [min(range(len(x_val)), key=lambda j: abs(x_val[j]-(first_x + sub_time)))]
        while x_val[end_idx[-1]] < last_x:
            i+=1
            interval.append(i)
            start_idx.append(end_idx[-1]+1)
            # find index of time value in x_val closest to time interval delta in remaining data as above
            # to locate last entry of data subset
            end_idx.append(min(range(len(x_val)), key=lambda j: abs(x_val[j] - (x_val[end_idx[-1]+1] + sub_time))))

    elif len_cut:
        end_idx = [sub_len-1]
        while end_idx[-1] < (len(x_val)-1):
            i+=1
            interval.append(i)
            start_idx.append(end_idx[-1]+1)
            end_idx.append(end_idx[-1] + sub_len)

        # make sure that last interval covers end of timeseries
        # this will result in outlier finding window for last interval becoming larger
        # but otherwise outlier detection would not be performed for last short segment of time series
        end_idx[-1] = len(x_val)-1

    n_intervals = i + 1
    if verbose:
        print(f'{n_intervals} intervals found.')
        print("start indices:", start_idx)
        print("end indices:", end_idx)
        if n_sliding > n_intervals:
            if n_intervals % 2 == 0:  # even number
                n_sliding = n_intervals-1  # Even
            else:  # odd number, incl 1
                n_sliding = n_intervals
            warnings.warn(f"Number of intervals ({n_intervals}) found is smaller than n_sliding value ({n_sliding}). "
                          f"n_sliding set to new value: {n_sliding}")

    case=0
    for i in range(n_intervals):
        # first interval with too few preceding
        if i < add_win or i < n_sliding-add_win:
            tmp_start = 0
            tmp_end = end_idx[n_sliding-1]+1
            case=1
            case_descr = 'first interval with too few preceding'

        # normal intervals from center of timeseries
        elif i >= (n_sliding-add_win) and (i+add_win) < n_intervals:
            tmp_start = start_idx[i-add_win]
            tmp_end = end_idx[i+add_win]+1
            case=2
            case_descr = 'normal interval from center of timeseries'

        # last intervals with too few following
        else:
            tmp_start = start_idx[-n_sliding+1]
            tmp_end = end_idx[-1]+1
            case=3
            case_descr = 'last intervals with too few following'

        center_len = end_idx[i]-start_idx[i]+1

        if verbose:
            print(f'\n Case {case}: {case_descr}')
            if test:
                print(f'Length of input data: {len(x_val)}')
                print(f'Interval number {i} {case_descr} ranging from index {start_idx[i]} to {end_idx[i]}.',
                      f'Sliding window length is  {tmp_end-tmp_start} data points ranging from {tmp_start} to {tmp_end}.')

        tmp_x_val = x_val[tmp_start:tmp_end]

        if any(isinstance(x, datetime.date) for x in tmp_x_val):
            tmp_x_val = datetime_to_fractionalyear(tmp_x_val, method='exact')
        tmp_y_val =  y_val[tmp_start:tmp_end]
        if y_err is None:
            tmp_y_err = None
        else:
            tmp_y_err = y_err[tmp_start:tmp_end]
        if flag is None:
            tmp_flag = None
        else:
            tmp_flag = flag[tmp_start:tmp_end]

        # do outlier fitting in selected window
        if not test:
            if verbose:
                print('iteration:', i)
                print (f'length of data: {len(tmp_x_val)}')
                print('boundaries', start_idx[i], end_idx[i], start_idx[i]-tmp_start, end_idx[i]-tmp_start)
                print('interval length', end_idx[i]+1-start_idx[i], end_idx[i]-tmp_start+1 -(start_idx[i]-tmp_start))

            tmp = find_ol(func, tmp_x_val, tmp_y_val, tmp_y_err, tmp_flag, direction=direction,
                          verbose=verbose, plot=False, ctrl_plots=False, limit=limit, stop_rel=stop_rel)
            new_sub_flag = tmp[0]
            new_sub_baseline = tmp[4]

        # extract subset from fitting window and add to new flag and baseline series
        if not test:
            new_flag[start_idx[i]:end_idx[i]+1] = new_sub_flag[start_idx[i] - tmp_start:end_idx[i] - tmp_start+1]
            new_baseline[start_idx[i]:end_idx[i]+1] = new_sub_baseline[start_idx[i] - tmp_start:end_idx[i] - tmp_start+1]


    if not test:
        tmp_x_val, tmp_y_val, tmp_y_err = get_no_nan(x_frac, y_val, y_err, flag=new_flag, flagged=False)
        popt1 = fit_data(func, tmp_x_val, tmp_y_val, tmp_y_err)
        if verbose:
            print(f'{np.count_nonzero(new_flag)} points flagged as outliers.')
            print('final fit', popt1)

        # smooth baseline
        sm_baseline = smooth_wrapped_baseline(x_frac, new_baseline, plot=False)
        if verbose:
            print('Baseline has been re-sampled and smoothed.')
        if flag_dummy in new_flag:
            warnings.warn('WARNING: remaining initialization values found in flag. Probably something has gone wrong. \n')

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.set_xlim(np.nanmin(x_frac) - 1, np.nanmax(x_frac) + 1)
            ax.set_ylim(np.nanmin(y_val) * 0.9, np.nanmax(y_val) * 1.1)

            # add vertical lines to indicate interval boundaries
            ymin, ymax = ax.get_ylim()
            vert_line_pos = [x_frac[0]] + [x_frac[idx] for idx in end_idx]
            ax.vlines(vert_line_pos, ymin, ymax , color='gray', linestyles='dashed', zorder=0)

            bl_val = [x if new_flag[j]==0 else None for j, x in enumerate(y_val)]
            fl_val = [x if new_flag[j]!=0 else None for j, x in enumerate(y_val)]
            ax.scatter(x_frac, fl_val, c=new_flag)
            ax.scatter(x_frac, bl_val, color='lightgray', label='final baseline data')
            ax.scatter(x_frac, sm_baseline, color='dimgray', label='baseline centre')

            tmp_x_val, tmp_y_val, tmp_y_err = get_no_nan(x_frac, y_val, y_err, flag=new_flag, flagged=False)
            yfit = func(np.array(x_frac), *popt1)
            ax.plot(x_frac, yfit, color='red', label='final baseline fit', linewidth=2)

            anc = AnchoredText(f" sliding {n_sliding}\n sub_len {sub_len} \n sub_time {sub_time} \n"
                               f" outliers {np.count_nonzero(new_flag)} \n func {str(func).split(' ')[1]} \n"
                               f" fit result {popt1}", loc="upper left", frameon=False)
            ax.add_artist(anc)

            fig.legend()
            fig.show()

    else:
        sm_baseline = None

    return new_flag, sm_baseline


# =============================================================================
# outlier identification core function, not supposed to be called directly
# but should be called from find_ol() only
# =============================================================================
def find_ol_next_iteration(func: fct, x_val: Sequence[float], y_val: Sequence[float], y_err: Sequence[float],
                           flag: Sequence[int], iteration: int, direction: str='p', verbose: bool=True,
                           plot: bool=False, ctrl_plots: bool=False, limit: float=0.1,
                           cf_ax1: Tuple[matplotlib.figure.Figure, plt.Axes]=None, stop_rel: bool=True):
    """
    func needs to be a function from ol_fit_functions, supplied as fct.XXX (without quotes)
    x_val, y_val, y_err are components of a dependent and an independent variable, y_err contains abolute error values
    flag is a list/array containg outlier identification flags from a previous run, will be created and initialized
        with flag = len(y_val)*[0] if flag is None
    iteration is the loop number of the outlier identification and should be counted
        in a loop within the calling function, iteration > 0; negative sign will be applied for
        searching outliers below baseline
    verbose: make the function more talkative
    plot: if True a plot window need to be provided as parameter cf_ax1
    ctrl_plots: if True additional control plots will be provided, no window needs to be supplied
    limit: stop criterion for outlier search. if the residual changes by less than the limit value,
        no outliers will be identified
    cf_ax1: plot window to which results are added
    """

    if direction == 'n':
        iteration = iteration * (-1)

    # extract valid data with flag==0
    fit_x_val, fit_y_val, fit_y_err = get_no_nan(x_val, y_val, y_err, flag=flag, flagged=False)

    if len(fit_y_val) < 10:
        if verbose:
            print('Too few valid data points.')
        return(0, 0, None)    # here returning None should be okay as it may occur under regular conditions

    popt1 = fit_data(func, fit_x_val, fit_y_val, fit_y_err)

    # calculate residuals and standard deviations
    yfit = func(np.array(fit_x_val), *popt1)
    fit_res = fit_y_val - yfit  # does not include NaNs

    std_initial = fit_res.std()
    if stop_rel:
        width = 2 * std_initial
    else:
        width = limit

    if y_err is not None:
        if std_initial < np.mean(fit_y_err):
            width = 2 * np.mean(fit_y_err)
            if verbose:
                print("width smaller than mean error of data, using 2*mean(fit_d_mxr)")

    fit_res_subset = fit_res

    if 'p' in direction:
        fit_res_subset = fit_res[fit_res < width]
        fit_res = fit_res_subset
    if 'n' in direction:
        fit_res_subset = fit_res[fit_res > (-1) * width]
    std_new = fit_res_subset.std()

    diff = std_initial - std_new
    change_rel = diff / std_initial

    if verbose:
        print('Change rel:', '{0:.2f}'.format(change_rel * 100), '%')
        print(f' Value of new stdv: {std_new} and stop criterion: {limit}.')
        print(f' Difference: {diff}.')

    # calculate residual and standard deviation
    yfit = func(np.array(x_val), *popt1)
    res = y_val - yfit
    # does include NaNs

    count_ol = 0
    if stop_rel:
        compare = change_rel
    else:
        compare = std_new

    if compare > limit:
        # check for outliers higher than 2 sigma ABOVE residual and flag them
        for j, x in enumerate(res):
            if 'p' in direction:
                if (~np.isnan(x)) & (x > width) & (flag[j] == 0):
                    flag[j] = iteration
            if 'n' in direction:
                if (~np.isnan(x)) & (x < (-1) * width) & (flag[j] == 0):
                    flag[j] = iteration*(-1)

        # count outliers
        if 'p' and 'n' in direction:
            count_ol = list(flag).count(iteration) + list(flag).count((-1) * iteration)
        else:
            count_ol = list(flag).count(iteration)
        if plot:
            if cf_ax1 is None:
                if verbose:
                    print('No plot window provided. Outlier identification was run without plotting results')
            else:
                # add fit to existing graph
                cf_ax1.plot(x_val, yfit, label=iteration, linestyle='dotted')

    # control plots
    if ctrl_plots:
        cp.plot_residual(x_val, res, 0.5*width, flag)
        cp.plot_timeseries(x_val, y_val, yfit, 0.5*width, flag)

    if verbose:
        print(f'iteration {iteration}: {count_ol} {direction} outliers found.')

    return change_rel, flag, res, std_new, count_ol, popt1


# =============================================================================
# fit data with options flagged (fit_all=False) or all (fit_all=True)
# plot relies on a global variable cf_ax1 containing a dataplot that the fit is added to
# which may not work in a special project
# =============================================================================
def fit_data(func: fct, fit_x_val: Sequence[float], fit_y_val: Sequence[float],
             fit_y_err: Sequence[float], verbose: bool=False) -> Sequence[float]:
    # func has to be a function from ol_fit_functions, e. g. fct.simple
    # d_mxr may be None

    func_params = signature(func).parameters
    p0_len = len(func_params) - 1
    p0 = [0.] * p0_len  # does not work if len(func_params) is used directly
    p0[0] = np.mean(fit_y_val)
    if p0_len == 1:
        popt1, pcov1 = curve_fit(func, fit_x_val, fit_y_val,
                                 p0=p0, check_finite=True)
    else:
        p0[1] = fit_y_val[len(fit_y_val)-1] - fit_y_val[0]/ fit_x_val[len(fit_y_val)-1]
        # fit extracted data subset
        popt1, pcov1 = curve_fit(func, fit_x_val, fit_y_val, sigma=fit_y_err,
                                 p0=p0, check_finite=True)

    if verbose:
        print(popt1)

    return(popt1)


# =============================================================================
# extract non-NaN data for fitting
# =============================================================================
def get_no_nan(x_val: Sequence[float], y_val: Sequence[float], y_err: Sequence[float]=None,
               flag:Sequence[int]=None, flagged: bool=True):
    # returns data with NaNs removed (checking at time and mxr)
    # flagged = False removes all flagged data and returns unflagged data only
    # to perform flag check flag array has to be supplied

    # use list() to make this work with arrays, does not cause issues if lists were supplied
    x_val = list(x_val)
    y_val = list(y_val)

    if y_err is not None:
        y_err = list(y_err)
    if flag is not None:
        flag = list(flag)

    tmp_list= [x for x in [x_val, y_val, y_err, flag] if not x is None]

    if len({len(i) for i in tmp_list}) != 1:
        raise ValueError('Time, mxr, d_mxr, flag of different length supplied.')

    if flagged:
        # copy data
        ufl_x_val = x_val
        ufl_y_val = y_val
        ufl_y_err = y_err
    else:
        if flag is None:
            raise ValueError('Flag data not supplied.')

        # extract unflagged dataset without NaNs
        ufl_x_val = [x for i,x in enumerate(x_val) if flag[i]==0]
        ufl_y_val = [x for i,x in enumerate(y_val) if flag[i]==0]
        ufl_y_err = None
        if not y_err is None:
            ufl_y_err = [x for i,x in enumerate(y_err) if flag[i]==0]

    if y_err is None:
        where = list([~np.isnan(ufl_y_val) & ~np.isnan(ufl_x_val)][0])
    else:
        where = list([~np.isnan(ufl_y_val) & ~np.isnan(ufl_y_err) & ~np.isnan(ufl_x_val)][0])

    no_nan_x_val = [x for i,x in enumerate(ufl_x_val) if where[i]]
    no_nan_y_val = [x for i,x in enumerate(ufl_y_val) if where[i]]

    if y_err is None:
        no_nan_y_err = None
    else:
        no_nan_y_err = [x for i, x in enumerate(ufl_y_err) if where[i]]

    return(no_nan_x_val, no_nan_y_val, no_nan_y_err)


# identify points which deviate from the mean data by a given multiple of precision:
# such values should not be counted as extreme values / outliers
# flags are reset to 0
def unfilter_within_prc(y_data: Sequence[float], y_err: Sequence[float], y_base: Sequence[float],
                        offset: float,  factor: int=3, direction: str='pn', flag_value: int=99, verbose: bool=True)\
    ->  Sequence[int]:
    '''
        function written April 2024 by Tanja Schuck
        this function flags datapoints in an interval around a baseline function data
        y_data is a dataset to be filtered
        y_err: absolute error values of y_data
        y_base: decribes the baseline around which points are to be filtered
        offset: a measure for the interval width around function that should be filtered for
        factor: the multiple of offset that defines the interval width
        direction: check above ('p') and/or below ('n') the baseline range  (options 'p', 'n, 'pn', 'np')
        flag_value: value to be used as flag to mark unflitered data
        verbose: set console output
    '''

    if verbose:
        print(f'Checking for values with deviation from baseline function by less than {factor} times the mean error ...')

    tmp_flag = [np.nan] * len(y_data)

    if 'p' in direction:
        tmp_flag = [flag_value if y-factor*err < offset+base else np.nan for (y, err, base) in zip(y_data, y_err, y_base)]

    if 'n' in direction:
        tmp_flag = [-1*flag_value if y+factor*err > base-offset else p_flag
                    for (y,  err, base, p_flag) in zip(y_data, y_err, y_base, tmp_flag)]

    count = sum(map(abs, tmp_flag))/flag_value

    if verbose:
        print(f'{count} values found inside {factor} times {offset} interval after iterative procedure finished.')

    return tmp_flag


# baseline from wrapper is concatenated and likely has sharp steps at interval bounderies
def smooth_wrapped_baseline(x: Sequence[float], y: Sequence[float], plot: bool=True)-> Sequence[float]:
    '''
    function written 8. April 2024 by Tanja Schuck
    status: experimental
    :param x:  x values of dataset to be smoothed
    :param y:  y values of dataset to be smoothed
    :param plot: if True a control plot is created to check the result
    :return: smoothed x dataset
    '''


    x_nonan = [x for (x, y) in zip(x, y) if not np.isnan(y)]
    y_nonan = [y for (x, y) in zip(x, y) if not np.isnan(y)]

    first_x = x_nonan[0]
    last_x = x_nonan[-1]
    ini_length = len(x_nonan)
    # take 0.2 % of initial point density
    # empirical compromise to keep important features and get rid of the unwanted ones
    low_res_length = int(np.floor(0.002 * ini_length))

    # do downsampling
    x_low_res = [first_x + x*(last_x-first_x)/low_res_length for x in range(low_res_length)]
    x_low_res[-1] = x[-1]
    y_low_res = np.interp(x_low_res, x_nonan, y_nonan)

    # cubicSpline approximation
    cs = CubicSpline(x_low_res, y_low_res)

    # do upsampling
    x_high_res = [first_x + x*(last_x-first_x)/ini_length for x in range(ini_length)]
    x_high_res[-1] = x[-1]
    y_high_res = cs(x_high_res)

    if plot:
        fig = plt.figure()
        plt.plot(x,y, 'o', label='initial', ms=3)
        # plt.plot(time, bl, '-', label='initial')

        # plt.plot(x_low_res, y_low_res, 'x', label='downsampled',  ms=3)
        plt.plot(x_low_res, y_low_res, '-', label='downsampled')

        # plt.plot(x_high_res, bl_high_res, 'x', label='upsampled cubicspline',  ms=3)
        plt.plot(x, cs(x), '-', label='upsampled cubicspline orig. res')
        plt.plot(x_high_res, y_high_res, '-', label='upsampled cubicspline high res')

        plt.legend()

    # return approximated smoothed baseline on initial x axis
    return cs(x)


# =============================================================================
# DO NOT CALL IN NEW CODE
# substance loop for backward compatibility
# =============================================================================
# #%%
def ol_iteration_for_subst(subst, data_in, data_flag_in,
                           func=fct.simple, direction='pn', limit=0.1,
                           plot=True, res_return=False):

    '''

    :param subst: name of substance
    :param data_in: DataFrame with input mixing ratio data
    :param data_flag_in: DataFrame with input flag data  CAREFUL will be changed
    :param func: function to be used to fit baseline
    :param direction:  search for 'p'ositive and/or 'n'egative outliers
    :param limit: stop crietion (percentage)
    :param plot: have a plot to check results
    :param res_return: return resudual values
    :return:
    '''

    # data_in and data_flag_in are Dataframes of identical length,
    # both need to contain a column subst and 'd_'+subst with mixing ratios and error values (absolute)
    # direction identifies outliers above (positive) and below (negative) baseline, options are 'p', 'n, 'pn'
    # set parameter res_return to True to have variables data_flag_in and residual returned

    raise ValueError('ERROR: Function ol_iteration_for_subst() cannot be used anymore. '
                     'Use find_ol() and do loop over substances in code outside. '
                     'Nothing will be done.\n')


    if len(data_in) == 0:
        print('no data')
        return
    if len(data_in) != len(data_flag_in):
        print('Dataframes containing data and flags do not match in length.')
        return

    # number of NaNs in data column:       sum(np.isnan(df_merge['n2o']))
    if (len(data_in) - sum(np.isnan(data_in[subst]))) <= 10:
        print('too few data points')
        return
    if sum(np.isnan(data_in[subst])) != sum(np.isnan(data_in['d_'+subst])):
        tmp1 = sum(np.isnan(data_in[subst]))
        tmp2 = sum(np.isnan(data_in['d_'+subst]))
        print(f'Different number of NaN entries in mxr ({tmp1}) '
              f'and error ({tmp2}) columns. Please check input data')

    print(f'\n {subst}: using function {func}')

    # working on the dataframe itself things become VERY slow
    time_col = 'year_delta'
    time = data_in[time_col].values
    mxr = data_in[subst].values
    d_mxr = data_in[f'd_{subst}'].values
    flag = data_flag_in[subst].values  # modifications on this will affect original dataframe and vice versa

    tmp = find_ol(func, time, mxr, d_mxr, flag,
            direction=direction, verbose=True, plot=plot, ctrl_plots=False, limit=0.1)
    data_flag_in[subst] = tmp[0]  # will modify original dataframe, old behaviour
    residual = tmp[1]
    warning = tmp[2]

    if warning:
        print('\n There have been warnings about data issues. Check screen output at beginning of function run.')

    if plot:
        plt.ylabel(subst)

    if res_return:
        return data_flag_in, residual
    else:
        return


# =============================================================================
# time conversion
# =============================================================================


from datetime import datetime as dt

def datetime_to_fractionalyear(date_time, method="exact"):
    """Convert date_time to fractionalyear.

    source: https://stackoverflow.com/questions/6451655/how-to-convert-python-datetime-dates-to-decimal-float-years
    edited in order to handle list-likes and don't accidently add summer time shift

    Parameters:
    ---------
    date_time: datetime object or list-like of datetime objects
    method: string, OPTIONAL
        choose between "exact" and "raw". The default is "exact". "raw" is there
        for historical reasons.

    Returns:
    ---------
    float: fractional year
    """
    if method == "exact":

        def create_fractyear(date_time):
            year = date_time.year
            startOfThisYear = dt(year=year, month=1, day=1)
            startOfNextYear = dt(year=year + 1, month=1, day=1)

            yearElapsed = date_time - startOfThisYear
            yearDuration = startOfNextYear - startOfThisYear
            fraction = yearElapsed.total_seconds() / yearDuration.total_seconds()

            return date_time.year + fraction

    elif method == "raw":

        def create_fractyear(date_time):
            year = date_time.year
            year_frac = (date_time.month - 1) / 12
            month_frac = date_time.day / 30 / 12
            # we do not need to calculate sub-day precision, as the month and day
            # are already not precise.
            return year + year_frac + month_frac

    try:
        fractionalyear = []
        for element in date_time:
            fractionalyear.append(create_fractyear(element))
        return fractionalyear
    except:
        fractionalyear = create_fractyear(date_time)
        return fractionalyear
