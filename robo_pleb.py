import get_data as gd
from scipy import stats
import pdb
import datetime
import gpc
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import os.path
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy


#stage_of_trend: Return the index of the first zero in a trend
def stage_of_trend(t):
    """returns the index of the first zero in a trend"""
    j=-1
    for i in t:
        j = j + 1 
        if i == 0:
            return j

#Trend Negated: returns boolean
def trend_negated(t, series, i, params):
    """returns whether the trend t is negated by the ith value of a series"""

    """Inputs:
    t: the current trend (A, B, C, D, E)
    series: the timeseries for t
    i: the current location in the series
    params: the parameters for determining the trend"""

    s = stage_of_trend(t)

    j = -1
    a, b, c, d, e= t
    if(a==i):
        return False
    
    if series.ix[i][4] < series.ix[a][4]:    
        return True
    if s == 1:
        #Cannot negate the trend here as it could progress higher
        return False
    t_ia = len(series[a:i])-1   # the first index is 0, if there are 11 
                                #observations the x coord of the last one 
                                #will be 10!
    m1, c1, stdev_p1 = time_regression(series, a, i)   
    #If the series' standard deviation is more than param[5] x the first close
    #it is negated:
    if stdev_p1 > params[5]*series.ix[a][2]:
        return True
    if s == 2:
        #Negate the trend if the retracement is larger than params(2)
        if (series.ix[i][4] < (series.ix[b][3]- params[2]*(series.ix[b][3] - series.ix[a][4]))):
            return True
    if s == 3:
        #negate the setup here if the high is 2.5sd above the trend?
        if series.ix[i][3] > c1 + m1*t_ia+params[4]*stdev_p1:
            return True
        #Also negate the trend if B is not the Highest point between a and c
        ac_hi, ac_lo, ac_agv = max_min_mean(series[a:c])
        if(series.ix[b][3] < ac_hi):
            return True
    if s == 4:
        #negate the trend if the retracement is below the time regression
        if series.ix[i][4] < c1 + m1*t_ia - 2*stdev_p1:
            return True
    return False

def down_trend_negated(t, series, i, params):
    """returns whether the trend t is negated by the ith value of a series"""

    """Inputs:
    t: the current trend (A, B, C, D, E)
    series: the timeseries for t
    i: the current location in the series
    params: the parameters for determining the trend"""

    s = stage_of_trend(t)

    j = -1
    a, b, c, d, e= t
    t_ia = len(series[a:1])-1
    if(a==i):
        return False
    
    if series.ix[i][3] > series.ix[a][3]:    
        return True
    if s == 1:
        #Cannot negate the trend here as it could progress lower
        return False
    
    m1, c1, stdev_p1 = time_regression(series, a, i)   
    #If the series' standard deviation is more than param[5] x the first close
    #it is negated:
    if stdev_p1 > params[5]*series.ix[a][2]:
        return True
    if s == 2:
        #Negate the trend if the retracement is larger than params(2)
        if (series.ix[i][3] > (series.ix[b][4]+ params[2]*(-series.ix[b][4] + series.ix[a][3]))):
            return True
    if s == 3:
        #negate the setup here if the low is 2.5sd below the trend?
        if series.ix[i][4] < c1 + m1*t_ia-params[4]*stdev_p1:
            return True
        #Also negate the trend if B is not the lowest point between a and c
        ac_hi, ac_lo, ac_agv = max_min_mean(series[a:c])
        if(series.ix[b][4] > ac_lo):
            return True
    if s == 4:
        #negate the trend if the retracement is above the time regression
        if series.ix[i][3] > c1 + m1*t_ia + 2*stdev_p1:
            return True
    return False


def progress_trend(t, series, i, params):
    "Returns a trend that is the next possible progression of the input trend t"""
    
    """Inputs:
    t: the current trend
    series: the series to which t relates
    i: the current location within the series
    params=(5, 1, 0.8, 0.7, 2.05)- a tuple of parameters in order:
    a: the parameter for determining the low
    sigma_ab: parameter for determining a high after the first low(vol based)
    sigma_bc: parameter for determining a low after high based on vol
    ret_bc: parameter for excluding a low based on retracement.
    abs_stdev: negate any trend more than this #st deviations from trend
    %stdev: negate any trend who's stdev is > this % of a's close
    """
    
    #
    s = stage_of_trend(t)
    

    j = -1
    a, b, c, d, e= t
    if a == i:
        return t
    
    m1, c1, stdev_p = time_regression(series, a, i)
    m2, c2, stdev2 = time_regression(series, a-datetime.timedelta(params[0]), i)
    #This Function assumes the trend cannot be negated:
    if s==1:
        mx, mn, avg = max_min_mean(series[a-datetime.timedelta(params[0]):i])
        mx_tr, mn_tr, avg_tr, stdev_tr = \
            true_range_stats(series[a-datetime.timedelta(params[0]):i])
        if((series.ix[i][3] > series.ix[a][4] + mx_tr) and series.ix[i][3]>=mx):
            return (a, i, 0, 0, 0)
        else:
            return t
        # A L T E R N A T I V E  S P E C I F I C A T I O N
        #Check to see if b>1*sd from "low a"
        #if(series[i][3] - series[a][4] > params[1]*stdev2):
        #    return (a, i, 0, 0, 0)
        #else:
        #    return t
    if s==2:
        #check to see if z_score(i) > z_score(b) -  if so make this the new high if it satisfies
        #the condition for b
        if(z_score(series, 3, a, i, m1, c1, stdev_p) > \
            z_score(series, 3, a, b, m1, c1, stdev_p)):
            #if(series[i][3] - series[a][4] > params[1]*stdev2):
            return (a, i, 0, 0, 0)
        
        #Check whether the low satisfies the retracement condition (use close)
        if(series.ix[b][2] - series.ix[i][4] > params[2]*stdev2):
            return(a, b, i, 0, 0)
        else:
            return t
    if s==3:
        #Check to see if z_score(i) < z_score(c)
        if(z_score(series, 4, a, i, m1, c1, stdev_p) < \
            z_score(series, 4, a, c, m1, c1, stdev_p)):
            return (a, b, i, 0, 0)
        #Check to see if this high satisfies the gradient comparison 
        #Must also be above b.
        if((series.ix[i][3] - series.ix[b][3])/(len(series[b:i])-1)> params[3] * \
          ((series.ix[c][4] - series.ix[a][4])/(len(series[a:c])-1))):
            if(series.ix[i][3]>series.ix[b][3]):
                return(a, b, c, i, 0)
        else:
            return t
    if s==4:
        #Check to see if z-score(i)>z_score(d)
        if(z_score(series, 3, a, i, m1, c1, stdev_p) > \
            z_score(series, 3, a, d, m1, c1, stdev_p)):
            return (a, b, c, i, 0)
        #Check to see if this point satisfies the retracement parameter
        if(series.ix[i][4] < c1 + m1*(len(series[a:i])-1) - params[1]*stdev_p):
            return (a, b, c, d, i)
        else:
            return t

def progress_down_trend(t, series, i, params):
    """Returns a trend that is the next possible progression of the input trend t"""
    
    """Inputs:
    t: the current trend
    series: the series to which t relates
    i: the current location within the series
    params=(5, 1, 0.8, 0.7, 2.05)- a tuple of parameters in order:
    a: the parameter for determining the low
    sigma_ab: parameter for determining a high after the first low(vol based)
    sigma_bc: parameter for determining a low after high based on vol
    ret_bc: parameter for excluding a low based on retracement.
    abs_stdev: negate any trend more than this #st deviations from trend
    %stdev: negate any trend who's stdev is > this % of a's close
    """
    
    #
    s = stage_of_trend(t)
    

    j = -1
    a, b, c, d, e= t
    if a == i:
        return t
    m1, c1, stdev_p = time_regression(series, a, i)
    m2, c2, stdev2 = time_regression(series, a-datetime.timedelta(params[0]), i)
    #This Function assumes the trend cannot be negated:
    if s==1:
        mx, mn, avg = max_min_mean(series[a-datetime.timedelta(params[0]):i])
        mx_tr, mn_tr, avg_tr, stdev_tr = \
            true_range_stats(series[a-datetime.timedelta(params[0]):i])
        if((series.ix[i][4] < series.ix[a][4] - mx_tr) and series.ix[i][4]<=mn):
            return (a, i, 0, 0, 0)
        else:
            return t
        # A L T E R N A T I V E  S P E C I F I C A T I O N
        #Check to see if b>1*sd from "low a"
        #if(series[i][3] - series[a][4] > params[1]*stdev2):
        #    return (a, i, 0, 0, 0)
        #else:
        #    return t
    if s==2:
        #check to see if z_score(i) > z_score(b) -  if so make this the new high if it satisfies
        #the condition for b
        if(z_score(series, 4, a, i, m1, c1, stdev_p) < \
            z_score(series, 4, a, b, m1, c1, stdev_p)):
            #if(series[i][3] - series[a][4] > params[1]*stdev2):
            return (a, i, 0, 0, 0)
        
        #Check whether the low satisfies the retracement condition (use close)
        if(series.ix[i][3]-series.ix[b][2] > params[2]*stdev2):
            return(a, b, i, 0, 0)
        else:
            return t
    if s==3:
        #Check to see if z_score(i) < z_score(c)
        if(z_score(series, 3, a, i, m1, c1, stdev_p) > \
            z_score(series, 3, a, c, m1, c1, stdev_p)):
            return (a, b, i, 0, 0)
        #Check to see if this high satisfies the gradient comparison 
        #Must also be above b.
        if((series.ix[i][4] - series.ix[b][4])/(len(series[b:i])-1)< params[3] * \
          ((series.ix[c][3]-series.ix[a][3])/(len(series[a:c])-1))):
            if(series.ix[i][4]<series.ix[b][4]):
                return(a, b, c, i, 0)
            else:
                return t
        else:
            return t
    if s==4:
        #Check to see if z-score(i)>z_score(d)
        if(z_score(series, 4, a, i, m1, c1, stdev_p) < \
            z_score(series, 4, a, d, m1, c1, stdev_p)):
            return (a, b, c, i, 0)
        #Check to see if this point satisfies the retracement parameter
        if(series.ix[i][3] > c1 + m1*(len(series[a:i])-1) + params[1]*stdev_p):
            return (a, b, c, d, i)
        else:
            return t

            
def time_regression(series, a, i):
    """Returns a tuple (m, c, v) of slope, intercept and variance vs time"""
    
    """
    Requires: "from scipy import stats"
    Takes as input:    
    series: a timeseries 
    a: the start index
    i the final index.
    Outputs: gradient, intercept, standard deviation of_the_population"""
    x = list(xrange(len(series[a:i])))
    y = series.ix[a:i][2].values  # closing values
    gradient, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    error = []
    i=0
    for oobservation in y:
        error.append(intercept + gradient*(i)-y[i])
        i = i+1
    standard_deviation_p = np.std(error, dtype=np.float64, ddof=1)
    return (gradient, intercept, standard_deviation_p)

    #LOW_A_IS_SATISFIED: returns boolean for whether this counts as start of trend

def series_date(series, date, d):
    """Returns the date in the series d days away from date"""
    return series.shift(-d).ix[date][0]

def low_a_is_satisfied(series, i, params):
    """Function to determine if a new low has been satisfied."""

    """inputs:
    series: timeseries of data in format  (date, open, close, high, low, volume)
    params: the parameters:
    param a: the parameter for determining the low
    param sigma_ab: parameter for determining a high after the first low(vol based)
    param sigma_bc: parameter for determining a low after high based on vol
    param ret_bc: parameter for determing a low based on retracement."""

    #first determine if this satisfies a low:
    #it is lower than the previous 5 observations
    j = 1
    
       
    while j<params[0]:
        
       
        
        if series.ix[series_date(series, i, -j)][4]<series.ix[i][4]:
            return False
        j=j+1
        
    return True

def high_a_is_satisfied(series, i, params):
    """Function to determine if a new low has been satisfied."""

    """inputs:
    series: timeseries of data in format  (date, open, close, high, low, volume)
    params: the parameters:
    param a: the parameter for determining the low
    param sigma_ab: parameter for determining a high after the first low(vol based)
    param sigma_bc: parameter for determining a low after high based on vol
    param ret_bc: parameter for determing a low based on retracement."""

    #first determine if this satisfies a low:
    #it is lower than the previous 5 observations
    j = 1
    while j<params[0]:
        if series.ix[series_date(series, i, -j)][3]>series.ix[i][3]:
            return False
        j=j+1
    return True
#up_trend_finder: finds established uptrends and returns the locations
#(A, B, C, D, E) of each stage of the setup in the series.
def incremental_up_trend_finder(series, livetrends, last_update, params=(5, 1.75, 0.8, 0.7, 2.05, 0.2),ticker = 'TY'):
    """function to identify established up trends"""

    """Takes inputs:
    timeseries: a list of tuples - (date, open, close, high, low, volume)
    a tuple of parameters in order:
    a: the parameter for determining the low
    sigma_ab: parameter for determining a high after the first low(vol based)
    sigma_bc: parameter for determining a low after high based on vol
    ret_bc: parameter for determing a low based on retracement.
    abs_stdev: negate any trend more than this #st deviations from trend
    %stdev: negate any trend who's stdev is >= this ammount of a's close
    And gives as output:
    a list of n 5-tuples of each completed setup stage A-E for each of the n 
    completed setups in the series."""
    
    #i is the index of where we are in the series
    i = -1
    
    #n_livetrends is the number of incomplete trends "alive"
    n_livetrends = 0
    #the livetrends are the trends in setup 
    
    #n_setups is the number of complete setups
    bn_setups = 0
    #setups will store the setups
    
    setups = []    
    if last_update==None:
        last_update = series_date(series, min(series.index), params[0]+5)
    
    for i in series.ix[last_update:].index:
        
        #check to see if we need to add a new trend
        if low_a_is_satisfied(series, i, params):
            t = Trend(series, "Long", (i, 0, 0 ,0, 0), ticker ,"incomplete")
            livetrends.append(t)
            n_livetrends = len(livetrends)
        #for each live trend check to see if the live trends are negated
        j=-1
        new_livetrends = []
        for tnd in livetrends:
            t = tnd.trnd
            if(not(trend_negated(t, series, i, params))):
                new_livetrends.append(tnd)
        livetrends = new_livetrends
        new_livetrends=[]
        for tnd in livetrends:
            t = tnd.trnd
            #now advance any remaining trends if necessary
            new_t = progress_trend(t, series, i, params) 
            new_tnd = Trend(series, "Long", t, ticker ,"incomplete")
            if new_t[4] != 0:
                #setup complete: check that it passes quality controll then
                #add it to setups and tidy up
                t = Trend(series, "Long", new_t, ticker ,"inactive")
                t.log_quality(series)
                setups.append(t)
                n_setups = len(setups)
            else:
                #Setup is not complete
                #If the new trend is different add it
                if new_t != t:
                    if not(new_tnd in new_livetrends):
                        new_livetrends.append(Trend(series, "Long", new_t, ticker ,"incomplete"))
                        #Only keep the original trend if its a "low a"
                        #We keep all "low a's" because a new trend could emerge.
                #else the trend has not changed so keep it for the next time
                else:
                    if not(tnd in new_livetrends):
                        new_livetrends.append(tnd)
                #Regardless check whether t was an a and add it if so
                if stage_of_trend(t)==1 and not(tnd in new_livetrends):
                    new_livetrends.append(tnd)
        #end of livetrends loop
        livetrends = new_livetrends
        
    #end of series loop
    return (setups, livetrends)
#down_trend_finder: finds established downtrends and returns the locations
#(A, B, C, D, E) of each stage of the setup in the series.
def incremental_down_trend_finder(series,livetrends, last_update, params=(5, 1.75, 0.8, 0.7, 2.05, 0.2),ticker = 'TY'):
    """function to identify established up trends"""

    """Takes inputs:
    timeseries: a list of tuples - (date, open, close, high, low, volume)
    a tuple of parameters in order:
    a: the parameter for determining the low
    sigma_ab: parameter for determining a high after the first low(vol based)
    sigma_bc: parameter for determining a low after high based on vol
    ret_bc: parameter for determing a low based on retracement.
    abs_stdev: negate any trend more than this #st deviations from trend
    %stdev: negate any trend who's stdev is >= this ammount of a's close
    And gives as output:
    a list of n 5-tuples of each completed setup stage A-E for each of the n 
    completed setups in the series."""
    
    #i is the index of where we are in the series
    i = -1
    
    #n_livetrends is the number of incomplete trends "alive"
    n_livetrends = 0
    #the livetrends are the trends in setup - there can be at most 3 at a time
    
    #n_setups is the number of complete setups
    n_setups = 0
    #setups will store the setups
    setups = []    
    if last_update==None:
        last_update = series_date(series, min(series.index), params[0]+5)
    
    for i in series.ix[last_update:].index:
        #check to see if we need to add a new trend
        
        if high_a_is_satisfied(series, i, params):
            livetrends.append(Trend(series, "Short", (i, 0, 0, 0, 0), ticker ,"inactive"))
            n_livetrends = len(livetrends)
        #for each live trend check to see if the live trends are negated
        j=-1
        new_livetrends = []
        for tnd in livetrends:
            t = tnd.trnd
            if(not(down_trend_negated(t, series, i, params))):
                new_livetrends.append(tnd)
        livetrends = new_livetrends
        new_livetrends=[]
        for tnd in livetrends:
            t = tnd.trnd
            #now advance any remaining trends if necessary
            new_t = progress_down_trend(t, series, i, params) 
            new_tnd =Trend(series, "Short", new_t, ticker ,"inactive")
            if new_t[4] != 0:
                #setup complete: check that it passes quality controll then
                #add it to setups and tidy up
                t = Trend(series, "Short", new_t, ticker ,"inactive")
                t.log_quality(series)
                setups.append(t)
                n_setups = len(setups)
            else:
                #Setup is not complete
                #If the new trend is different add it
                if new_t != t:
                    if not(new_tnd in new_livetrends):
                        new_livetrends.append(Trend(series, "Short", new_t, ticker ,"incomplete"))
                        #Only keep the original trend if its a "low a"
                        #We keep all "low a's" because a new trend could emerge.
                #else the trend has not changed so keep it for the next time
                else:
                    if not(tnd in new_livetrends):
                        new_livetrends.append(tnd)
                #Regardless check whether t was an a and add it if so
                if stage_of_trend(t)==1 and not(tnd in new_livetrends):
                    new_livetrends.append(tnd)
        #end of livetrends loop
        livetrends = new_livetrends
        
    #end of series loop
    return (setups, livetrends)


def true_range_stats(series):
    """returns the max, min, averavge and st_dev of the daily range"""
    
    """this will not provide the true range of the first item in series as 
    there is no previous close"""
    
    if len(series)==1:
        return (0, 0, 0, 0)
    true_range = []
    skipfirst = True
    
    for i in series.index:
        if skipfirst:
            skipfirst = False
            j= i
            continue
        
        true_range.append(max(series.ix[i][3]-series.ix[j][2], \
            series.ix[j][2]-series.ix[i][4], \
            series.ix[i][3]-series.ix[i][4]))    
            #high - prev close
            #prev close - low
            #high-low  
        j = i
            
    if(len(series)==2):
        print "Unable to calculate standard deviation for single observation"
        st_devp = 0
    else:
        st_devp = np.std(true_range, dtype=np.float64, ddof=1)
        
    return (max(true_range), min(true_range), np.mean(true_range), st_devp)

def max_min_mean(series):
    """returns the max, min and mean of series"""
    mx = series[3].max()
    mn = series[4].min()
    avg = series.mean()[0:4]
    return (mx, mn, avg)
        
def z_score(series, t, a, i, m, c, stdev):
    """the z-score of i starting at a with parameters: m, c for t in series"""
    res = (series.ix[i][t] - (c + m * len(series[a:i])-1 ))/(stdev)
    return res
    
class Trend:
    """Class to hold and manipulate trends as tuple: (a, b, c, d, e)"""
    
    """A trend can have four status types:
    active - a trend is currently under assessment
    live - a trend has been traded
    negated - an active but not live trend is negated by price action
    stopped - a Live trade is stopped out
    inactive - The trend is inactive"""
    
    trend_count = 0
    
    def __init__(self, series, type, in_trend=(0, 0, 0, 0, 0), in_ticker = "", \
        in_status = "inactive"):
        
        self.a, self.b, self.c, self.d, self.e = in_trend
        self.trnd = in_trend
        self.status = in_status
        Trend.trend_count += 1
        self.ticker = in_ticker
        self.start = series.ix[self.a][0]
        if in_trend[4] != 0:
            self.end = series.ix[self.e][0]
        else:
            self.end = 0
        self.trend_type = type
        self.risk_factor = 1.0
        if self.trend_type=="Long":
            self.sgn = 1
        else:
            self.sgn = -1
        
        
        
    def get_trend_count(self):
        return self.trend_count
    
    def activate(self):
        self.status = "active"
    def is_active(self):
        if self.status == "active":
            return True
        else:
            return False
        
    def stopped(self):
        self.status = "stopped"
    def is_stopped(self):
        if self.status == "stopped":
            return True
        else:
            return False
        
    def deactivate(self):
        self.status = "deactiveated"
    def is_deactivated(self):
        if self.status == "deactiveated":
            return True
        else:
            return False
    
    def make_live(self):
        self.status = "live"
    def is_live(self):
        if self.status == "live":
            return True
        else:
            return False
        
    def negate(self):
        self.status = "negated"
    def is_negated(self):
        if self.status == "negated":
            return True
        else:
            return False
    def is_complete(self):
        if self.status != "incomplete":
            return True
        else:
            return False
    def get_status(self):
        return self.status
    
    def get_negate_level(self, series, i, trade_params):
        """method to determine a level below which the trend is negated"""
        
        """trade params: tuple -
        stop: - Stop level as z-score from trend average
        buy_level - as z-score from e in the trend
        limit_level - as z-score from buy level
        negate_level: z-score below which the trend is negated. 
        """
        m1, c1, stdev_p1 = time_regression(series, self.a, self.e)
        if self.trend_type == "Long":
            return c1 + m1 * (len(series[self.a:i])-1) - stdev_p1*trade_params[3]
        else:
            return c1 + m1 * (len(series[self.a:i])-1) + stdev_p1*trade_params[3]
        
    def get_stop_level(self, series, i, trade_params):
        """method to determine a level below which a live trend is negated"""
        m1, c1, stdev_p1 = time_regression(series, self.a, self.e)
        if self.trend_type == "Long":
            return c1 + m1 * (len(series[self.a:i])-1)- stdev_p1*trade_params[0]
        else:
            return c1 + m1 *(len(series[self.a:i])-1) + stdev_p1*trade_params[0]
            
    def get_execution_level(self, series, i, trade_params):
        """method to determine a level where a trade is entered"""
        m1, c1, stdev_p1 = time_regression(series, self.a, self.e)
        if self.trend_type == "Long":
            return max(c1 + m1 * (len(series[self.a:i])-1) - stdev_p1*trade_params[1],\
                series.ix[self.e][2]+(2-trade_params[1])*stdev_p1)
        else:
            return min(c1 + m1 * (len(series[self.a:i])-1) + stdev_p1*trade_params[1],\
                series.ix[self.e][2]-(2-trade_params[1])*stdev_p1)
    def get_limit_level(self, series, i, trade_params):
        """method to determine a level below which a live trend is negated"""
        
        #new limit: 1sd above the stop:
        return self.get_stop_level(series, i, trade_params) + self.stdev_p_reg
        
        #old limit:    
        #m1, c1, stdev_p1 = time_regression(series, self.a, self.e)
        #if self.trend_type == "Long":
        #    return c1 + m1 * (len(series[self.a:i])-1) - stdev_p1*trade_params[2]
        #else:
        #    return c1 + m1 * (len(series[self.a:i])-1) + stdev_p1*trade_params[2]
    def log_quality(self, series):
        """method to determine whether a trend is of sufficient quality"""
        #FOr QC we require that the trend's points a, b, c, d all be over 1.8
        #SD from trend
        self.m_reg, self.c_reg, self.stdev_p_reg = time_regression(series, self.a, self.e)
        if self.trend_type=='Long':
            ac = 4
            bd=3
            ac_demark = "Low"
            bd_demark = "High"
        else:
            ac=3
            bd=4
            ac_demark = "High"
            bd_demark = "Low"
        #def z_score      (series, t,       a,     i,          m,          c,       stdev):    
        self.za = z_score(series, ac, self.a, self.a, self.m_reg, self.c_reg, self.stdev_p_reg)
        self.zc = z_score(series, ac, self.a, self.c, self.m_reg, self.c_reg, self.stdev_p_reg)
        self.zb = z_score(series, bd, self.a, self.b, self.m_reg, self.c_reg, self.stdev_p_reg)
        self.zd = z_score(series, bd, self.a, self.d, self.m_reg, self.c_reg, self.stdev_p_reg)
        self.ze = z_score(series, ac, self.a, self.e, self.m_reg, self.c_reg, self.stdev_p_reg)
        #now log the inner z-scores (the reaction function of each of the extremes
        self.za_inner = z_score(series, bd, self.a, self.a, self.m_reg, self.c_reg, self.stdev_p_reg)
        self.zc_inner = z_score(series, bd, self.a, self.c, self.m_reg, self.c_reg, self.stdev_p_reg)
        self.zb_inner = z_score(series, ac, self.a, self.b, self.m_reg, self.c_reg, self.stdev_p_reg)
        self.zd_inner = z_score(series, ac, self.a, self.d, self.m_reg, self.c_reg, self.stdev_p_reg)
        self.ze_inner = z_score(series, bd, self.a, self.e, self.m_reg, self.c_reg, self.stdev_p_reg)
        
        #Now log the "tom demark" demand/resistance factor for a, b, c, d 
        #self.td_a = self.log_demark_quality(series, self.a, ac_demark)
        #self.td_c = self.log_demark_quality(series, self.c, ac_demark)
        #self.td_b = self.log_demark_quality(series, self.b, bd_demark)
        #self.td_d = self.log_demark_quality(series, self.d, bd_demark)
        #self.td_e = self.log_demark_quality(series, self.e, ac_demark)
    
    def log_demark_quality(self, series, a, type="Low"):
        """THis method returns the tom-demark support/resistance strength"""
        """series is the timeseries, a is the point we are interested in"""
        
        #This is defined as the min of the number of time steps either side 
        #before a lower(higher) low(high) is observed 
        pass
        if(type=="Low"):
            is_low = True
        else:
            is_low = False
        for j in range(1,30):
            i2_pls = series_date(series, a, j)
            i2_mns = series_date(series, a, -j)
            i = 0
            if(not(np.isnan(i2_mns))):
                if(is_low):
                    if(series.ix[i2_mns][4]<series.ix[a][4]):
                        return j
                else:
                    if(series.ix[i2_mns][3]>series.ix[a][3]):
                        return j
            if(not(np.isnan(i2_pls))):
                if(is_low):
                    if(series.ix[i2_pls][4]<series.ix[a][4]):
                        return j
                else:
                    if(series.ix[i2_pls][3]>series.ix[a][3]):
                        return j
        return 30 #search capped at 30 days (massive support and a v.long trend)            
class Trade:
    num_trades = -1
    def __init__(self, in_trend, in_execution_info = [(0,0), (0,0), (0,0)], \
            in_errors =[0,0] ):
        self.trend = in_trend
        self.execution_info = in_execution_info
        self.errors = in_errors
        self.trade_ID = self.next_ID()
        self.orig_ID = self.trade_ID #for grandparenting amended trades!
        self.order_level = 0.0
        self.stop_level = 0.0
        self.limit_level = 0.0
        self.negate_level = 0.0
        self.trade_size = 1
        self.pl_hist=[]
        self.risk_hist=[]
            
        
    def next_ID(self):
        Trade.num_trades += 1
        return Trade.num_trades
        
    
    def update_execution_info(self, i, in_execution_info):
        self.execution_info[i] = in_execution_info
        
    def update_errors(self, in_errors):
        self.errors = in_errors
        
    def plot(self):
        s = self.trend.start
        e = self.execution_info[2][0]
        series = gd.get_data_yahoo_orig(s, e, fut_code=self.trend.ticker, format='candle')
        if series!=None and len(series)!=0:
            gpc.gpcs(series)    
        
    def plot_pending(self, date):
        s = self.trend.start + datetime.timedelta(-7)
        series = gd.get_data_yahoo_orig(s, date, fut_code=self.trend.ticker, format='candle')
        if series!=None and len(series)!=0:
            gpc.gpcs_trade(series, self)
        
    def filter(self, i):
        if (self.trend.e - self.trend.a).days >750:
            self.trend.deactivate()
            return
        if (self.trend.e - self.trend.a).days <30:
            self.trend.deactivate()
            return
        
    def set_levels(self, series, i, trade_params):
        """sets the trade levels for a trade"""
        self.order_level = self.trend.get_execution_level(series, i, trade_params)
        self.stop_level = self.trend.get_stop_level(series, i, trade_params)
        self.negate_level = self.trend.get_negate_level(series, i, trade_params)
        self.limit_level = self.trend.get_limit_level(series, i, trade_params)
        
    def update(self, series, i, trade_params):
        """Function to update a trade - does nothing to livetrades list"""
        #Each trade has
        #order_level
        #limit_level
        #negate_level
        #stop_level
        #
        #newly activated trades -  have 0.0 for everything
        #live trades -  will have a negate level of 0.0
        #pending trades - will have all 4 levels
        
        #sanity check:
        
        if i <self.trend.e:
            print "cannot update trend - error!\n i:"
            print i
            print "e:"
            print self.trend.e
        #first get the index of the highs(lows) and lows(highs) for this type of trend:
        if self.trend.trend_type == "Long":
            pve = 3 # positive
            nve = 4 # negative
        else:
            pve = 4
            nve = 3
        #check if it is a new trade:
        if self.order_level == 0:
            self.set_levels(series, i, trade_params)
        else:
            if self.trend.is_active():
            #it was a pending trade - check to see if it was negated
                if (self.negate_level - series.ix[i][nve])*self.trend.sgn > 0:
                    self.trend.negate()
                    
            #check if it has been traded:
                if (series.ix[i][pve] - self.order_level)*self.trend.sgn >0 and ( self.limit_level - series.ix[i][nve])*self.trend.sgn >0 :
                    #trade was executed:
                    self.set_levels(series, i, trade_params)
                    self.negate_level = 0.0
                    self.trend.make_live()
                    
                    #update the execution_info
                    if self.trend.sgn == 1:
                        lvl = min(self.order_level, min(series.ix[i][1], self.limit_level))
                    else:
                        lvl = max(self.order_level, max(series.ix[i][1], self.limit_level))
                    self.update_execution_info(0, (i,lvl)) 
                    
                    #update the history for the newly live trade:
                    self.update_pl_hist(series, i)
                else:
                    #it is a pending trade that was not traded
                    self.set_levels(series, i, trade_params)
            else:
                #it is a live trade -  check to see if it was stopped:
                if (series.ix[i][nve] - self.stop_level)*self.trend.sgn<0:
                    #trade was stopped
                    self.trend.stopped()
                    if self.trend.sgn == 1:
                        lvl = min(self.stop_level, max(series.ix[i][1],series.ix[i][nve]))
                    else:
                        lvl =max(self.stop_level, min(series.ix[i][1],series.ix[i][nve]))
                    self.update_execution_info(2, (i,lvl))

                else:
                    self.set_levels(series, i, trade_params)
                    self.negate_level = 0.0
                #update the history for the live trades    
                self.update_pl_hist(series, i)
    
    def update_pl_hist(self, series, i):
        """adds the cumulative pl to the pl_hist for date i"""
        if self.trend.is_live():
            px = series.ix[i][2] #the close px
        else:
            px = self.execution_info[2][1] # the stop px
        
        
        if(len(self.pl_hist)==0):
            self.pl_hist = Series({i:\
                self.trade_size*self.trend.sgn*(px - self.execution_info[0][1])/self.avg_tr})
        else:
            if i in self.pl_hist.index:
                self.pl_hist[i] = self.trade_size*self.trend.sgn*(px - self.execution_info[0][1])/self.avg_tr
            else:    
                self.pl_hist = self.pl_hist.append(Series({i:\
                self.trade_size*self.trend.sgn*(px - self.execution_info[0][1])/self.avg_tr}))
        
        self.update_risk_hist(series, i)

    
    def update_risk_hist(self, series, i):
        """adds the open pl to the risk_hist for date i"""
        if self.trend.is_live():
            stop_pl = self.trade_size * self.trend.sgn*(self.stop_level - self.execution_info[0][1])/self.avg_tr
            if(len(self.risk_hist)==0):
                self.risk_hist = Series({i: self.pl_hist[i]-stop_pl })
            else:
                if i in self.risk_hist.index:
                    self.risk_hist[i] =self.pl_hist[i]-stop_pl
                else:
                    self.risk_hist = self.risk_hist.append(Series({i: self.pl_hist[i]-stop_pl }))
    
    def close_trade(self, cls, i):
        """clsoes a trade at the closing level"""
        self.trend.stopped()
        self.update_execution_info(2, (i, cls))
        self.update_pl_hist([], i)
        
class Trade_Manager:
    """Class to manage a list of trades and output executed trades and P&L"""
    
    """trade list is a list of trade objects
    
    money_Params is a tuple of length 3:
    1 the % of capital invested per position
    2 the maximum number of long only positions
    3 the maximum number of long positions if there are sufficient shorts"""
    
    
    def __init__(self, in_trade_list=[], money_params=(2.0, 20, 30),\
            find_params=(5, 1.75, 0.8, 0.7, 2.05,0.2),\
            trade_params=(2.1, 1.5, 1, 2.25)):
        self.params = money_params
        self.finder_params = find_params
        self.trader_params = trade_params
        self.trade_list = in_trade_list
        
        
        #money management variables:
        self.portfolio_hist = []
        self.portfolio_size = 100
        self.trade_size = 1.0
        
        
        #these are new variables for incremental trading....
        self.historic_trades = []
        self.live_trades = []
        self.pending_trades =[]
        self.incomplete_setups = []
        self.last_update = None
        self.ticker_updates = None
                
        
    
    def get_at_risk(self, i):
        """returns a list of trades that a have more than 7% at risk"""
        risky_business = []
        for t in self.live_trades:
            if self.trade_at_risk(t, i):
                risky_business.append(t)
        return risky_business
    
    def trade_at_risk(self, t, i):
        """returns true/false if trade i has 7%at risk for day i"""
        if t.risk_hist[i]>7.0:
            return True
        return False
        #TO-DO: THIS SHOULD BE 7% of original assets: we want to make sure all trades are correctly scaled!
        
        
    def incremental_trend_trader(self, series, trade_params):
        #self.historic_trades - put stopped trades here
        #self.live_trades 
        #self.pending_trades
        #self.new_setups
        #self.incomplete_setups - not relevant for this method
        
        
        pending_tickers=[]
        if self.last_update == None:
            indy = series.index
        else:
            print self.last_update.isoformat()      
            st = series_date(series, self.last_update, 1)
            print st  
            if not(type(st) == datetime.datetime):
                print "up-to-date"
                indy = []
            else:
                indy = series[st:].index
            
        for i in indy:    
            
            #for each day in the latest price history:
            
            
            #update live trades
            tmp_livetrades = []
            for t in self.live_trades:
                t.update(series, i, self.trader_params)
                if t.trend.is_stopped():
                    self.historic_trades.append(t)
                else:
                    tmp_livetrades.append(t)
            self.live_trades = tmp_livetrades
            
            
            
            #modify pending list:
            #remove any negated trades from the pending list
            #check whether trades that would have been traded can be used to modify a live trade's stop
            #or whether they are ok to be made live themselves:
            
            at_risk = self.get_at_risk(i)#a list of live trades with too much open PL
            new_pending_trades=[] # this will hold the pending trades we wish to keep
            for t in self.pending_trades:
                t.filter(i)
                #this deactivates avtive trades:
                #if the trade is no longer active forget it and move on

                #print t.trend.get_status()
                if not(t.trend.status in ['active']):
                    continue 
                    
                #else: update the pending trade and if it is still pending, keep it    
                t.update(series, i, self.trader_params)
                if t.trend.is_active():
                    t.trade_size = self.trade_size
                    new_pending_trades.append(t)
                else:
                    #Otherwise, if the trade is live, see if it replaces a previous trade
                    if t.trend.is_live():
                        
                        
                        for r in at_risk:
                            print "risky frisky"
                            if t.trend.ticker == r.trend.ticker:
                                #if the new slope is not (abs) steeper:
                                if(r.trend.m_reg*r.trend.sgn > t.trend.m_reg* t.trend.sgn):
                                    t.trend.deactivate()
                                    continue
                                #check to see if the old and new trades are both the same type
                                if r.trend.trend_type != t.trend.trend_type:
                                    t.trend.deactivate()
                                    continue
                                #check that the new trade does not start on the same day:
                                if r.execution_info[0][0] == t.execution_info[0][0]:
                                    t.trend.deactivate()
                                    continue
                                #check that the new trade has less risk than the old trade:
                                if r.risk_hist[i]<t.risk_hist[i]:
                                    t.trend.deactivate()
                                    continue
                                #check that the trade is not at risk itself!
                                if self.trade_at_risk(t, i):
                                    t.trend.deactivate()
                                    continue
                                #if we are here then t is a legitimate replacement trend for the at risk trade!
                                
                                #note the parent trade:
                                t.orig_ID = r.orig_ID 
                                t.trend.risk_factor = t.trend.sgn*( t.execution_info[0][1] - t.stop_level)
                                #close the trade and add it to historical
                                r.close_trade(t.execution_info[0][1], i)
                                self.historic_trades.append(r)
                                
                                #replace livetrades with the new trade:
                                j=-1
                                for lt in self.live_trades:
                                    j += 1
                                    if lt.trend.ticker == r.trend.ticker:
                                        livetrades_idx = j
                                self.live_trades[livetrades_idx] = t
                        print self.ticker_is_live(t.trend.ticker)
                        if not self.ticker_is_live(t.trend.ticker):
                            #this ticker is not currently live so add it to livetrades
                            t.trend.risk_factor = t.trend.sgn*( t.execution_info[0][1] - t.stop_level)
                            self.live_trades.append(t)
                            print "trade made live"
                            print len(self.live_trades)
                        else:
                            #doing nothing essentially removes the trade from the list 
                            pass
                            
            
            #End of processing currently pending trades:
            
            
            
            #activate self.new_setups TRENDS if the current day == e, MAKE TEHM RADES, add them to pending, and remove them from self.new_setups
            tmp_new_complete = []
            for trend in self.new_setups:
                # if this trend was completed today make it a TRADE!
                if i == trend.e: 
                    trend.activate()
                    trade_errors = [0, 0]
                    current_result = [(0,0), (0,0), (0,0)]
                    trade = Trade(trend, current_result, trade_errors)
                    trade.mx_tr, trade.mn_tr, trade.avg_tr, trade.stdev_tr = \
                    true_range_stats(series[trade.trend.a:trade.trend.e])
                    trade.update(series, i, self.trader_params)
                    self.pending_trades.append(trade)
                else:
                    tmp_new_complete.append(trend)
            self.new_complete = tmp_new_complete
            self.pending_trades = []
            self.pending_trades = new_pending_trades
            print "num pending trades:"
            print len(self.pending_trades)
            print "num live trades:"
            print len(self.live_trades)
                    
        #end of for i in series
        
        
    
    
    def ticker_is_live(self, tk):
        """returns true if a trade with ticker tk is live"""
        for t in self.live_trades:
            if t.trend.ticker==tk:
                return True
        return False
    
    def num_active_type(self):
        """returns the number of active longs and shorts as a tuple"""
        long = 0
        short = 0
        for t in self.live_trades:
            if(t.trend.trend_type=="Long"):
                long = long+1
            else:
                short = short + 1
        
        return (long, short)
                    
        
                
    def load_file2(self, tk):
        """loads the historic files for ticker tk"""
        if os.path.exists(tk+"trade_data2.pkl"):
            input = open(tk+"trade_data2.pkl", 'rb')
            self.historic_trades, self.live_trades, self.pending_trades, self.incomplete_setups, self.last_update = \
                pickle.load(input)  
            #print "previous file located, last update:"
            #print self.last_update
            input.close()
        else:
            print "no previous file found"
            self.historic_trades = []
            self.live_trades = []
            self.pending_trades = []
            self.incomplete_setups = []
            self.last_update = None
    
    def offline_incremental_trader_null(self, ticker_list, start_date, end_date):
        """calculates P&L, running algorithm on ticker_list over the period"""
        
        if end_date == "today" or end_date=="now":
            end_date=datetime.datetime.now()
            
        self.portfolio_size = 100
        self.trade_size = 1
        #input = open("Portfolio_size.pkl", 'rb')
        #self.portfolio_hist, self.portfolio_size,self.trade_size = pickle.load(input) 
        #input.close()
        
        
        
        self.skipped_tickers=[]
        for tk in ticker_list:
            print tk
            if os.path.exists(tk+"trade_data2.pkl"):
                input = open(tk+"trade_data2.pkl", 'rb')
                self.historic_trades, self.live_trades, self.pending_trades, self.incomplete_setups, self.last_update = \
                    pickle.load(input)
                print "previous file located, last update:"
                print self.last_update
                input.close()
            else:
                print "no previous file found"
                self.historic_trades = []
                self.live_trades = []
                self.pending_trades = []
                self.incomplete_setups = []
                self.last_update = None
            
            run = False
            if self.last_update==None:
                run = True
            else:
                if self.last_update < end_date:
                    run = True
            if run:
                tmp_incomplete_lng_trends_this_ticker = []
                tmp_incomplete_sht_trends_this_ticker = []
                
                    
                series2 = gd.get_data_yahoo(start_date + datetime.timedelta(-self.finder_params[0]-4), end_date, tk, 'candle')
                if len(series2) == 0:
                    self.skipped_tickers.append(tk)
                    continue
                dt, o, c, h, l, v = zip(*series2)
                series = pd.DataFrame(data = {0:dt, 1:o, 2:c, 3:h, 4:l, 5:v}, index = dt )     
                #Skip tickers we can't get data for
                if len(series)==0:
                    self.skipped_tickers.append(tk)
                    continue
                
                    
                for t in self.incomplete_setups:
                    if t.trend_type=="Long":
                        tmp_incomplete_lng_trends_this_ticker.append(t)
                    else:
                        tmp_incomplete_sht_trends_this_ticker.append(t)
                
                complete_new, incomplete_new =\
                 incremental_up_trend_finder(series, tmp_incomplete_lng_trends_this_ticker, self.last_update, self.finder_params, tk)
                #pdb.set_trace()
                complete_new2, incomplete_new2 =\
                    incremental_down_trend_finder(series,tmp_incomplete_sht_trends_this_ticker, self.last_update, self.finder_params, tk)
                
                #save the complete trades for processing now
                self.new_setups = []
                self.new_setups.extend(complete_new)    
                self.new_setups.extend(complete_new2)
                #save the incomplete trends:
                self.incomplete_setups = incomplete_new
                self.incomplete_setups.extend(incomplete_new2)
                
                
                #pdb.set_trace()
                #process the new setups and the existing trades:
                self.incremental_trend_trader(series, self.trader_params)
                
                self.last_update = max(series.index)
                
                #pdb.set_trace()
                output = open(tk+"trade_data2.pkl", 'wb')
                pickle.dump([self.historic_trades, self.live_trades, self.pending_trades, self.incomplete_setups, self.last_update], output) 
                output.close()
            
            
        
        print "skipped tickers:"
        print list(self.skipped_tickers)

    def offline_incremental_primer(self, ticker_list, start_date, end_date):
        """Updates pending, complete + incomplete trades for portfolio manager"""
        #to do - once updated - needs trade_data.pkl to me changed to trade_data2.pkl
        #and for new_setups to be added to load statement
        if end_date == "today" or end_date=="now":
            end_date=datetime.datetime.now()
            
        
        self.skipped_tickers=[]
        for tk in ticker_list:
            print tk
            self.load_file2(tk)
            
            run = False
            if self.last_update==None:
                run = True
            else:
                if self.last_update < end_date:
                    run = True
            if run:
                tmp_incomplete_lng_trends_this_ticker = []
                tmp_incomplete_sht_trends_this_ticker = []
                
                    
                series2 = gd.get_data_yahoo(start_date + datetime.timedelta(-self.finder_params[0]-4), end_date, tk, 'candle')
                if len(series2) == 0:
                    self.skipped_tickers.append(tk)
                    continue
                dt, o, c, h, l, v = zip(*series2)
                series = pd.DataFrame(data = {0:dt, 1:o, 2:c, 3:h, 4:l, 5:v}, index = dt )     
                #Skip tickers we can't get data for
                if len(series)==0:
                    self.skipped_tickers.append(tk)
                    continue
                
                    
                for t in self.incomplete_setups:
                    if t.trend_type=="Long":
                        tmp_incomplete_lng_trends_this_ticker.append(t)
                    else:
                        tmp_incomplete_sht_trends_this_ticker.append(t)
                
                complete_new, incomplete_new =\
                 incremental_up_trend_finder(series, tmp_incomplete_lng_trends_this_ticker, self.last_update, self.finder_params, tk)
                #pdb.set_trace()
                complete_new2, incomplete_new2 =\
                    incremental_down_trend_finder(series,tmp_incomplete_sht_trends_this_ticker, self.last_update, self.finder_params, tk)
                
                #save the complete trades for processing now
                self.new_setups = []
                self.new_setups.extend(complete_new)    
                self.new_setups.extend(complete_new2)
                #save the incomplete trends:
                self.incomplete_setups = incomplete_new
                self.incomplete_setups.extend(incomplete_new2)
                
                # here we use the incremental trend trader to ensure any new_setuups are either pending or live (to be discarded)
                self.incremental_trend_trader(series, self.trader_params)
                self.last_update = max(series.index)
                print len(self.live_trades)
                output = open(tk+"trade_data2.pkl", 'wb')
                pickle.dump([[], [], self.pending_trades, self.incomplete_setups, self.last_update], output) 
                output.close()
            
        
        print "skipped tickers:"
        print list(self.skipped_tickers)
    

    def get_trade_data_pending(self):
        """returns a DataFrame of trade details"""
        traded=self.pending_trades
        tk = []
        start = []
        end = []
        a = []
        b = []
        c = []
        d = []
        e = []
        start_t = []
        start_p = []
        hlfp_t = []
        hlfp_p = []
        end_t = []
        end_p = []
        is_live = []
        i=-1
        pl=[]
        dir = []
        za = []
        zb=[]
        zc=[]
        zd=[]
        mx_tr = [] 
        mn_tr = []
        avg_tr = []
        stdev_tr=[]
        setup_stdev = []
        risk_factor=[]
        direction = []
        pct = []
        stop_lvl = []
        order_lvl = []
        limit_lvl = []
        negate_lvl = []
        for t in traded:
            if t.trend.trend_type == "Long":
                dir='Long'
            else:
                #use "-" because risk factor is -ve for short trades
                dir= 'Short'
            tk.append(t.trend.ticker)
            start.append(t.trend.start)
            end.append(t.trend.end)
            a.append(t.trend.a)
            b.append(t.trend.b)
            c.append(t.trend.c)
            d.append(t.trend.d)
            e.append(t.trend.e)
            za.append(t.trend.za)
            zb.append(t.trend.zb)
            zc.append(t.trend.zc)
            zd.append(t.trend.zd)
            mx_tr.append(t.mx_tr)
            mn_tr.append(t.mn_tr)
            avg_tr.append(t.avg_tr)
            stdev_tr.append(t.stdev_tr)
            direction.append(t.trend.trend_type)
            setup_stdev.append(t.trend.stdev_p_reg)
            risk_factor.append(t.trend.risk_factor)
            order_lvl.append(t.order_level)
            stop_lvl.append(t.stop_level)
            negate_lvl.append(t.stop_level)
            limit_lvl.append(t.limit_level)
        trade_data = pd.DataFrame({'ticker':tk,\
                                        'start_date':start,\
                                        'end_date':end,\
                                        'a':a,
                                        'b':b,\
                                        'c':c,\
                                        'd':d,\
                                        'e':e,\
                                        'dir':direction,\
                                        'za':za,\
                                        'zb':zb,\
                                        'zc':zc,\
                                        'zd':zd,\
                                        'mx_tr':mx_tr,\
                                        'mn_tr':mn_tr,\
                                        'avg_tr':avg_tr,\
                                        'stdev_tr':stdev_tr,\
                                        'setup_stdev': setup_stdev,\
                                        'risk_factor': risk_factor,\
                                        'order_lvl':order_lvl,\
                                        'limit_lvl':limit_lvl,\
                                        'stop_lvl':stop_lvl,\
                                        'negate_lvl':negate_lvl})
        if(len(trade_data.index)!=0):
            trade_data['is_long'] = (trade_data['dir']=='Long')
            trade_data['a_e'] = (trade_data['e'] - trade_data['a'])
        
        return trade_data
        
            
class Portfolio_Manager:
    """Class to manage the overall portfolio position"""
    def __init__(self):
        #This will store a dataframe of the update status for the tickers:
        #the earliest_date of a setup and the last_update date
        self.ticker_updates = None
        #This will be a list of the live trades:
        self.live_trades = []
        #this will be a list of historic trades:
        self.historic_trades = []
        #This will store the value of the portfolio for scaling purposes
        self.portfolio_size = 0
        #THis is used to determine when to re-size positions as the portfolio changes
        self.portfolio_scale = 0
        if os.path.exists("portfolio_status.pkl"):
            input = open("portfolio_status.pkl", 'rb')
            self.ticker_updates, self.live_trades,  self.portfolio_size, self.portfolio_scale = pickle.load(input)
            input.close()
        #store the skipped tickers:
        self.skipped_tickers=[]
            
    
    def update_single_security_files(self, end_date=None, execute = False):
        """This increments the files storing setups for each security"""
        #if the end_date is not set make it one day later:
        if end_date == None:
            end_date = max(self.ticker_updates.last_update).value + datetime.timedelta(1)
        #if end_date is "latest" just update all the files so that they are 
        #up to date with the most up-to-date file
        if end_date == "Latest" or end_date == "latest":
            end_date = max(self.ticker_updates.last_update).value
        #for each ticker update the file:
        self.skipped_tickers = []
        for tk in self.ticker_updates.index:
            if(self.ticker_updates.last_update.ix[tk]<=end_date):
                start_date = self.ticker_updates.ix[tk]['earliest_date']
                tm = Trade_Manager()
                #overwrite the live trades
                tm.live_trades.extend(self.live_trades)
                tm.last_update = self.ticker_updates.ix[tk]['last_update']
                tm.offline_incremental_primer([tk], start_date, end_date)
                print len(tm.live_trades)
                if tm.skipped_tickers != []:
                    self.skipped_tickers.append(tk)
                else:
                    self.ticker_updates.last_update.ix[tk] = end_date
                    if execute:
                        self.live_trades = deepcopy(tm.live_trades)
                        print "pm live trades: %d" % len(self.live_trades)
        
        #inform the user of the status of the tickers:    
        if len(self.skipped_tickers)==0:
            print "all tickers up-to-date"
        else:
            print "unnable to update trends for tickers:"
            print list(self.skipped_tickers)
            
    
    
    
    def create_trade_report(self, ticker_list = []):
        """Creates a list of amended orders and new orders: if a trade has been updated for date d, the orders are for d+1"""
        if ticker_list==[]:
            print "amended orders:"
            
            for tr in self.live_trades:
                print "id: %d" % tr.orig_ID
                print "status: " + tr.trend.get_status()
                if(tr.orig_ID == tr.trade_ID):
                    t = tr
                    pl = 0
                else:
                    i = 0
                    pl = 0
                    for x in self.historic_trades:
                        if x.trade_ID == tr.orig_ID:
                            t = self.historic_trades[i]
                            pl += t.pl_hist[t.execution_info[2][0]]
                        else:
                            if x.orig_ID == tr.orig_ID:
                                pl += x.pl_hist[x.execution_info[2][0]]
                        i += 1
                        
                    
                    
                
                print "entry date: " + datetime.datetime.isoformat(t.execution_info[0][0])
                print "entry price: %f" % t.execution_info[0][1]
                print "a: "+datetime.datetime.isoformat(tr.trend.a)
                print "e: "+datetime.datetime.isoformat(tr.trend.e)
                print "P&L: %f" % (pl + tr.pl_hist[max(t.pl_hist.index)])
                print "open risk = %f" % (tr.risk_hist[max(t.risk_hist.index)]*(tr.trade_size/self.portfolio_size))
                print "new stop levl = %f \n \n " % tr.stop_level
                
        
        print "\n\n new pending orders:\n \n"
        if ticker_list == []:
            ticker_list = self.ticker_updates.index
            
        for tk in ticker_list:
            tm = Trade_Manager()
            #load the pending trades:
            tm.load_file2(tk)
            
            
            pending_data = tm.get_trade_data_pending()
            
            i = -1
            #print i
            for tr in tm.pending_trades:
                i = i + 1
                if tr.order_level in pending_data.order_lvl[:i].values:
                    continue
                print tr.trend.trend_type
                print "ticker: " + tr.trend.ticker
                print "status: " + tr.trend.get_status()
                print "start date: "+datetime.datetime.isoformat(tr.trend.a)
                print "trend end date: "+datetime.datetime.isoformat(tr.trend.e)
                print "entry price: %f" % tr.order_level
                print "limit price: %f" % tr.limit_level
                print "negate_level: %f" % tr.negate_level
                print "stop level if entered: %f" % tr.stop_level
                position_size = self.portfolio_scale/(100.0*tr.avg_tr)
                print "ATR: %f" % tr.avg_tr
                print "position size: %f" % position_size
                tr.plot_pending(self.ticker_updates.last_update.ix[tr.trend.ticker])
                
    def change_last_update(self, ticker, new_last_update):
        """ loads file for ticker, amends the last_update value and re-saves"""
        input = open(ticker+"trade_data2.pkl", 'rb')
        historic_trades, live_trades, pending_trades, incomplete_setups, last_update = \
        pickle.load(input)
        input.close()
        
        output = open(ticker+"trade_data2.pkl", 'wb')
        pickle.dump([historic_trades, live_trades, pending_trades, incomplete_setups, new_last_update], output)
        output.close()
        self.ticker_updates.ix[ticker]['last_update'] = new_last_update
        
        
    
        