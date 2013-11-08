from pylab import *
import get_data as gd
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series, DataFrame

from matplotlib.dates import  DateFormatter, WeekdayLocator, HourLocator, \
         DayLocator, MONDAY
from matplotlib.finance import quotes_historical_yahoo, candlestick,\
         plot_day_summary, candlestick2
def histplt(series, nbins=20):
    plt.figure()
    hist, bins = np.histogram(series,nbins)
    width = 0.7*(bins[1]-bins[0])
    center = (bins[:-1]+bins[1:])/2
    plt.bar(center, hist, align = 'center', width = width)
    plt.show()



def gpc(fut_code = "TY"):
    import get_data as gd
    from matplotlib.finance import quotes_historical_yahoo
    """Function to plot a candle graph"""
    
    """takes as input the security code, default TY"""
   


    #quotes contains a list of tuples:
    #(date, open, close, high, low, volume)
    quotes = gd.get_data(fut_code, "candle")
    if len(quotes) == 0:
        today = datetime.datetime.now()
        
        date2 = (today.year, today.month, today.day)
        date1 = ( today.year -1, today.month, 1)
        quotes = quotes_historical_yahoo('fut_code', date1, date2)
    if len(quotes) == 0:
        raise SystemExit

    gpcs(quotes)

def gpcs(series):
    """Function to plot a candle graph from a given series"""
    
    """takes as input the security code, default TY"""
    from matplotlib.dates import  DateFormatter, WeekdayLocator, HourLocator, \
         DayLocator, MONDAY
    from matplotlib.finance import quotes_historical_yahoo, candlestick,\
         plot_day_summary, candlestick2
    
    
    mondays = WeekdayLocator(MONDAY)  # major ticks on the mondays
    
    alldays    = DayLocator()              # minor ticks on the days
    monthFormatter = DateFormatter('%b %d')  # Eg, Jan 12
    dayFormatter = DateFormatter('%d')      # Eg, 12


    #quotes contains a list of tuples:
    #(date, open, close, high, low, volume)
    quotes = series
    if len(quotes) == 0:
        today = datetime.datetime.now()
        
        date2 = (today.year, today.month, today.day)
        date1 = ( today.year -1, today.month, 1)
        quotes = quotes_historical_yahoo('fut_code', date1, date2)
    if len(quotes) == 0:
        raise SystemExit

    fig = figure()
    fig.subplots_adjust(bottom=0.2)
    ax = fig.add_subplot(111)
    #ax.xaxis.set_major_locator(mondays)
    #ax.xaxis.set_minor_locator(mondays)
    #ax.xaxis.set_major_formatter(monthFormatter)
    #ax.xaxis.set_minor_formatter(dayFormatter)

    #plot_day_summary(ax, quotes, ticksize=3)
    candlestick(ax, quotes, width=0.6)

    #ax.xaxis_date()
    ax.autoscale_view()
    setp( gca().get_xticklabels(), rotation=45, horizontalalignment='right')

    #show()
def gpcs_trade(series, trade):
    """Function to plot a candle graph from a given series"""
    
    """takes as input the security code, default TY"""
    from matplotlib.dates import  DateFormatter, WeekdayLocator, HourLocator, \
         DayLocator, MONDAY
    from matplotlib.finance import quotes_historical_yahoo, candlestick,\
         plot_day_summary, candlestick2
    
    
    mondays = WeekdayLocator(MONDAY)  # major ticks on the mondays
    
    alldays    = DayLocator()              # minor ticks on the days
    monthFormatter = DateFormatter('%b %d')  # Eg, Jan 12
    dayFormatter = DateFormatter('%d')      # Eg, 12


    #quotes contains a list of tuples:
    #(date, open, close, high, low, volume)
    quotes = series
    if len(quotes) == 0:
        today = datetime.datetime.now()
        
        date2 = (today.year, today.month, today.day)
        date1 = ( today.year -1, today.month, 1)
        quotes = quotes_historical_yahoo('fut_code', date1, date2)
    if len(quotes) == 0:
        raise SystemExit

    fig = figure()
    fig.subplots_adjust(bottom=0.2)
    ax = fig.add_subplot(111)
    #ax.xaxis.set_major_locator(mondays)
    #ax.xaxis.set_minor_locator(mondays)
    #ax.xaxis.set_major_formatter(monthFormatter)
    #ax.xaxis.set_minor_formatter(dayFormatter)

    #plot_day_summary(ax, quotes, ticksize=3)
    dt, o, c, h, l, v = zip(*quotes)
    dt2 = []
    for d in dt:
        dt2.append(datetime.datetime.fromordinal(int(d)))
    ser = DataFrame(data = {0:o, 1:c, 2:h, 3:l, 4:v}, index = dt2)
    tmdelta = len(ser[trade.trend.a:])
    ax.plot([trade.trend.a, max(ser.index)], [trade.trend.c_reg, (tmdelta*trade.trend.m_reg + trade.trend.c_reg)], color='r', linestyle='-', linewidth=2)
    ax.plot([trade.trend.a, max(ser.index)], [trade.trend.c_reg+trade.trend.stdev_p_reg, (trade.trend.stdev_p_reg +tmdelta*trade.trend.m_reg + trade.trend.c_reg)], color='g', linestyle='-', linewidth=2)
    ax.plot([trade.trend.a, max(ser.index)], [trade.trend.c_reg-trade.trend.stdev_p_reg, (-trade.trend.stdev_p_reg +tmdelta*trade.trend.m_reg + trade.trend.c_reg)], color='g', linestyle='-', linewidth=2)
    tmdelta = len(ser[trade.trend.a:trade.trend.e])
    ax.plot([trade.trend.a, trade.trend.e], [trade.trend.c_reg, (tmdelta*trade.trend.m_reg + trade.trend.c_reg)], color='b', linestyle='-', linewidth=2)
    
    candlestick(ax, quotes, width=0.6)

    #ax.xaxis_date()
    ax.autoscale_view()
    setp( gca().get_xticklabels(), rotation=45, horizontalalignment='right')

    show()