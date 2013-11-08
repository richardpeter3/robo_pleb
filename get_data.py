from pylab import *
from matplotlib.dates import  DateFormatter, WeekdayLocator, HourLocator, \
         DayLocator, MONDAY
from matplotlib.finance import quotes_historical_yahoo, candlestick,\
         plot_day_summary, candlestick2
import datetime
import pandas as pd
from pandas import DataFrame, Series
        
def get_data(start_date, end_date, fut_code="TY", format = "candle"):
    """function to pull the historic data for the securities in arg from the db"""

    """Takes inputs:
    fut_code = the futures code that you require
    format =  string 'close'|'candle' (candle include volume)
    """
    import psycopg2
    from pylab import datetime as pydt
    con = psycopg2.connect(host = 'localhost', database='financial_data', user='fredfeed', password = 'something')
    cur = con.cursor()
    
    if(format=="candle"):
        query = """select datestring, opn, cls, hi, lo, vol from"""
    else:
        query = """select datestring, cls from"""   
    query = query + """
        dailydatatest as ddt 
        left outer join
        futurescontractspecification as fcs
        on ddt.contractid = fcs.id
        left outer join futcodes as fc
        on fcs.shtcodeid = fc.codeid
        where code = '""" + fut_code+ """'
        limit 1250"""
    cur.execute(query)
    data = cur.fetchall()
    if(len(data)!=0):
        print "obtained data from db"
        #Next we change the format of the datetime to ordinal for easier access
        #and consistency with the yahoo finance functions
        #start by unziping it
        if format=="candle":
            dt, op, cl, hi, lo, vol = zip(*data)
        else:
            dt, cl = zip(*data)
        #next create a lamda function to apply the transformation to each element:
        def apply_to_list(some_list, f):
            return [f(x) for x in some_list]
        
        ordinate_date = apply_to_list(dt, lambda x: pydt.datetime.toordinal(x))
        
        #re-zip and return 
        if format=="candle":
            return zip(ordinate_date, op, cl, hi, lo, vol)
        else:
            return zip(ordinate_date, cl)
    else:
        return get_data_yahoo(start_date, end_date, fut_code, format)
        
def write_series(series, filepath = "output.txt"):
    """output a series to csv format"""
        
    file = open(filepath, 'w')
    for s in series:
        dt, o, c, h, l, v = s
        file.write("{}, {}, {}, {}, {}, {}\n".format(dt, o, c, h, l, v))
    
    file.close()

 
def get_data_yahoo_orig(start_date, end_date, fut_code="INTC", format = "candle"):
    try:
        quotes = quotes_historical_yahoo(fut_code, start_date, end_date)
    except:
        print("Unable to get quotes for %s", fut_code)
        return []
    if quotes == None:
        print("Unable to get quotes for %s", fut_code)
        return []
    if len(quotes)==0:
        print("Unable to get quotes for %s", fut_code)
        return []
    else:
        if(quotes[0][0] < float(datetime.datetime.toordinal(start_date)) or \
            quotes[len(quotes)-1][0] > float(datetime.datetime.toordinal(end_date))):
            print "unnable to get yahoo quotes for ticker %s for required dates" %fut_code
            return None
        if(format=="candle"):
            return quotes
        else:
            dt, op, cl, hi, lo, vol = zip(*quotes)
            return zip(dt, cl)

def get_data_yahoo(start_date, end_date, fut_code="INTC", format = "candle"):
    try:
        quotes = quotes_historical_yahoo(fut_code, start_date, end_date)
    except:
        print("Unable to get quotes for %s", fut_code)
        return []
    if quotes == None:
        print("Unable to get quotes for %s", fut_code)
        return []
    if len(quotes)==0:
        print("Unable to get quotes for %s", fut_code)
        return []
    else:
        if(quotes[0][0] < float(datetime.datetime.toordinal(start_date)) or \
            quotes[len(quotes)-1][0] > float(datetime.datetime.toordinal(end_date))):
            print "unnable to get yahoo quotes for ticker %s for required dates" %fut_code
            return None
        if(format=="candle"):
            dt, op, cl, hi, lo, vol = zip(*quotes)
            dt2 = []
            for d in dt:
                dt2.append(datetime.datetime.fromordinal(int(d)))
            quotes = zip(dt2, op, cl, hi, lo, vol)
            return quotes
        else:
            dt, op, cl, hi, lo, vol = zip(*quotes)
            return zip(dt, cl)
def get_snp_tickers():
    "returns a list of SnP 500 ticker strings"
    import csv
    file = open("/home/richard/SnP_Tickers.csv", 'r')
    csv_rdr = csv.reader(file)
    tkrs = []
    for row in csv_rdr:
        for column in row:
            tkrs.append(column)
    
    return tkrs

def get_ftse_tickers():
    "returns a list of SnP 500 ticker strings"
    import csv
    file = open("/home/richard/FTSE100_Tickers.csv", 'r')
    csv_rdr = csv.reader(file)
    tkrs = []
    for row in csv_rdr:
        for column in row:
            tkrs.append(column + ".L")
    return tkrs
def quote_dataframe(quotes):
    "turns a list of tuples into a dataframe"
    dt, o, c, h, l, v = zip(*quotes)
    return pd.DataFrame(data = {0:dt, 1:o, 2:c, 3:h, 4:l, 5:v}, index = dt )   
