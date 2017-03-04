#-*- coding: utf-8 -*-

from datetime import datetime as dt
import datetime

def next_day(datestr, formatstr="%Y-%m-%d"):
    day = dt.strptime(datestr, formatstr)
    day = day + datetime.timedelta(days=1)

    return day.strftime(formatstr)

def next_ndays(n, datestr, formatstr="%Y-%m-%d"):
    day = dt.strptime(datestr, formatstr)
    day = day + datetime.timedelta(days=n)

    return day.strftime(formatstr)

'''
获取指定范围的字符串形式日期列表
start_date:     字符串格式开始日期
end_date:       字符串格式结束日期
formatstr:      日期格式
'''
def get_date_list(start_date, end_date, formatstr= "%Y-%m-%d"):
    start_day = dt.strptime(start_date, formatstr)
    end_day = dt.strptime(end_date, formatstr)
    delta = end_day - start_day
    days = delta.days

    date_list = []

    p_day = start_day
    
    for count in range(days):
        date_list.append(p_day.strftime(formatstr))
        p_day = p_day + datetime.timedelta(days=1)

    return date_list



def get_week(datestr, formatstr="%Y-%m-%d"):
    day = dt.strptime(datestr, formatstr)
    
    return day.weekday()
