import time

def get_timestamp(datestr, formatstr = "%Y-%m-%d"):
     return time.mktime(time.strptime(datestr, formatstr))

def get_datestr(seconds, formatstr = "%Y-%m-%d"):
    return time.strftime(formatstr, time.localtime(seconds))


