# *-* coding:utf-8 *-*

import sys

import pylab as pl
from matplotlib.ticker import MultipleLocator, FuncFormatter

import time_util

def formatter(x, pos):
    return time_util.get_datestr(x, "%a")

def parse_line(line):
    line = line.strip()
    line = line.split(",")

    day = line[0]
    count = line[1]

    return day, count

if __name__ == "__main__":
    fn = sys.argv[1]   
    
    day_list = []
    count_list = []
    with open(fn) as fp:
        for line in fp:
            day, count = parse_line(line)
            
            day_list.append(day)
            count_list.append(count)

        pl.plot(map(time_util.get_timestamp, day_list), count_list)
    
        ax = pl.gca()
        xmajorLocator   = MultipleLocator(3600*24)
        xmajorFormatter = FuncFormatter(formatter)

        #设置主刻度标签的位置,标签文本的格式  
        ax.xaxis.set_major_locator(xmajorLocator)  
        ax.xaxis.set_major_formatter(xmajorFormatter)

        pl.show()
