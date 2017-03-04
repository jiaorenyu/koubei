# -*- coding: utf-8 -*-
import sys

import numpy as np
from scipy.optimize import leastsq
import pylab as pl

import random

def func(x, p):
    """
    数据拟合所用的函数: x*p'
    """
    return [np.sum(np.multiply(x[i], p)) for i in range(len(x))]

def residuals(p, y, x):
    """
    实验数据x, y和拟合函数之间的差，p为拟合需要找到的系数
    """
    return y - func(x, p)


def parse_line(line):
    line = line.strip().split(",")
    x = map(int, line[0:-1])
    y = int(line[-1])

    return x, y

def read_sample(samples):
    x = []
    y = []
    for sample in samples:
        x1 = map(int, sample[0:-1])
        y1 = int(sample[-1])
        x.append(x1)
        y.append(y1)

    return x, y

def read_sample_pay(samples):
    x = []
    y = []
    for sample in samples:
        x1 = map(int, sample[7:-1])
        y1 = int(sample[-1])
        x.append(x1)
        y.append(y1)

    return x, y

def train(samples):
    x, y = read_sample(samples)

    x = np.array(x)
    y = np.array(y)

    p0 = np.zeros(len(x[0]))
    p0 = np.add(p0, 0.14)

    # 调用leastsq进行数据拟合
    # residuals为计算误差的函数
    # p0为拟合参数的初始值
    # args为需要拟合的实验数据
    try:
        plsq = leastsq(residuals, p0, args=(y, x))
    except Exception, e:
        return None
    return plsq[0]
