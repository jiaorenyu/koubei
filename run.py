# -*- coding: utf-8 -*-

import sys

from common import *


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("usage:  python sample.py pay_file view_file start_date end_date split_dir stat_dir")
        exit()
        
    user_pay_action = sys.argv[1]
    user_view_action = sys.argv[2]
    start_date = sys.argv[3]
    end_date = sys.argv[4]
    split_dir = sys.argv[5]
    stat_dir = sys.argv[6]
    # 拆分shop    
    handle_split(split_dir, user_pay_action, user_view_action)
    
    # 分天统计支付和浏览量
    handle_stat(split_dir, stat_dir)
    
    cost = 0

    # 对总共2000个shop进行模型训练，并使用训练的参数预测2016-11-01~2016-11-14日的客流量
    for i in range(2000):
        pay_fn = os.path.join(stat_dir, str(i+1)+"_"+"pay_stat_sorted")
        view_fn = os.path.join(stat_dir, str(i+1)+"_"+"view_stat_sorted")
        cost += handle_predict_v2(i+1, pay_fn, view_fn, start_date, end_date)
    
    cost = cost/(2000)
    
    # 误差
    print(cost)

    


