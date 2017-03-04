#-*- coding:utf-8 -*-
import os

import numpy as np

import file_util
import time_util
import date_util
import train

def parse_user_action(line):
    line = line.strip()
    action = line.split(",")
    if len(action) != 3 or "" in action:
        return False,"",""
    
    shop_id = action[1]
    day = action[2].split()[0]

    return True,shop_id,day

def write_shop(basedir, shop_fp, shop_id, day, action_type):
    if shop_id not in shop_fp:
        shop_fp[shop_id] = open(os.path.join(basedir, shop_id+"_"+action_type), "a")
    
    shop_fp[shop_id].write(shop_id + "," + day + "\n")

def split_shop(basedir, fn, action_type):
    shop_fp = dict()
    if len(file_util.list_files(basedir)) != 0:
        return

    with open(fn) as fp:
        for line in fp:
            st, shop_id, day = parse_user_action(line)
            if st:
                write_shop(basedir, shop_fp, shop_id, day, action_type)

    for _,fp in shop_fp.items():
        fp.close()
'''
拆分shop
'''
def handle_split(basedir, user_pay_action, user_view_action):
    split_shop(basedir, user_pay_action, "pay")
    split_shop(basedir, user_view_action, "view")



def count_action(action_count, day):
    if day not in action_count:
        action_count[day] = 0
    
    action_count[day] += 1

def parse_action_line(line):
    line = line.strip()
    line = line.split(",")
    shop_id = line[0]
    day = line[1]
    return shop_id, day

def day_key(datestr):
    return time_util.get_timestamp(datestr, "%Y-%m-%d")

def stat(path, stat_dir):
    action_count = dict()
    with open(path, "r") as fp:
        for line in fp:
            shop_id, day = parse_action_line(line)
            count_action(action_count, day)

        stat_path = os.path.join(stat_dir, path.split("/")[-1] + "_stat_sorted")
        
        with open(stat_path, "a") as fp1:
            day_list = action_count.keys()
            day_list_sorted = sorted(day_list, key=day_key)
            for day in day_list_sorted:
                fp1.write(day+","+str(action_count[day]) + "\n")

    print("finished " + path)

def handle_stat(split_dir, stat_dir):
    if len(file_util.list_files(stat_dir)) != 0:
        return
    file_list = file_util.list_files(split_dir)
    for fn in file_list:
        stat(fn, stat_dir)

def parse_stat_line(line):
    items = line.strip().split(",")
    day = items[0]
    count = items[1]
    return day, count

def load_action_stat(fn):
    if not os.path.exists(fn):
        return dict()

    day_count = dict()
    with open(fn) as fp:
        for line in fp:
            day, count = parse_stat_line(line)
            day_count[day] = count
    
    return day_count

def week_align(date_list):
    start_date_week = date_util.get_week(date_list[0])
    end_date_week = date_util.get_week(date_list[-1])
    
    week_steps = 7
    start_index = (week_steps - start_date_week) % 7
    end_index = -1 * (end_date_week + 1) 

    return date_list[start_index : end_index]

def get_week_list(date_list):
    week_list = []

    week_step = 7
    weeks = int(len(date_list) / week_step)

    for i in range(weeks):
        week_list.append(date_list[i*week_step:(i+1)*week_step])
    
    return week_list

def parse_date_list(start_date, end_date, formatstr = "%Y-%m-%d"):
    date_list = date_util.get_date_list(start_date, end_date, formatstr)
    date_list = week_align(date_list)
    
    week_list = get_week_list(date_list)
    return week_list

def pre_handle_week_list(week_list):
    
    week_dict = dict()
    for i in range(7):
        week_dict[i] = []
    
    from copy import deepcopy
    for i in range(len(week_list)-1):
        week_date = week_list[i]
        next_week_date = week_list[i+1]
        for week_num, date in enumerate(next_week_date):
            new_week = deepcopy(week_date)
            new_week.append(date)
            week_dict[week_num].append(new_week)
    
    return week_dict
    
def get_sample(pay_action_count, view_action_count, week):
    sample = []
    pay_count = []
    view_count = []
    
    for day in week:
        pc = pay_action_count[day] if day in pay_action_count else '0' 
        vc = view_action_count[day] if day in view_action_count else '0' 
        pay_count.append(pc)
        view_count.append(vc)
    
    if pay_count.count("0") > 3:
        return None
    view_count = view_count[0:-1]
    sample += view_count
    sample += pay_count
    return sample

def get_sample_v2(pay_action_count, view_action_count, duration, predict_no):
    sample = []
    pay_count = []
    view_count = []
    
    for day in duration:
        pc = pay_action_count[day] if day in pay_action_count else '0' 
        vc = view_action_count[day] if day in view_action_count else '0' 
        pay_count.append(pc)
        view_count.append(vc)
    
    if pay_count.count("0") > 0:
        return None
    sample += view_count
    sample += pay_count
    predict_day = date_util.next_ndays(predict_no, duration[-1])
    predict_pay_count = pay_action_count[predict_day] if predict_day in pay_action_count else '0'
    sample.append(predict_pay_count)
    return sample

def get_sample_list(pay_action_count, view_action_count, week_list):
    sample_list = []

    for week in week_list:
        sample = get_sample(pay_action_count, view_action_count, week)
        if sample:
            sample_list.append(sample)

    return sample_list

'''
不考虑周几的影响
'''
def get_sample_list_v2(pay_action_count, view_action_count, history_count, predict_no, date_list):

    sample_list = []
    for i in range(len(date_list) - predict_no - history_count):
        sample = get_sample_v2(pay_action_count, view_action_count, date_list[i:i+history_count], predict_no)
        if sample:
            sample_list.append(sample)

    return sample_list

def handle_sample_v2(history_count, predict_count, pay_action_count, view_action_count, start_date, end_date, formatstr):

    date_list = date_util.get_date_list(start_date, end_date, formatstr)
    predict_count_sample = dict()

    for i in range(predict_count):
        predict_count_sample[i+1] = get_sample_list_v2(pay_action_count, view_action_count, history_count, i+1, date_list)

    return predict_count_sample

def handle_sample(pay_action_count, view_action_count, start_date, end_date, formatstr):
    week_list = parse_date_list(start_date, end_date, formatstr)
    week_dict = pre_handle_week_list(week_list)
    
    # week : sample_list
    week_sample = dict()    
    for week_num, week_list in week_dict.items():
        week_sample[week_num] = get_sample_list(pay_action_count, view_action_count, week_list)

    return week_sample

def get_latest_sample_pay(start_date, end_date, pay_action_count, formatstr="%Y-%m-%d"):
    date_list = date_util.get_date_list(start_date, end_date, "%Y-%m-%d")
    
    pay = []
    for day in date_list:
        pay_count = pay_action_count[day] if day in pay_action_count else "0"

        pay.append(pay_count)

    return pay

def get_latest_sample(start_date, end_date, predict_no,  pay_action_count, view_action_count, formatstr="%Y-%m-%d"):
    date_list = date_util.get_date_list(start_date, end_date, "%Y-%m-%d")
    
    view = []
    pay = []
    for day in date_list:
        pay_count = pay_action_count[day] if day in pay_action_count else "0"
        view_count = view_action_count[day] if day in view_action_count else "0"

        pay.append(pay_count)
        view.append(view_count)
    
    day = date_util.next_ndays(predict_no-1, end_date)

    real_pay_count = pay_action_count[day] if day in pay_action_count else '0'

    return view + pay, real_pay_count

def post_handle(value):
    value = 0 if value < 0 else int(value)
    return value

def shop_cost(predict_list, real_list):
    cost = 0
    for i in range(len(predict_list)):
        predict = predict_list[i]
        real = real_list[i]

        cost += abs(predict - real)*1.0/(predict + real)

    return cost/len(predict_list)

def handle_predict_v2(shop_id, pay_fn, view_fn, start_date, end_date):
    pay_action_count = load_action_stat(pay_fn)
    view_action_count = load_action_stat(view_fn)
    
    predict_count_sample = handle_sample_v2(7, 14, pay_action_count, view_action_count, start_date, end_date, "%Y-%m-%d")

    predict_list = []
    real_list = []
    for predict_no, sample_list in predict_count_sample.items():
        p = train.train(sample_list)
        
        if p == None:
            predict_list.append(0)
            continue
        sample, real_pay_count = get_latest_sample("2016-10-11", "2016-10-18", predict_no, pay_action_count, view_action_count)
        
        sample = map(int, sample)
        rt = np.sum(np.multiply(p, sample))
        predict_list.append(rt)
        real_list.append(int(real_pay_count))


    predict_list = map(post_handle, predict_list)
    
    result = []
    result.append(shop_id)
    result += predict_list

    result = map(str, result)

    if len(result) != 15:
        print("error")
    print(",".join(result))

    cost = shop_cost(predict_list, real_list)   

    return cost


def handle_predict(shop_id, pay_fn, view_fn, start_date, end_date):
    pay_action_count = load_action_stat(pay_fn)
    view_action_count = load_action_stat(view_fn)
    
    week_sample = handle_sample(pay_action_count, view_action_count, start_date, end_date, "%Y-%m-%d")

    predict_list = []
    for week_num, sample_list in week_sample.items():
        p = train.train(sample_list)
        if p == None:
            predict_list.append(0)
            continue
        sample = get_latest_sample_pay("2016-10-24", "2016-10-31", pay_action_count)
        sample = map(int, sample)
        rt = np.sum(np.multiply(p, sample))
        predict_list.append(rt)

    predict_list = map(post_handle, predict_list)

    result = []
    result.append(shop_id)
    result += predict_list[1:] # 周二到周日
    result += predict_list  # 周一到周日
    result += predict_list[0:1] # 周一

    result = map(str, result)

    print(",".join(result))



