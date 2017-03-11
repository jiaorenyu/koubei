import pandas as pd
import pyflux as pf
import sys
import date_util
import common

def get_history_pay(action_count, start_date, end_date, predict_start, predict_end):
    date = []
    count = []
    date_list = date_util.get_date_list(start_date, end_date)
    predict_list = date_util.get_date_list(predict_start, predict_end)
    real_list = []
    
    for day in date_list:
        value = int(action_count[day]) if day in action_count else 0
        count.append(value)
        date.append(len(count))

    for day in predict_list:
        value = int(action_count[day]) if day in action_count else 0
        real_list.append(value)

    return count, date, real_list

def arima(count, index, forward):
    data = pd.DataFrame(data=count, index=index)
    model = pf.ARIMA(data=data, ar=5, ma=5, integ=0)
    x = model.fit("MLE")
    result = model.predict(h=forward, intervals=False)
    kv = result['0'].to_dict()
    keys = list(kv.keys())
    keys.sort()
    value = []
    for key in keys:
        v = kv[key]
        if v < 0:
            v = 0
        value.append(int(v))

    return value


if __name__ == "__main__":
    fn = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]
    predict_start = sys.argv[4]
    predict_end = sys.argv[5]
    action_count = common.load_action_stat(fn)
    count, date, real_value = get_history_pay(action_count, start_date, end_date, predict_start, predict_end)
    predict_value = arima(count, date, 14)
    print(predict_value, real_value)
    if len(real_value) == 0 or len(predict_value) == 0:
        exit()
    print(common.shop_cost(predict_value, real_value))
