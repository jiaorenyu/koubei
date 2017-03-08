import pandas as pd
import pyflux as pf
import sys

def get_history_pay(fn):
    date = []
    count = []
    with open(fn) as fp:
        for line in fp:
            item = line.strip()
            item = item.split(",")
            date.append(len(count)+1)
            count.append(int(item[1]))

    return count, date

def arima(count, index):
    data = pd.DataFrame(data=count, index=index)
    model = pf.ARIMA(data=data, ar=4, ma=4, integ=0)
    x = model.fit("MLE")
    result = model.predict(h=14, intervals=False)
    kv = result['0'].to_dict()
    keys = list(kv.keys())
    keys.sort()
    value = []
    for key in keys:
        value.append(str(int(kv[key])))

    return ",".join(value)


if __name__ == "__main__":
    fn = sys.argv[1]
    date = []
    count = []
    with open(fn) as fp:
        for line in fp:
            item = line.strip()
            item = item.split(",")
            date.append(len(count)+1)
            count.append(int(item[1]))

    arima(count, date)
