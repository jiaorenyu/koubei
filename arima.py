#coding=utf-8
from model import *
import redis
#{shop_id:[user_pay]}
import datetime 
import pandas as pd
from pandas import read_csv
from pandas import DataFrame, Series
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


from matplotlib import pyplot as plt
import statsmodels.api as sm
import numpy as np
import math

def connect_redis():
    client =redis.Redis(host='127.0.0.1',port=6379,db=0)
    return client
        
def distinct_weekday_weekend(str_left,str_right,date_format):       
    weekday = []
    weekend = []
    start = datetime.datetime.strptime(str_left, date_format) 
    while  start <= datetime.datetime.strptime(str_right, date_format):
        if start.weekday() in range(0, 5): 
            weekday.append(start.strftime(date_format))
        else:
            weekend.append(start.strftime(date_format))
        start += datetime.timedelta(days = 1)
    return weekday, weekend


def mysql_to_redis(client):
    #count user by day
    shops_sales = {}
    for shop_id in range(1,2001):
        print shop_id
        #group by sql too show
        #shop_query = UserPay.select(fn.DATE(UserPay.pay_time).alias('day'),fn.Count(UserPay.user_id).alias('user_per_shop')).where(UserPay.shop_id== str(shop_id)).group_by(SQL('day')).order_by(SQL('day asc'))
        shop_query = UserPay.select(UserPay.pay_time).where(UserPay.shop_id == str(shop_id))
        #shops_sales[shop_id] = []
        day_count = {}
        for shop in shop_query:
            day_count[shop.pay_time.strftime('%Y-%m-%d')] = day_count.setdefault(shop.pay_time.strftime('%Y-%m-%d'), 0) + 1 
        
        shops_sales[shop_id] = day_count
    #write to redis
    for shop_id in shops_sales.keys():        
        for day in shops_sales[shop_id].keys():
            client.hset(shop_id, day, shops_sales[shop_id][day])
    #put day count into array

def sorted_dict_values(adict): 
    keys = adict.keys() 
    keys.sort() 
    return [(k, adict[k]) for k in keys]   

def get_pq_with_aic(data):
    import warnings
    with warnings.catch_warnings():
    # catch a hessian inversion and convergence failure warning
        warnings.simplefilter("ignore")
        return sm.tsa.arma_order_select_ic(np.array(data),max_ar=6,max_ma=4,ic='aic')['aic_min_order']

def get_pq_with_test(data):
    pmax=6
    qmax=4
    bic_matrix=[] #bic矩阵
    for p in range(pmax+1):
    	tmp=[]
    for q in range(qmax+1):
        try: #存在部分报错，所以用try来跳过报错。
            tmp.append(ARIMA(data,(p,1,q)).fit().aic)
        except:
            tmp.append(None)
    bic_matrix.append(tmp)
    bic_matrix=pd.DataFrame(bic_matrix) #从中可找出最小值
    p,q=bic_matrix.stack().idxmin() #先用stack展平，然后用idxmin找出最小位置。
    return p, q

def arma_train(data):
    from statsmodels.tsa.stattools import adfuller as ADF
    p, q = get_pq_with_aic(data)
    model = sm.tsa.ARMA(data,(p, q)).fit()
    return model

def sarima_train(data):
    model = sm.tsa.statespace.SARIMAX(data, trend='n', order=(0,1,0), seasonal_order=(1,1,1,7)).fit()
    return model

def sarima_predict(data, start, end,  model):
    predicts = model.predict(start, end, dynamic = True)
    #print predicts, len(predicts)
    #print data, len(data)
    #plt.plot(range(14), predicts)
    #plt.show()
    #plot_data(data, predicts)
    return map(int, predicts)

def arima_train(data):
    ''' 
        train with algorithm arima
    '''
    #train unique shop with history data
    #ADF test
    print '====================='
    print len(data), data
    from statsmodels.tsa.stattools import adfuller as ADF
    print ADF(pd.Series(np.array(data)))[1]
    p, q = get_pq_with_test(data)
    #p, q = get_pq_with_aic(data)
    print 'ARIMA p值和q值为: %s、%s'%(p,q) #0,1
    #p, q = 0, 0

    model=ARIMA(data,(p,1,q)).fit() #建立ARIMA模型
    return model

def arma_predict(data, start, model):
    predicts = model.predict(start, start + len(data) - 2, dynamic = False)
    print predicts, len(predicts)
    print data, len(data)
    plot_data(data, predicts)

def arima_train_broadcast(data):
    X = data 
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0][0]
        predictions.append(yhat)
        obs = test[t]
	history.append(yhat)
	#print('predicted=%f, expected=%f' % (yhat, obs))
    train.extend(predictions)
    plot_data(X, train)

def plot_data(origin, predict):
    df = pd.DataFrame()
    print len(origin), len(predict)
    error = mean_squared_error(origin, predict)
    print('Test MSE: %.3f' % error)
    df['original'] = origin 
    df['predict'] = predict
    df.plot()
    plt.show()

def arima_predict(data,start, model):
    #predicts = []
    #for t in range(len(data)):
    #    predicts.append(model.forecast()[0])
    predicts = model.predict(start, start + len(data) - 2, typ='levels')
    print predicts, len(predicts)
    print data, len(data)
    plot_data(data, predicts)

if __name__ == '__main__':
    #get day from mysql orm to redis 
    #print distinct_weekday_weekend('2015-07-01', '2016-10-31', '%Y-%m-%d')
    #connect redis
    client = connect_redis()
    #mysql_to_redis()
    res = []
    for x in range(1,2001):
        print "shop id is %d" % x
        day_users = client.hgetall(str(x)) 
        users = [ int(day_user[1]) for day_user in sorted_dict_values(day_users) ]
        #arima_train_broadcast(users)
        #split_pos = int(math.floor(len(users) * 0.66))
        split_pos = int(math.floor(len(users)))
        try:
            model = sarima_train(users[:split_pos])
            shop_res = sarima_predict([], split_pos, split_pos + 13, model)
            head = str(x) + ','
            res.append(head + ','.join(map(str,shop_res)))
        except ValueError:
            continue
    with open('result.csv', 'w') as wf:
        wf.write('\n'.join(res))
