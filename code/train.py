#coding:utf-8
import time
import pandas as pd
import numpy as np
import gc
import re
from tqdm import tqdm  
from utils import *
import xgboost as xgb
import time
filename=time.strftime('../submit/submit_%Y%m%d_%H%M%S.csv',time.localtime(time.time()))

# encoding=utf8 
import sys
stdi, stdo, stde = sys.stdin, sys.stdout, sys.stderr
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdin, sys.stdout, sys.stderr = stdi, stdo, stde

train=pd.read_csv('../data/train_set.txt',sep='$',encoding='utf8',low_memory=False)
test=pd.read_csv('../data/test_set.txt',sep='$',encoding='utf8',low_memory=False)
pred_list=[u'收缩压',u'舒张压',u'血清甘油三酯',u'血清高密度脂蛋白',u'血清低密度脂蛋白']

train.loc[train[u'收缩压']==0,u'收缩压']=train[u'收缩压'].mean()
train.loc[train[u'舒张压']==0,u'舒张压']=train[u'舒张压'].mean()
train.loc[train[u'舒张压']==100164.0,u'舒张压']=100
train.loc[train[u'舒张压']==974.0,u'舒张压']=97.4
train.loc[train[u'血清甘油三酯']>10,u'血清甘油三酯']=10
train[u'血清低密度脂蛋白']=abs(train[u'血清低密度脂蛋白'])
for col in pred_list:
    train[col]=train[col].fillna(train[col].mean())

train['model']=1
test['model']=0
alldata=train.append(test)

print "查找数值特征"
digit_dataframe = pd.DataFrame()
digit_index = []
digit_radio = []

for col in tqdm(alldata.columns):
    tmp = alldata[col].apply(find_digit)
    digit_index.append(col)
    digit_radio.append(tmp.sum()/float(alldata.shape[0]))
digit_dataframe['digit_index'] = digit_index
digit_dataframe['digit_radio'] = digit_radio
digit_dataframe = digit_dataframe.sort_values('digit_radio',ascending=False).reset_index(drop=True)

num_features2=list(digit_dataframe[digit_dataframe.digit_radio>0.03]['digit_index'].values)

num_features=num_features2[:]
for c,i in enumerate(num_features2):
    if i in pred_list+['vid','model']:
        num_features.remove(i)
del num_features2

for col in tqdm(num_features):
    alldata[col+'_fix']=alldata[col].apply(deal_num_feat)
    alldata[col+'_fix']=alldata[col+'_fix'].fillna(alldata[col+'_fix'].mean())
num_features2=num_features[:]
for i in xrange(len(num_features)):
    num_features2[i]=num_features[i]+'_fix'
alldata.loc[alldata['2405_fix']==0,'2405_fix']=alldata['2405_fix'].mean()
alldata.loc[alldata['2404_fix']==0,'2404_fix']=alldata['2404_fix'].mean()
alldata.loc[alldata['2403_fix']==0,'2403_fix']=alldata['2403_fix'].mean()

print "处理文本特征"

new_list=['0426_zayin','0912_zhongda','1001_zhengchang','1001_buqi','0101_deal1','0101_deal2',
          '0101_deal3','0113_deal','0102_qianliexian','0102_ruxian','0101_ruxian','0101_wenluan','0113_lunkuo','shengzang','0121_zigong','fujian']
alldata['0426_zayin']=alldata['0426'].apply(deal0426)
alldata['0912_zhongda']=alldata['0912'].apply(deal0912)
alldata['1001_zhengchang']=alldata['1001'].apply(deal1001_1)
alldata['1001_buqi']=alldata['1001'].apply(deal1001_2)
alldata['0101_deal1']=alldata['0101'].apply(deal0101_1)
alldata['0101_deal2']=alldata['0101'].apply(deal0101_2)
alldata['0102_qianliexian']=alldata['0102'].apply(deal0102_qianliexian)
alldata['0102_ruxian']=alldata['0102'].apply(deal0102_ruxian)
alldata['0101_ruxian']=alldata['0101'].apply(deal0101_ruxian)
alldata['0101_wenluan']=alldata['0101'].apply(deal0101_wenluan)
alldata['0113_lunkuo']=alldata['0113'].apply(deal_0113_lunkuo)
alldata['shengzang']=alldata['0117'].apply(deal_0117_0118)|alldata['0118'].apply(deal_0117_0118)
alldata['0121_zigong']=alldata['0121'].apply(deal_0121)
alldata['fujian']=alldata['0122'].apply(deal_0122_0123)|alldata['0123'].apply(deal_0122_0123)
alldata['0101_deal3']=alldata['0101'].apply(deal0101_3)
alldata['0113_deal']=alldata['0113'].apply(deal_0113)
alldata['0113_zhifanggan']=alldata['0113'].apply(zhifanggan0113)

alldata['0102_zhifang']=alldata['0102'].apply(deal0102_zhifang)
alldata['0102_xin']=alldata['0102'].apply(deal0102_xin)
alldata['0114_dan']=alldata['0114'].apply(deal0114)
his_list=['0409_bool','0409_tang','0409_xuezhi','0409_xueya','0409_zhifanggan',
          '0434_bool','0434_tang','0434_xuezhi','0434_xueya','0434_zhifanggan']
alldata['0434_bool']=alldata['0434'].apply(gaoxueya)
alldata['0434_tang']=alldata['0434'].apply(tangniaobing)
alldata['0434_xuezhi']=alldata['0434'].apply(xuezhi)
alldata['0434_xueya']=alldata['0434'].apply(xueya)
alldata['0434_zhifanggan']=alldata['0434'].apply(zhifanggan)
alldata['0409_bool']=alldata['0409'].apply(gaoxueya)
alldata['0409_tang']=alldata['0409'].apply(tangniaobing)
alldata['0409_xuezhi']=alldata['0409'].apply(xuezhi)
alldata['0409_xueya']=alldata['0409'].apply(xueya)
alldata['0409_zhifanggan']=alldata['0409'].apply(zhifanggan)

alldata['3195']=alldata['3195'].apply(deal_3195)
alldata['3191']=alldata['3191'].apply(deal_3195)
alldata['3192']=alldata['3192'].apply(deal_3195)
alldata['3197']=alldata['3197'].apply(deal_3195)
alldata['3190']=alldata['3190'].apply(deal_3195)
alldata['3196']=alldata['3196'].apply(deal_3196)
alldata['100010']=alldata['100010'].apply(deal_3195)
alldata['2302']=alldata['2302'].apply(deal_2302)
alldata['3399']=alldata['3399'].apply(deal_3399)
alldata['0102_test']=alldata['0102'].apply(deal0102_test)
print "处理完毕"


train=alldata[alldata['model']==1]
test=alldata[alldata['model']==0]

feat=num_features2+['3195','3191','3192','3197','3190','3196','100010','2302','0113_zhifanggan',
                  '0114_dan','0102_xin','0102_zhifang','0102_test']+his_list+new_list
best_iter_dict={u'收缩压':542,u'舒张压':709,u'血清甘油三酯':491,u'血清高密度脂蛋白':1178,u'血清低密度脂蛋白':1036}

for col in pred_list:
    print col
    features=feat
    target=col
    xgb_params = {'max_depth':5, 'eta':.05, 'objective':'reg:linear', 'verbose':0, 'eval_metric':['rmse'],
                 'subsample':0.75, 'min_child_weight':50, 'gamma':0.1,'lambda':10,
                 'nthread': 4, 'colsample_bytree':.7, 'base_score':train[target].mean(), 'seed': 2018}
    xgb_train = xgb.DMatrix(train[features],label= train[target])
    model = xgb.train(xgb_params,xgb_train,num_boost_round=best_iter_dict[col],feval=xgbloss)
    
    xgb_test = xgb.DMatrix(test[features])
    test.loc[:,col]=model.predict(xgb_test)

for col in [u'血清甘油三酯']:
    print col
    features=feat+[u'收缩压',u'舒张压',u'血清高密度脂蛋白',u'血清低密度脂蛋白']
    target=col
    xgb_params = {'max_depth':5, 'eta':.05, 'objective':'reg:linear', 'verbose':0, 'eval_metric':['rmse'],
                 'subsample':0.75, 'min_child_weight':50, 'gamma':0.1,'lambda':10,
                 'nthread': 4, 'colsample_bytree':.7, 'base_score':train[target].mean(), 'seed': 2018}
    xgb_train = xgb.DMatrix(train[features],label= train[target])
    model = xgb.train(xgb_params,xgb_train,num_boost_round=471,feval=xgbloss)
    
    xgb_test = xgb.DMatrix(test[features])
    test.loc[:,col+'_2']=model.predict(xgb_test)
test[u'血清甘油三酯']=(test[u'血清甘油三酯_2']+test[u'血清甘油三酯'])/2


for col in pred_list:
    test.loc[:,col]=np.ndarray.round(test[col].values,3)
test[['vid']+pred_list].to_csv(filename,header=False,index=False,sep=',')
for col in pred_list:
    print test[col].describe()
