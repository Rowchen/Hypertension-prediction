#coding:utf-8
import time
import pandas as pd
import numpy as np
import gc

# encoding=utf8 
import sys
stdi, stdo, stde = sys.stdin, sys.stdout, sys.stderr
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdin, sys.stdout, sys.stderr = stdi, stdo, stde

print "read the orgin data"
part_1 = pd.read_csv('../data/meinian_round1_data_part1_20180408.txt',sep='$')
part_2 = pd.read_csv('../data/meinian_round1_data_part2_20180408.txt',sep='$')
part_1_2 = pd.concat([part_1,part_2])
part_1_2 = pd.DataFrame(part_1_2).sort_values('vid').reset_index(drop=True)
del part_1
del part_2
gc.collect()

train=pd.read_csv('../data/meinian_round1_train_20180408.csv',sep=',',encoding='gbk')
test=pd.read_csv('../data/meinian_round1_test_b_20180505.csv',sep=',',encoding='gbk')
vid_set=pd.concat([train['vid'],test['vid']],axis=0)
vid_set=pd.DataFrame(vid_set).sort_values('vid').reset_index(drop=True)
part_1_2=part_1_2[part_1_2['vid'].isin(vid_set['vid'])].reset_index(drop=True)


print "filter useless"
def filter_None(data):
    data=data[data['field_results']!=u'']
    data=data[data['field_results']!=u'未查']
    data=data[data['field_results']!=u'弃检']
    return data
part_1_2=filter_None(part_1_2)

filter_list=['0203','0209','0702','0703','0705','0706','0709','0726','0730','0731','3601',
             '1308','1316']

part_1_2=part_1_2[~part_1_2['table_id'].isin(filter_list)]


vid_tabid_group = part_1_2.groupby(['vid','table_id']).size().reset_index()
vid_tabid_group['new_index'] = vid_tabid_group['vid'] + '_' + vid_tabid_group['table_id']
vid_tabid_group_dup = vid_tabid_group[vid_tabid_group[0]>1]['new_index']
part_1_2['new_index'] = part_1_2['vid'] + '_' + part_1_2['table_id']


dup_part = part_1_2[part_1_2['new_index'].isin(list(vid_tabid_group_dup))]
dup_part = dup_part.sort_values(['vid','table_id'])
unique_part = part_1_2[~part_1_2['new_index'].isin(list(vid_tabid_group_dup))]


def merge_table(df):
    df['field_results'] = df['field_results'].astype(str)
    if df.shape[0] > 1:
        merge_df = " ".join(list(df['field_results']))
    else:
        merge_df = df['field_results'].values[0]
    return merge_df

print "mergedata"
part1_2_dup = dup_part.groupby(['vid','table_id']).apply(merge_table).reset_index()
part1_2_dup.rename(columns={0:'field_results'},inplace=True)
part1_2_res = pd.concat([part1_2_dup,unique_part[['vid','table_id','field_results']]])
part1_2_res=part1_2_res.reset_index(drop=True)


print "转换坐标轴"
merge_part1_2 = part1_2_res.pivot(index='vid',columns='table_id',values='field_results').reset_index()

def remain_feat(df,thresh=0.9):
    exclude_feats = []
    print('----------移除数据缺失多的字段-----------')
    print('移除之前总的字段数量',len(df.columns))
    num_rows = df.shape[0]
    for c in df.columns:
        num_missing = df[c].isnull().sum()
        if num_missing == 0:
            continue
        missing_percent = num_missing / float(num_rows)
        if missing_percent > thresh:
            exclude_feats.append(c)
    print("移除缺失数据的字段数量: %s" % len(exclude_feats))
    # 保留超过阈值的特征
    feats = []
    for c in df.columns:
        if c not in exclude_feats:
            feats.append(c)
    print('剩余的字段数量',len(feats))
    return feats
feats=remain_feat(merge_part1_2,thresh=0.98)

print "删除缺失字段过多的"
merge_part1_2=merge_part1_2[feats]
train_of_part=merge_part1_2[merge_part1_2['vid'].isin(train['vid'])]
test_of_part=merge_part1_2[merge_part1_2['vid'].isin(test['vid'])]

print "汇总到train和test中"
train=pd.merge(train,train_of_part,on='vid')
test=pd.merge(test,test_of_part,on='vid')

def clean_label(x):
    x=str(x)
    if '+' in x:#16.04++
        i=x.index('+')
        x=x[0:i]
    elif '>' in x:#> 11.00
        i=x.index('>')
        x=x[i+1:]
    elif len(x.split('.'))>2:#2.2.8
        i=x.rindex('.')
        x=x[0:i]+x[i+1:]
    elif u'未做' in x or u'未查' in x or u'弃查' in x:
        x=np.nan
    elif len(x)>6:
        x=x[0:4]
    return x

def data_clean(df):
    for c in [u'收缩压',u'舒张压',u'血清甘油三酯',u'血清高密度脂蛋白',u'血清低密度脂蛋白']:
        print c
        df[c]=df[c].apply(clean_label)
        df[c]=df[c].astype('float64')
    return df

print "train的label清洗"
train=data_clean(train)

print "保存文件"
train.to_csv('../data/train_set.txt',index=False,sep='$',encoding='utf-8')
test.to_csv('../data/test_set.txt',sep='$',index=False,encoding='utf-8')

print "预处理完毕"

