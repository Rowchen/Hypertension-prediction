#coding:utf-8
import re
import numpy as np

def deal_0122_0123(x):
    x=str(x)
    return (not re.search('附件区未见明显异常|附件未见明显异常',x)==None)
def deal_0121(x):
    x=str(x)
    return (not re.search('大小正常|形态正常',x)==None)
def deal_0117_0118(x):
    x=str(x)
    return (not re.search('肾内可见多个强回声|肾内可见多个点状强回声|肾 盏可见一个点状强回声|肾下盏可见一个强回声|肾盏可见多个点状强回声',x)==None)
def deal0101_wenluan(x):
    x=str(x)
    return (not re.search('内部结构紊乱|内部结构稍紊乱',x)==None)
def deal0101_ruxian(x):
    x=str(x)
    return (not re.search('双侧乳腺腺体层增厚|双侧乳腺腺体层轻度增厚',x)==None)
def deal0102_qianliexian(x):
    x=str(x)
    return (re.search('前列腺钙化灶|前列腺稍大|前列腺增生并钙化',x)==None)
def deal0102_ruxian(x):
    x=str(x)
    return (re.search('双侧乳腺小叶增生|双侧乳腺小叶增生',x)==None)
def deal0117(x):
    x=str(x)
    return (not re.search('CDFI肾内血流分布正常',x)==None)


def deal_0113_lunkuo(x):
    x=str(x)
    return (not re.search('轮廓规整',x)==None)
def deal_0113(x):
    x=str(x)
    if (not re.search('肝内管道结构尚清晰|肝内管道走向尚清晰',x)==None):
        return 1
    elif (not re.search('肝内管道结构欠清晰|肝内管道结构不清晰|肝内管道走向欠清晰|肝内管状系统走行欠清晰',x)==None):
        return 2
    else:
        return 0
def zhifanggan0113(x):
    x=str(x)
    if (not re.search('肝内回声呈点状密集弥漫性增强|后方伴衰减',x)==None):
        return 2
    elif (not re.search('肝内回声稍呈点状密集弥漫性增强',x)==None):
        return 1
    else:
        return 0 

def deal0101_3(x):
    x=str(x)
    return (not re.search('动脉.*可见.*回声',x)==None)or(not re.search('动脉.*探及.*回声',x)==None)
def deal0101_2(x):
    x=str(x)
    return (not re.search('左室后壁无增厚|左室后壁不厚|左室后壁运动良好|左室后壁厚度及搏幅正常|各瓣膜无增厚',x)==None)
def deal0101_1(x):
    x=str(x)
    a=re.search('(左室后壁厚)([0-9]+|\d+\.\d+)mm',x)
    if (not a==None):
        if float(a.group(2))>10:
            return True
    else:
        a=re.search('(左室后壁)([0-9]+|\d+\.\d+)mm',x)
        if (not a==None):
            if float(a.group(2))>10:
                return True
        else:
            a=re.search('左室后壁增厚',x)
            if(not a==None):
                return True
    return False
def deal1001_1(x):
    x=str(x)
    return (not re.search('正常',x)==None)
def deal1001_2(x):
    x=str(x)
    return (not re.search('不齐',x)==None)
def deal0912(x):
    x=str(x)
    return (not re.search('大',x)==None)&(not re.search('无肿大|不肿大',x)==None)
def deal0426(x):
    x=str(x)
    return (not re.search('杂音',x)==None)&(re.search('无|未',x)==None)
def deal0102_test(x):
    x=str(x)
    return not re.search('.肾结|.肾囊肿',x)==None
def deal0114(x):
    x=str(x)
    return (re.search('未见.*回声|无.*回声',x)==None)&(not re.search('囊腔.*回声',x)==None)
def deal0102_xin(x):
    x=str(x)
    return not re.search('左心室',x)==None
def deal0102_zhifang(x):
    x=str(x)
    if re.search('脂肪肝',x)==None:
        return 0
    elif not re.search('脂肪肝（轻度）',x)==None:
        return 1
    elif not re.search('脂肪肝（中度）',x)==None:
        return 2
    elif not re.search('脂肪肝（重度）',x)==None:
        return 3
    elif not re.search('脂肪肝',x)==None:
        return 1



def tangniaobing(x):
    x=str(x)
    return not re.search('糖尿病',x)==None
def gaoxueya(x):
    x=str(x)
    return not re.search('高血压史',x)==None
def xueya(x):
    x=str(x)
    return not re.search('血压偏高',x)==None
def zhifanggan(x):
    x=str(x)
    return not re.search('脂肪肝',x)==None
def xuezhi(x):
    x=str(x)
    return not re.search('血脂偏高',x)==None




def myloss(pred,dtrain):
    true_label=dtrain.get_label()
    rows=pred.shape[0]
            
    pred2=np.log(1+pred)
    true_label2=np.log(1+true_label)
    return ('myloss',np.sum((pred2-true_label2)**2)/rows,False)

def xgbloss(pred,dtrain):
    true_label=dtrain.get_label()
    rows=pred.shape[0]   
    pred2=np.log(1+pred)
    true_label2=np.log(1+true_label)
    return 'myloss',np.sum((pred2-true_label2)**2)/rows

def myloss2(pred,true_label):
    rows=pred.shape[0]
    pred2=np.log(1+pred)
    true_label2=np.log(1+true_label)
    return np.sum((pred2-true_label2)**2)/rows

def find_digit(x):
    x=str(x)
    a=re.search('\d+\.\d+',x)
    if not a==None:
        return True
    else:
        a=re.search('\d+',x)
        if not a==None:
            return True
        
def deal_num_feat(x):
    x=str(x)
    a=re.search('\d+\.\d+',x)
    if a==None:
        a=re.search('\d+',x)
        if a==None:
            x=np.nan
        else:
            x=float(a.group())
    else:
        x=float(a.group())
    return x


def deal_0424(x):
    try:
        a=re.search('\d+\.\d+',x)
        if a==None:
            a=re.search('\d+',x)
        x=float(a.group())
    except:
        try:
            x=x.encode('utf-8')
            if not re.search(u'未见异常|正常',x)==None:
                x=np.nan
            elif not re.search(u'窦性心动过缓',x)==None:
                x=57.0
            elif not re.search(u'窦性心动过速',x)==None:
                x=105.0
            else:
                    #print x
                x=np.nan
        except:
            #print "捕捉不到数字了",x
            x=np.nan
    return x

def deal_0425(x):
    try:
        x=float(x)
    except:
        try:
            a=re.search('\d+\.\d+',x)
            if a==None:
                a=re.search('\d+',x)
            x=float(a.group())
        except:
            x=x.encode('utf-8')
            if not re.search(u'未见异常|正常|无异常',x)==None:
                x=np.nan
            elif not re.search(u'缓慢',x)==None:
                x=12.0
            elif not re.search(u'急促',x)==None:
                x=24.0
            elif not re.search(u'粗糙',x)==None:
                x=21.0
            else:
                x=np.nan
    return x


def deal_2302(x):
    try:
        x=str(x)
        if x=='亚健康' or x=='肥健康':
            x=1
        elif x=='疾病'or x=='y疾病':
            x=2
        else:
            x=0
    except:
        pass
    return x

def deal_3399(x):
    x=str(x)
    if x=='黄色':
        x=0
    elif x=='淡黄色':
        x=1
    else:
        x=2
    return x

def deal_3195(x):
    try:
        x=str(x)
        if not re.search(u'\-|阴性',x)==None:
            if x==u'-':
                x=0
            elif not re.search(u'\+-',x)==None:
                x=1
            else:
                x=0
        elif x==u'+' or not re.search(u'阳性',x)==None:
            x=2
        elif x==u'++':
            x=3
        elif x==u'+++':
            x=4 
        elif x==u'++++':
            x=5 
        else:
            x=1
    except:
        x=1
    return x

def deal_3196(x):
    try:
        if not re.search(u'\-|阴性|Normal|正常',x)==None:
            if x==u'-':
                x=0
            elif not re.search(u'\+-',x)==None:
                x=1
            else:
                x=0
        elif x==u'+' or not re.search(u'阳性',x)==None:
            x=2
        elif x==u'++':
            x=3
        elif x==u'+++':
            x=4 
        else:
            x=1
    except:
        x=1
        pass
    return x