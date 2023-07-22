#导入库包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier

# 计算WOE和IV
def cal_woe(X,Y,bins):
    """ 计算卡方值
    Args:
        Y: 标签值
        X：变量值
        bins: 变量X的分箱
    Returns:
        result,woe,iv
    """
    
    total_bad = Y.sum()    # 总的坏样本数
    total_good = Y.count() - total_bad   # 总的好样本数
    
    df = pd.DataFrame({'X':X, 'Y':Y, 'bins':pd.cut(X,bins)})

    #分组计算好坏样本数
    bad=df.query("Y==1").groupby('bins')['Y'].count().reset_index(name='badnum')
    good=df.query("Y==0").groupby('bins')['Y'].count().reset_index(name='goodnum')

    result=pd.merge(bad,good,on='bins')


    result['badrate'] = result['badnum'] / total_bad
    result['goodrate'] = result['goodnum'] / total_good
    result['woe'] = np.log(result['goodrate'] / result['badrate'])
    result['iv']=(result['goodrate'] - result['badrate']) * result['woe']
    iv = result['iv'].sum()

    # woe值列表
    woe = list(result['woe'])
    
#     print("变量{0}的iv值为：{1}".format(X.name,round(iv,2)))
    
    plt.figure(figsize=(8,6))
    plt.bar(range(result.shape[0]),result['woe'],tick_label=result['bins'],label='WOE')
    plt.title("{0}(iv={1})".format(X.name,round(iv,2)))
    plt.xticks(rotation=30)
    plt.rc('legend', fontsize=10)
    plt.legend(loc='best')
    plt.show()
    
    return result,woe,iv

def dt_bins(X,Y,max_leaf_num): 
    """利用决策树获得最优分箱的边界值"""  
    """
    基于CART算法的连续变量最优分箱，实现步骤如下：
   （1）给定连续变量 V，对V中的值进行排序；
   （2）依次计算相邻元素间中位数作为二值划分点的基尼指数；
   （3）选择最优（划分后基尼指数下降最大）的划分点作为本次迭代的划分点；
   （4）递归迭代步骤2-3，直到满足停止条件。（一般是以划分后的样本量作为停止条件，比如叶子节点的样本量>=总样本量的10%）
   """
    
    """
    :param X: 待分箱特征
    :param Y: 目标变量
    :param max_leaf_num: 分箱数
    
    :return: 统计值、分箱边界值列表、woe值、iv值
    """
    bins = []  
    x = X.values  
    y = Y.values 
    clf = DecisionTreeClassifier(criterion='entropy',  # 信息熵最小化准则划分 
                                 max_leaf_nodes=max_leaf_num,  # 最大叶子节点数  
                                 min_samples_leaf = 0.05)  # 叶子节点样本数量最小占比 
    clf.fit(x.reshape(-1,1),y)  # 训练决策树 
     
    n_nodes = clf.tree_.node_count  
    children_left = clf.tree_.children_left  
    children_right = clf.tree_.children_right  
    threshold = clf.tree_.threshold  
     
    for i in range(n_nodes): 
        if children_left[i] != children_right[i] : # 获的决策时节点上的划分边界 
            bins.append(threshold[i]) 
    bins.sort() 
    min_x = x.min()-0.1
    max_x = x.max() + 0.1 # 加0.1是为了考虑后续groupby操作时, 能包含特征最大值得样本 
    bins=[min_x]+bins
    bins = bins +[max_x]
    
    result,woe,iv=cal_woe(X,Y,bins)
    
    return result,bins,woe,iv
    
    
    
def spearman_bins(X,Y,n):
    """
    :param Y: 目标变量
    :param X: 待分箱特征
    :param n: 分箱数初始值
    :return: 统计值、分箱边界值列表、woe值、iv值
    """
    r = 0    # 初始值
    total_bad = Y.sum()    # 总的坏样本数
    total_good = Y.count() - total_bad    # 总的好样本数
    
    # 分箱过程
    while np.abs(r) < 1:    
        df1 = pd.DataFrame({'X':X, 'Y':Y, 'bin':pd.qcut(X, n, duplicates='drop')})    # qcut():基于量化的离散化函数
        df2 = df1.groupby('bin')
        r, p = stats.spearmanr(df2.mean().X, df2.mean().Y)
        n = n - 1
        
        if n==1:
            break
        else:
            continue
    bins = []
    bins.append(-np.inf)
    for i in range(1, n+1):
        qua = X.quantile(i / (n+1))
        bins.append(round(qua, 6))
    bins.append(np.inf)
    bins =list(np.unique(bins))
    
    
    #计算woe和iv
    result,woe,iv=cal_woe(X,Y,bins)
    
    return result,bins,woe,iv