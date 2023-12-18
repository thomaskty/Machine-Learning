import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,roc_curve,auc
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import warnings
import matplotlib.pyplot as plt 
import seaborn as sns

warnings.filterwarnings('ignore')
pd.set_option('display.float_format',lambda x : '%.5f' % x)
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',1000)

from functools import * 

def optimal_nobins(data):
    """
    The Freedman-Diaconis rule is very robust and works well in practice. T
    he bin-width is set to h=2*IQR*n-1/3. 
    So the number of bins is $(max-min)/h$, 
    where n is the number of observations, 
    max is the maximum value 
    and min is the minimum value
    adjustment : instead of dividing max_min we divide iqr(to reduce the number of bins)
    """
    n = len(data)
    min_max = data.max()-data.min()
    q3, q1 = np.percentile(data, [75 ,25])
    iqr = q3 - q1
    h = 2*(iqr*(n**(-1/3)))
    return {'iqr':int(iqr/h),'min_max':int(min_max/h)}

def analyse_feature(df,feat1,feat2,relation,bins=False):
    """
    relation argument values : ['cat2cat','con2con','cat2con','con2cat']
    bins values : integer,optimal,list of tuples 
    """
    if relation.lower() in ['cat2cat','con2cat']:
        print('feature analysis output is a dictionary with keys counts,total_ratio,bin_ratio')
    try:
        temp = df[[feat1,feat2]].copy()
        if relation.lower()=='cat2cat':
            output_dict = dict() 
            feat2_labels = temp[feat2].value_counts().index
            for i in feat2_labels:
                temp[i] = np.where(temp[feat2]==i,1,0)
            output1 = temp.groupby(feat1).agg({i:'sum' for i in feat2_labels})
            output1.loc['total'] = output1.sum()
            
            output2 = (output1/temp.shape[0])*100
            output3 = output1.div(output1.sum(axis=1), axis=0)
            output3 = (output3*100)
            
            output1['total'] = output1.sum(axis=1)
            output_dict['counts'] = output1 
            
            output2['total'] = output2.sum(axis=1)
            for j in output2.columns:
                output2[j] = output2[[j]].applymap(lambda x: "{0:.2f}%".format(x))
            output_dict['total_ratio'] = output2
            
            output3['total'] = output3.sum(axis=1)
            for j in output3.columns:
                output3[j] = output3[[j]].applymap(lambda x: "{0:.2f}%".format(x))
            output_dict['bin_ratio'] = output3 

            return output_dict 
            
        elif relation.lower()=='con2con':
            # creating bins for feature a if not given
            if type(bins)==str:
                nob = optimal_nobins(temp[feat1])['iqr']
                temp[feat1+'_bins'] = pd.qcut(temp[feat1],q = nob,duplicates = 'drop')
            elif type(bins)==int:
                temp[feat1+'_bins'] = pd.qcut(temp[feat1],q = bins,duplicates = 'drop')
            elif type(bins)==list:
                bins = pd.IntervalIndex.from_tuples(bins)
                temp[feat1+'_bins'] = pd.cut(temp[feat1],bins=bins,duplicates = 'drop')

            output1 = temp.groupby(feat1+'_bins').describe()[feat2]
            output1['ratio'] = output1['count']/output1['count'].sum()
            output1['ratio'] = output1[['ratio']].applymap(lambda x: "{0:.2f}%".format(x*100))
            
            return output1
        
        elif relation.lower()=='cat2con':
            output1 = temp.groupby(feat1).describe()[feat2]
            output1['ratio'] = output1['count']/output1['count'].sum()
            output1['ratio'] = output1[['ratio']].applymap(lambda x: "{0:.2f}%".format(x*100))
            return output1
            
        elif relation.lower()=='con2cat':
            output_dict = dict()

            if type(bins)==str:
                nob = optimal_nobins(temp[feat1])['iqr']
                temp[feat1+'_bins'] = pd.qcut(temp[feat1],q = nob,duplicates = 'drop')
            elif type(bins)==int:
                temp[feat1+'_bins'] = pd.qcut(temp[feat1],q = bins,duplicates = 'drop')
            elif type(bins)==list:
                bins = pd.IntervalIndex.from_tuples(bins)
                temp[feat1+'_bins'] = pd.cut(temp[feat1],bins=bins,duplicates = 'drop')
            
            feat2_labels = temp[feat2].value_counts().index
            for i in feat2_labels:
                temp[i] = np.where(temp[feat2]==i,1,0)
            output1 = temp.groupby(feat1+'_bins').agg({i:'sum' for i in feat2_labels})
            output1.loc['total'] = output1.sum()
            
            output2 = (output1/temp.shape[0])*100
            output3 = output1.div(output1.sum(axis=1), axis=0)
            output3 = (output3*100)
            
            output1['total'] = output1.sum(axis=1)
            output_dict['counts'] = output1
            
            output2['total'] = output2.sum(axis=1)
            for j in output2.columns:
                output2[j] = output2[[j]].applymap(lambda x: "{0:.2f}%".format(x))
            output_dict['total_ratio'] = output2
            
            output3['total'] = output3.sum(axis=1)
            for j in output3.columns:
                output3[j] = output3[[j]].applymap(lambda x: "{0:.2f}%".format(x))
            output_dict['bin_ratio'] = output3
            
            return output_dict 
    except Exception as e:
        print('Exception  : ',e)

def woe_iv(data,feature,target,events,bins,continuous):
    """
    target may contain more than two labels
    events,non events in the target columns 
    woe = log2(%events/%nonevents)
    iv = woe * (% events-% non events)
    """
    temp = data[[feature,target]].copy()
    if continuous:
        if type(bins)==str:
            temp[feature+'_bins'] = pd.qcut(
                temp[feature],
                q = optimal_nobins(temp[feature])['iqr'],
                duplicates='drop'
            )
        elif type(bins)==list:
            bins = pd.IntervalIndex.from_tuples(bins)
            temp[feature+'_bins'] = pd.cut(temp[feature],bins)
        elif type(bins)==int:
            temp[feature+'_bins'] = pd.qcut(temp[feature],q = bins,duplicates='drop')

        feature = feature+'_bins'

    target_labels = temp[target].value_counts().index 
    temp[events] = np.where(temp[target]==events,1,0)
    temp = temp.groupby(feature).agg({events:['sum','count']})
    temp['%events'] = temp[events]['sum']/temp[events]['sum'].sum()
    temp['#nonevents'] = temp[events]['count']-temp[events]['sum']
    temp['%nonevents'] = temp['#nonevents']/temp['#nonevents'].sum()
    temp['woe'] = np.log((temp['%events']) / (temp['%nonevents'] ))
    temp['IV'] = temp['woe']* (temp['%events'] - temp['%nonevents'])
    return temp

def thresh_analysis(model,x,y):
    cols = ['thresh','precision','recall','f1','accuracy','tp','fp','tn','fn']
    output = pd.DataFrame(columns=cols)
    thresh_list = [i/100 for i in list(range(30,80,1))]
    probs = model.predict_proba(x)[:,1]
    for i in range(len(thresh_list)):
        y_pred = np.where(probs>=thresh_list[i],1,0)
        cf = confusion_matrix(y,y_pred)
        tn,fp,fn,tp = cf[0,0],cf[0,1],cf[1,0],cf[1,1]
        p = tp/(tp+fp)
        r = tp/(tp+fn)
        f1 = (2*p*r)/(p+r)
        accuracy = (tp+tn)/(tn+fp+fn+tp)
        row = [thresh_list[i],p,r,f1,accuracy,tp,fp,tn,fn]
        output.loc[i] = row
    return output

def deciles(x,y,model):
    output = pd.DataFrame({'target':y,'proba':model.predict_proba(x)[:,1]})
    output['decile'],probability_cuts = pd.qcut(
        output['proba'],q = 10,
        duplicates='drop',retbins=True
    )
    return output 

def aggregate(data):
    temp = data.copy()
    temp = temp.groupby('decile').agg({'target':['count',sum],'proba':['min','max']})
    temp.sort_index(ascending = False,inplace = True)
    temp['cust%'] = temp[('target','count')]/temp[('target','count')].sum()
    temp['cum_cust%'] = temp['cust%'].cumsum()
    temp['capture'] = temp[('target','sum')]/temp['target']['count'].sum()
    temp['gain'] = temp[('target','count')]/temp[('target','sum')].sum()
    temp['cum_gain'] = temp['gain'].cumsum()
    temp['lift'] = temp['cum_gain']/temp['cum_cust%'].cumsum() 
    return temp.sort_index(ascending = False)

def prediction_frame(x,y,model,threshold):
    pred_df = pd.concat([x,y],axis=1)
    pred_df['proba'] = model.predict_proba(x)[:,1]
    pred_df['prediction'] = np.where(pred_df['proba']>threshold,1,0)
    pred_df['tp'] = np.where(((pred_df[y.name]==1)&(pred_df['prediction']==1)),1,0)
    pred_df['tn'] = np.where(((pred_df[y.name]==0)&(pred_df['prediction']==0)),1,0)
    pred_df['fp'] = np.where(((pred_df[y.name]==0)&(pred_df['prediction']==1)),1,0)
    pred_df['fn'] = np.where(((pred_df[y.name]==1)&(pred_df['prediction']==0)),1,0)
    pred_df['decile'] = pd.qcut(pred_df['proba'],q=10,duplicates='drop')
    return pred_df 

def population_stability_index(population,sample,model,name):
    """
    name represents the dataset under consideration (test,validaiton) 
    """
    def get_pred_fr(x,model):
        pred_df = x.copy()
        pred_df['proba'] = model.predict_proba(x)[:,1]
        return pred_df 
    
    pred_df_population = get_pred_fr(population,model)
    pred_df_sample = get_pred_fr(sample,model)
    pred_df_population['psi_decile'],proba_cuts = pd.qcut(
        pred_df_population['proba'],
        q = 10,
        duplicates='drop',
        retbins=True
    )
    proba_cuts[0] = 0 
    proba_cuts[-1] = 1
    pred_df_population['psi_decile'] = pd.cut(pred_df_population['proba'],bins = proba_cuts)
    pred_df_sample['psi_decile'] = pd.cut(pred_df_sample['proba'],bins = proba_cuts)
    score_b = pred_df_sample.groupby('psi_decile').agg({
        'proba':'count'
    }).sort_index(ascending = False)
    score_a = pred_df_population.groupby('psi_decile').agg({
        'proba':'count'
    }).sort_index(ascending = False)
    score_a.columns = ['population_A']
    score_b.columns = [name]
    psi_df = pd.concat([score_a,score_b],axis=1).sort_index()
    psi_df['A%'] = np.round((psi_df['population_A']/psi_df['population_A'].sum())*100)
    psi_df['B%'] = np.round((psi_df[name]/psi_df[name].sum())*100)
    psi_df['A% - B%'] = psi_df['A%']- psi_df['B%']
    psi_df['log(A%/B%)'] = np.log(psi_df['A%']/psi_df['B%'])

    psi_df['A%'] = psi_df[['A%']].applymap(lambda x: "{0:.0f}%".format(x))
    psi_df['B%'] = psi_df[['B%']].applymap(lambda x: "{0:.0f}%".format(x))

    psi_df['psi'] = (psi_df['A% - B%']/100)*psi_df['log(A%/B%)']
    psi_df['A% - B%'] = psi_df[['A% - B%']].applymap(lambda x: "{0:.0f}%".format(x))

    psi_df.sort_index(ascending = False,inplace = True)
    
    return psi_df

def auc_gini(x,y,model):
    preds = model.predict_proba(x)[:,1]
    fpr,tpr,thresholds = roc_curve(y,preds)
    auc_ = auc(fpr,tpr)
    gini = (2*auc_)-1
    return {'auc':auc_,'gini':gini}

def error_analysis_feature_bins(x,y,model,t,f,bins):
    """
    Dividing the feature values into bins, showing the model performance in  each bin.
    t - threshold
    f - feature_name
    bins - [optimal,integer,actual_bins]
    """
    pred_df = prediction_frame(x,y,model,t)
    if type(bins)==int:
        pred_df[f]= pd.qcut(pred_df[f],q = bins,duplicates = 'drop')
    elif type(bins)==str:
        pred_df[f] = pd.qcut(pred_df[f],q = optimal_nobins(pred_df[f])['iqr'],duplicates='drop')
    elif type(bins)==list:
        bins = pd.IntervalIndex.from_tuples(bins)
        pred_df[f] = pd.cut(pred_df[f],bins = bins)

    mc = pred_df.groupby(f).agg({i:sum for i in ['tn','tp','fn','fp']}).reset_index()
    mc['+ preds'] = mc[['tp','fp']].sum(axis=1)
    mc['- preds'] = mc[['tn','fn']].sum(axis=1)
    mc['+ actual'] = mc[['tp','fn']].sum(axis=1)
    mc['- actual'] = mc[['tn','fp']].sum(axis=1)
    mc['true preds'] = mc[['tn','tp']].sum(axis=1)
    mc['false preds'] = mc[['fp','fn']].sum(axis=1)
    mc['precision'] = mc['tp']/mc['+ preds']
    mc['recall'] = mc['tp'] /mc['+ actual']
    mc['f1'] = (2*mc['precision']*mc['recall'])/(mc['precision']+mc['recall'])
    mc['accuracy'] = mc[['tp','tn']].sum(axis=1)/mc[['tp','tn','fn','fp']].sum(axis=1)
    return mc 

def get_metrices(x,y,model,thresh):
    pred_df = prediction_frame(x,y,model,thresh)
    dict_final = {}
    for i in ['tp','tn','fp','fn']:
        dict_final[i] = pred_df[i].sum()

    tp,tn = dict_final['tp'],dict_final['tn']
    fp,fn = dict_final['fp'],dict_final['fn']
    
    precision = np.round(tp/(tp+fp),2)
    recall = np.round(tp/(tp+fn),2)
    f1 = np.round((2*precision*recall)/(precision+recall),2)
    accuracy = np.round((tp+tn)/(tp+fn+fp+fn),2)
    return {'precision':precision,'recall':recall, 'f1':f1,'accuracy':accuracy}


def confusion_matrix_formatted(x,y,model,name,thresh):
    pred_df = prediction_frame(x,y,model,thresh)
    dict_final = {}
    for i in ['tp','tn','fp','fn']:
        dict_final[i] = pred_df[i].sum()
    d = pd.DataFrame(index = [0,1],columns= pd.MultiIndex.from_tuples([
        ('{}_{}'.format(name,thresh),'Predicted',0),
        ('{}_{}'.format(name,thresh),'Predicted',1)
    ]))
    d.index.name = 'Actual'
    d.loc[0] = [dict_final['tn'],dict_final['fp']]
    d.loc[1] = [dict_final['fn'],dict_final['tp']]
    d.loc['Total'] = [int(i) for i in d['{}_{}'.format(name,thresh)]['Predicted'].sum()]
    d[('{}_{}'.format(name,thresh),'Predicted','total')] = d.sum(axis=1)
    return d

def calculate_psi(expected,actual,buckettype='bins',buckets=10,axis=0):
    def psi(expected_array,actual_array,buckets):
        def scale_range(input_,min_,max_):
            input_ +=-(np.min(input_))
            input_ /=np.max(input_)/(max_-min_)
            input_ +=min
            return input_
        breakpoints = np.arange(0,buckets+1)/(buckets)*100
        if buckettype=='bins':
            breakpoints = scale_range(breakpoints,np.min(expected_array),np.max(expected_array))
        elif buckettype=='quantiles':
            breakpoints = np.stack([np.percentile(expected_array,b) for b in breakpoints])
        expected_percents = np.histogram(expected_array,breakpoints)[0]/len(expected_array)
        actual_percents = np.histogram(actual_array,breakpoints)[0]/len(actual_array)
        def sub_psi(e_perc,a_perc):
            if a_perc ==0:
                a_perc= 0.0001
            if e_perc ==0:
                e_perc=0.0001
            value = (e_perc - a_perc)*np.log(e_perc/a_perc)
            return (value)
        psi_value = np.sum(sub_psi(
            expected_percents[i],actual_percents[i]
        ) for i in range(0,len(expected_percents)))

        return (psi_value)
    if len(expected.shape)==1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_value = np.empty(expected.shape[axis])
    for i in range(0,len(psi_values)):
        if len(psi_values)==1:
            psi_values = psi(expected,actual,buckets)
        elif axis==0:
            psi_values[i] = psi(expected[:,i],actual[:,i],buckets)
        elif axis==1:
            psi_values[i] = psi(expected[i,:],actual[i,:],buckets)
    return (psi_values)


def calculate_psi_category(initial,new):
    df1 = pd.DataFrame(initial.value_counts(normalize = True).reset_index())
    df2 = pd.DataFrame(new.value_counts(normalize = True).reset_index())
    df = pd.merge(df1,df2,how = 'left',on='index')
    df.columns = ['index','initial_perc','new_perc']
    df.fillna(0.0001,inplace = True)
    df['PSI'] = (df['new_perc']-df['initial_perc'])*np.log(df['new_perc']/df['initial_perc'])
    return np.round(np.sum(df['PSI']),5)

def psi_alert_label(psi):
    return np.where(psi<0.1,'low',np.where(psi>=0.2,'high','medium'))

def calculate_covariate_drift(old_data,new_data):
    li_common_cols = old_data.columns.intersection(new_data.columns)
    old_data = old_data[li_common_cols]
    new_data = new_data[li_common_cols]
    num_cols = old_data._get_numeric_data().columns
    obj_cols = list(set(li_common_cols)-set(num_cols))
    t = pd.DataFrame(columns = ['PSI Value','Drift Level'])
    
    for var in num_cols:
        psi = calculate_psi(old_data[var],new_data[var],buckettype='quantiles',buckets = 10,axis=1)
        t.loc[var] = [psi,psi_alert_label(psi)]
    for  var in obj_cols:
        psi = calculate_psi_category(old_data[var],new_data[var])
        t.loc[var] = [psi,psi_alert_label(psi)]
    return t


def detect_model_drift(model,data_old,y_old,data_new,y_new,tolerance,thresh):
    model_score_old = get_metrices(data_old,y_old,model,thresh)
    f1_old = model_score_old['f1']
    precision_old = model_score_old['precision']
    recall_old = model_score_old['recall'] 

    new_scores = get_metrices(data_new,y_new,model,thresh)
    f1_new = new_scores['f1']
    precision_new = new_scores['precision']
    recall_new = new_scores['recall']

    precision_flag = False
    recall_flag = False
    f1_flag = False
    if f1_new<(f1_old-f1_old*tolerance):
        f1_flag=True
    if precision_new <(precision_old-precision_old*tolerance):
        precision_flag = True
    if recall_new <(recall_old-recall_old*tolerance):
        recall_flag = True
    t = pd.DataFrame(columns = ['Drift Observed'])
    t.loc['f1'] = [f1_flag]
    t.loc['recall'] = [recall_flag]
    t.loc['precision'] = [precision_flag]
    return t 

def calculate_residual_drift(model,x_new,y_new,x_old,y_old):
    """
    for regression models
    """
    old_predictions = model.predict(x_old)
    new_predictions = model.predict(x_new)
    residuals_old = y_old - old_predictions 
    residuals_new = y_new - new_predictions

    psi = calculate_psi(residuals_old,residuals_new,buckettype='quantiles',buckets=10,axis=1)
    output = pd.DataFrame(columns = ['psi_value','drift_level'])
    output.loc['observed_residual_shift'] = [psi,psi_alert_label(psi)]
    return output

def chi_selection(df,target_name):
    """
    chi-square test of independence.
    smaller p value tells us to reject the null 
    null hypothesis : two feature are independent
    """
    from scipy import stats
    temp_df = df.drop(target_name,axis=1)
    temp_y = df[target_name]
    temp_new = pd.DataFrame()
    try:
        object_cols = df.describe(include='O').T.index.tolist()
    except:
        object_cols = []
    
    cols_categorized = [i for i in df.columns if i not in object_cols+[target_name]]
    for j in cols_categorized:
        nob = optimal_nobins(temp_df[j])['iqr']
        temp_new[j+'_bin'] = pd.qcut(temp_df[j],q = nob,duplicates='drop')
    X,y = temp_new,temp_y
    p_values = [stats.chi2_contingency(pd.crosstab(X[i],y))[1] for i in X.columns]
    out = pd.DataFrame({'feature':X.columns,'p_value':p_values})
    out = out.sort_values(by = 'p_value',ascending=True).set_index('feature')
    return out

def entropy(values,probability=False):
    if probability:
        return -np.sum(values*np.log2(values))
    else:
        unique, counts = np.unique(values, return_counts=True)
        probs = counts / len(values)
        return -np.sum(probs * np.log2(probs))

def compute_probabilities(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / np.sum(counts)
    out = dict(zip(unique_labels, probabilities))
    return out

def cross_entropy(p, q, probabilities=False,verbose = True):
    """
    p * log(1/q) 
    """
    if not probabilities:
        labels_p = list(np.unique(p))
        labels_q = list(np.unique(q))
    
        if len(labels_q)>len(labels_p):
            final_labels = labels_q
        else:
            final_labels = labels_p
        p = compute_probabilities(p)
        q = compute_probabilities(q)
        for j in final_labels:
            if j not in p.keys():
                p[j] = 0 
            if j not in q.keys():
                q[j] = 0
        if verbose:
            print(p)
            print(q)
            print(final_labels)
            
        cross_entropy_measure = []
        for labs in final_labels:
            if q[labs]!=0:
                cross_entropy_measure.append(p[labs]*np.log2(1/q[labs]))
            else:
                cross_entropy_measure.append(0)
    else:
        p = np.array(p)
        q = np.array(q)
        
        cross_entropy_measure = p * np.log2(1/q)
        cross_entropy_measure[np.isinf(cross_entropy_measure)] = 0
        
    if verbose:
        print(cross_entropy_measure)
        
    return np.sum(cross_entropy_measure)

def joint_entropy(x, y):
    x_values, x_counts = np.unique(x, return_counts=True)
    y_values, y_counts = np.unique(y, return_counts=True)
    joint_counts = np.zeros((len(x_values), len(y_values)), dtype=int)
    for i in range(len(x)):
        x_idx = np.where(x_values == x[i])[0][0]
        y_idx = np.where(y_values == y[i])[0][0]
        joint_counts[x_idx, y_idx] += 1
    joint_probs = joint_counts / len(x)
    return -np.sum(joint_probs * np.log2(joint_probs + (joint_probs == 0)))


def mutual_information_value(x,y,verbose=False):
    """
    p(x,y) * log( p(x,y)/ (p(x)*q(x)))
    """
    try:
        x.name='feature'
        y.name='target'
    except:
        pass
    if type(y)!=pd.core.series.Series:
        y = pd.Series(y,name='target')
        
    if type(x)!=pd.core.series.Series:
        x = pd.Series(x,name='feature')
    output = pd.crosstab(x,y,margins=True,margins_name='total')
    if verbose:
        print(output)
    
    output /=len(x)
    if verbose:
        print(output)
    
    def compute_mi(join,marginals):
        return join*np.log2((join/reduce(lambda x,y:x*y,marginals)))
    
    total_mi = []
    rows = len(output.index)-1
    cols = len(output.columns)-1
    
    for r in range(rows):
        for c in range(cols):
            value = output.iloc[r,c]
            # print(rows-1,c)
            marg_a,marg_b = output.iloc[r,cols],output.iloc[rows,c]
            if verbose:
                print(value,[marg_a,marg_b],[r,c])
            if value!=0:
                mi_value = compute_mi(value,[marg_a,marg_b])
            else:
                mi_value=0
            if mi_value==np.inf:
                mi_value=0
            total_mi.append(mi_value)
    if verbose:
        print(total_mi)
    return np.sum(total_mi)

def mutual_information_frame(df,target_name):
    """
    applying mutual_information_value function on all the columns in a df
    """
    from scipy import stats
    temp_df = df.drop(target_name,axis=1)
    temp_y = df[target_name]
    temp_new = pd.DataFrame()
    try:
        object_cols = df.describe(include='O').T.index.tolist()
    except:
        object_cols = []
    
    cols_categorized = [i for i in df.columns if i not in object_cols+[target_name]]
    for j in cols_categorized:
        nob = optimal_nobins(temp_df[j])['iqr']
        temp_new[j+'_bin'] = pd.qcut(temp_df[j],q = nob,duplicates='drop')
    X,y = temp_new,temp_y
    out_dict = {'features':X.columns,'mi_score':[]}
    for i in X.columns:
        out_dict['mi_score'].append(mutual_information_value(X[i],y))
    return pd.DataFrame(out_dict)


def kl_divergence(p, q, probabilities=False,verbose = True):
    """
    kl(p|q) = p(x) * log(p(x)/q(x))
    """
    if not probabilities:
        labels_p = list(np.unique(p))
        labels_q = list(np.unique(q))
    
        if len(labels_q)>len(labels_p):
            final_labels = labels_q
        else:
            final_labels = labels_p
        p = compute_probabilities(p)
        q = compute_probabilities(q)
        for j in final_labels:
            if j not in p.keys():
                p[j] = 0 
            if j not in q.keys():
                q[j] = 0
        if verbose:
            print(p)
            print(q)
            print(final_labels)
        kl_divergence_measure = []
        for labs in final_labels:
            if q[labs]!=0:
                kl_divergence_measure.append(p[labs]*np.log2(p[labs]/q[labs]))
            else:
                kl_divergence_measure.append(0)
    else:
        p = np.array(p)
        q = np.array(q)
        
        kl_divergence_measure = p * np.log2(p / q)
        kl_divergence_measure[np.isinf(kl_divergence_measure)] = 0
        
    if verbose:
        print(kl_divergence_measure)
        
    return np.sum(kl_divergence_measure)

def plot_evaluation_curves(y_true, y_scores, title_pr='Precision-Recall Curve', title_roc='ROC Curve'):
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    average_precision = auc(recall, precision)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plotting side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Precision-Recall Curve
    axes[0].plot(recall, precision, color='blue',marker='*', label='Precision-Recall Curve', markersize=0.1)
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title(f'{title_pr} (AUC={average_precision:.2f})')
    axes[0].legend()

    # ROC Curve
    axes[1].plot(fpr, tpr, color='red', label='ROC Curve', marker='*',markersize=0.1)
    axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random', alpha=0.5)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate (Recall)')
    axes[1].set_title(f'{title_roc} (AUC={roc_auc:.2f})')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    
    
    
