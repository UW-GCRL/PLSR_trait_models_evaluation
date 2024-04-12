import numpy as np
import pandas as pd
from scipy import stats
import json
from sklearn.model_selection import LeaveOneOut,KFold,cross_val_score, train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import verde as vd


"""
Function lists: 
(1) R2; (2) NSE calculation; (3) PLSR VIP calculation; (4) extract leaf spectra and leaf traits from compiled dataset; 
(5) extrat seasonal measurements from compiled dataset;(6) Units for leaf traits;(7) random cross-validation; (8) Spatial CV; 
(9) Site_LOO CV; (10) site_extrapolation; (11) Cross-PFTs validation; (12) PFT_LOO CV;
(13) cross sites for each PFT; (14) PFT extropolation; (15) random_temporal_CV; (16) temporal_CV; 
(17) season_LOO CV; (18) cross_sensors CV.

"""
def rsquared(x, y): 
    """Return the metriscs coefficient of determination (R2)
    Parameters:
    -----------
    x (numpy array or list): Predicted variables
    y (numpy array or list): Observed variables
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y) 
    a = r_value**2
    return a
def nse(predictions, targets):
    """Return Nash–Sutcliffe model efficiency coefficient (NSE), which is used to assess the performance predictive models
    Parameters:
    -----------
    predictions (numpy array or list): Predicted variables
    targets (numpy array or list): Observed variables
    """
    return (1-(np.sum((predictions-targets)**2)/np.sum((targets-np.mean(targets))**2)))
def vip(x, y, model):
    """Return the Variable Importance in Projection (VIP) metric for trained PLSR model.
    Parameters:
    -----------
    x (numpy array, shape = (len(x),num_bands)): the training reflectance
    y (numpy array, shape  = len(x),1): the training variables
    model: the trained PLSE model.
    """
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    
    m, p = x.shape
    _, h = t.shape
    vips = np.zeros((p,))

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
    return vips

def data_extraction(tr, dataset):
    """Return the paired leaf spectra and traits values for the each trait samples.
    Parameters
    ------
    tr (str): the trait name in the compiled dataset ("Chla+b", "Ccar", "EWT" or "LMA").
    dataset: the path of the compiled dataset.
    """
    df = pd.read_csv(dataset)
    df = df[df[tr]>0]
    df.reset_index(drop = True, inplace = True)
    print(f"{tr} {len(df)} samples")
    leaf_reflectance, leaf_traits = df.loc[:, "450":"2400"], df.loc[:, "Dataset ID":]
    X, y = leaf_reflectance, leaf_traits 
    return X, y

def seasonal_data_extraction(dataset):
    """ Return the seasonal dataset in the compiled dataset
    Parameters:
    -----------
    dataset: the path of the compiled dataset.
    """
    df = pd.read_csv(dataset)
    # Extract seasonal data.
    df = df[(df["Dataset ID"]=="Dataset#3")|(df["Dataset ID"]=="Dataset#4")|(df["Dataset ID"]=="Dataset#8")]
    df["Sample date"] = df["Sample date"].astype(int).astype(str)
    # Define the DOY based on the sample dates
    df["DOY"] = [datetime.strptime(x, "%Y%m%d").timetuple().tm_yday for x in df["Sample date"].tolist()]
    # Divide the seasonality based on the DOY for specific collected datasets
    df.loc[(df['Dataset ID']=='Dataset#3')&((df['DOY']<165)|(df['DOY']==165)),'season']='early growing season'
    df.loc[(df['Dataset ID']=='Dataset#3')&((df['DOY']>240)|(df['DOY']==240)),'season']='post-peak season'
    df.loc[(df['Dataset ID']=='Dataset#3')&(df['DOY']>165)&(df['DOY']<240),'season']='peak growing season'

    df.loc[(df['Dataset ID']=='Dataset#4')&((df['DOY']<151)|(df['DOY']==151)),'season']='early growing season'
    df.loc[(df['Dataset ID']=='Dataset#4')&((df['DOY']>243)|(df['DOY']==243)),'season']='post-peak season'
    df.loc[(df['Dataset ID']=='Dataset#4')&(df['DOY']>151)&(df['DOY']<243),'season']='peak growing season'

    df.loc[(df['Dataset ID']=='Dataset#8')&((df['DOY']<175)|(df['DOY']==175)),'season']='early growing season'
    df.loc[(df['Dataset ID']=='Dataset#8')&((df['DOY']>243)|(df['DOY']==243)),'season']='post-peak season'
    df.loc[(df['Dataset ID']=='Dataset#8')&(df['DOY']>175)&(df['DOY']<243),'season']='peak growing season'
    df.reset_index(drop = True, inplace = True)
    # Extract leaf reflectrance and leaf traits
    X, y = df.loc[:, "450":"2400"], df.loc[:, "Dataset ID":]
    return X, y

def trait_units(tr):
    """Return the unit of different traits used in this study.
    Parameters:
    -----------
    tr (str): trait name ("Chla+b", "Ccar", "EWT" or "LMA")
    """
    unit = {"Chla+b":"µg/cm²", "Ccar":"µg/cm²", "EWT":"g/m²", "LMA":"g/m²"}
    return unit[tr]

def random_CV(X,y,tr,n_splits,n_iterations):
    """Random cross-validation of PLSR model for estimating leaf traits.
    Parameters:
    -----------
    X (numpy array): leaf spectra data used for PLSR modeling
    y (numpy array): leaf trait data used for PLSR modeling
    tr (str): trait name ("Chla+b", "Ccar", "EWT" or "LMA")
    n_splits (int): number of splits to divide the input data.
    n_iterations (int): How many iterations to train the PLSR model.

    Output files:
    -----------
    (1) Trained PLSR models in *.json format.
    (2) Leaf trait predictions in *.csv format.
    (3) PLSR VIP metric in *.csv format.
    (4) PLSR coefficients in *.csv format.
    (5) sample size files for each fold in *.csv format.
    """
    PRESS = []
    vip_score = pd.DataFrame(np.zeros(shape = (n_splits, X.shape[1])),columns = X.columns)
    plsr_coef = pd.DataFrame(np.zeros(shape = (n_splits, X.shape[1])),columns = X.columns)
    sample_size = pd.DataFrame(np.zeros(shape = (n_splits, 2)),columns = ['train_samples','test_samples'])

    k = 0
    saved_models = {"Trait_name": tr, "Units":trait_units(tr), "Type": "PLSR","Wavelength_units": "nanometers", "Wavelengths":np.arange(450,2401,10).tolist(), "Models": {},"Description":"coefficients, mean, std and itercept need to be converted to np.array() for model implementing, then employ the following equation for trait estimation: ((leaf spectra-Mean)/Std)@Coefficients + Intercept"}
    kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
    for i in np.arange(10,X.shape[1]):
        press = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            pls = PLSRegression(n_components=i)
            pls.fit(X_train, y_train[tr])

            pred = pls.predict(X_test)
            aa = np.array(pred.reshape(-1,).tolist())
            bb = np.array(y_test[tr].tolist())

            score = np.sum((aa - bb) ** 2)
            press.append(score)
        press_mean = np.mean(press)
        PRESS.append(press_mean)
    n_components = PRESS.index(min(PRESS))+10
    print(tr, 'model: random CV_n_components:',n_components)

    var = True
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        sample_size.iloc[k] = [len(X_train),len(X_test)]

        n_iterations = n_iterations
        var_start = True
        for iteration in range(n_iterations):
            XX_train, XX_test, yy_train, yy_test = train_test_split(X_train, y_train, test_size=0.3, random_state=iteration)
            pls = PLSRegression(n_components=n_components)
            pls.fit(XX_train, yy_train[tr])

            x_mean, x_std = pls._x_mean, pls._x_std
            coef, intercept = pls.coef_, pls._y_mean

            pred = pls.predict(X_test)
            pred = pd.DataFrame(pred,columns = [f'iteration_{iteration+1}'])
            vvv = vip(XX_train, yy_train[tr], pls)

            if var_start:
                iterative_pred = pred
                x_mean_, x_std_ = x_mean, x_std
                coefficients, intercept_, vip_ = coef, intercept, vvv
                var_start = False
            else:
                iterative_pred = pd.concat([iterative_pred,pred],axis = 1)
                x_mean_, x_std_, coefficients, intercept_, vip_ = x_mean_+x_mean, x_std_+x_std, coefficients+coef, intercept_+intercept, vip_+vvv

        final_mean, final_std = x_mean_/n_iterations, x_std_/n_iterations
        final_coef, final_intercept, final_vip = coefficients/n_iterations, intercept_/n_iterations, vip_/n_iterations
        final_model = {f"fold {k+1}": {"Coefficients": final_coef.tolist(), "Mean":final_mean.tolist(), "Std": final_std.tolist(), "Itercept": final_intercept.tolist()}}
        saved_models["Models"].update(final_model)

        y_test.reset_index(drop = True, inplace = True)
        final_pred = ((X_test-final_mean)/final_std)@final_coef + final_intercept

        final_pred.columns = ['final_model_result']
        final_pred.reset_index(drop = True, inplace = True)
        res = pd.concat([y_test,final_pred,iterative_pred],axis = 1)
        res['fold'] = f'fold{k+1}'

        if var:
            df = res
            var = False
        else:
            df = pd.concat([df,res],axis = 0)

        vip_score.iloc[k] = final_vip
        plsr_coef.iloc[k] = final_coef.reshape(-1,)
        k = k+1

    with open(f'../2_results/{tr}/0_saved_models/{tr}_{n_splits} fold random CV_saved_models.json', 'w') as json_file:
        json.dump(saved_models, json_file)

    prefix = f"../2_results/{tr}/{tr}_{n_splits}fold random CV_"
    df.to_csv(f"{prefix}df.csv", index=False)
    vip_score.to_csv(f"{prefix}VIP.csv", index=False)
    plsr_coef.to_csv(f"{prefix}coefficients.csv", index=False)
    sample_size.to_csv(f"{prefix}sample_size.csv", index=False)
    return

def spatial_CV(X,y,tr,n_splits,n_iterations):
    """spatial cross-validation of PLSR model for estimating leaf traits.
    Parameters:
    -----------
    X (numpy array): leaf spectra data used for PLSR modeling
    y (numpy array): leaf trait data used for PLSR modeling
    tr (str): trait name ("Chla+b", "Ccar", "EWT" or "LMA")
    n_splits (int): number of splits to divide the input data.
    n_iterations (int): How many iterations to train the PLSR model.

    Output files:
    -----------
    (1) Trained PLSR models in *.json format.
    (2) Leaf trait predictions in *.csv format.
    (3) PLSR VIP metric in *.csv format.
    (4) PLSR coefficients in *.csv format.
    (5) sample size files for each fold in *.csv format.
    """
    PRESS = []
    vip_score = pd.DataFrame(np.zeros(shape = (n_splits, X.shape[1])),columns = X.columns)
    plsr_coef = pd.DataFrame(np.zeros(shape = (n_splits, X.shape[1])),columns = X.columns)
    sample_size = pd.DataFrame(np.zeros(shape = (n_splits, 2)),columns = ['train_samples','test_samples'])

    k = 0
    saved_models = {"Trait_name": tr, "Units":trait_units(tr), "Type": "PLSR","Wavelength_units": "nanometers", "Wavelengths":np.arange(450,2401,10).tolist(), "Models": {},"Description":"coefficients, mean, std and itercept need to be converted to np.array() for model implementing, then employ the following equation for trait estimation: ((leaf spectra-Mean)/Std)@Coefficients + Intercept"}
    coordinates = (y.Longitude, y.Latitude)
    kfold = vd.BlockKFold(n_splits = n_splits, spacing=2.0, shuffle=True, random_state=5)
    feature_matrix = np.transpose(coordinates)

    for i in np.arange(10,X.shape[1]):
        press = []
        balanced = kfold.split(feature_matrix)
        for train_index,test_index in balanced:
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            pls = PLSRegression(n_components=i)
            pls.fit(X_train, y_train[tr])
            pred = pls.predict(X_test)

            aa= np.array(pred.reshape(-1,).tolist())
            bb = np.array(y_test[tr].tolist())

            score = mean_squared_error(aa,bb)
            press.append(score)
        press_mean = np.mean(press)
        PRESS.append(press_mean)         
    n_components = PRESS.index(min(PRESS))+10
    print(tr, 'model: spatial CV_n_components:',n_components)

    var = True
    balanced = kfold.split(feature_matrix)
    for train_index,test_index in balanced:
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        sample_size.iloc[k] = [len(X_train),len(X_test)]

        n_iterations = n_iterations    ##########
        var_start = True
        for iteration in range(n_iterations):
            XX_train, XX_test, yy_train, yy_test = train_test_split(X_train, y_train, test_size=0.3, random_state=iteration)
            pls = PLSRegression(n_components=n_components)
            pls.fit(XX_train, yy_train[tr])

            x_mean, x_std = pls._x_mean, pls._x_std
            coef, intercept = pls.coef_, pls._y_mean

            pred = pls.predict(X_test)
            pred = pd.DataFrame(pred,columns = [f'iteration_{iteration+1}'])
            vvv = vip(XX_train, yy_train[tr], pls)

            if var_start:
                iterative_pred = pred
                x_mean_, x_std_ = x_mean, x_std
                coefficients, intercept_, vip_ = coef, intercept, vvv
                var_start = False
            else:
                iterative_pred = pd.concat([iterative_pred,pred],axis = 1)
                x_mean_, x_std_, coefficients, intercept_, vip_ = x_mean_+x_mean, x_std_+x_std, coefficients+coef, intercept_+intercept, vip_+vvv

        final_mean, final_std = x_mean_/n_iterations, x_std_/n_iterations
        final_coef, final_intercept, final_vip = coefficients/n_iterations, intercept_/n_iterations, vip_/n_iterations
        final_model = {f"fold {k+1}": {"Coefficients": final_coef.tolist(), "Mean":final_mean.tolist(), "Std": final_std.tolist(), "Itercept": final_intercept.tolist()}}
        saved_models["Models"].update(final_model)

        y_test.reset_index(drop = True, inplace = True)
        final_pred = ((X_test-final_mean)/final_std)@final_coef + final_intercept

        final_pred.columns = ['final_model_result']
        final_pred.reset_index(drop = True, inplace = True)
        res = pd.concat([y_test,final_pred,iterative_pred],axis = 1)
        res['fold'] = f'fold{k+1}'

        if var:
            df = res
            var = False
        else:
            df = pd.concat([df,res],axis = 0)

        vip_score.iloc[k] = final_vip
        plsr_coef.iloc[k] = final_coef.reshape(-1,)
        k = k+1

    with open(f'../2_results/{tr}/0_saved_models/{tr}_{n_splits} fold spatial CV_saved_models.json', 'w') as json_file:
        json.dump(saved_models, json_file)

    prefix = f"../2_results/{tr}/{tr}_{n_splits}fold spatial CV_"
    df.to_csv(f"{prefix}df.csv", index=False)
    vip_score.to_csv(f"{prefix}VIP.csv", index=False)
    plsr_coef.to_csv(f"{prefix}coefficients.csv", index=False)
    sample_size.to_csv(f"{prefix}sample_size.csv", index=False)
    return 

def leave_one_out_CV(X,y,tr):
    """Leave one site out for PLSR model training for estimating leaf traits.
    Parameters:
    -----------
    X (numpy array): leaf spectra data used for PLSR modeling
    y (numpy array): leaf trait data used for PLSR modeling
    tr (str): trait name ("Chla+b", "Ccar", "EWT" or "LMA")

    Output files:
    -----------
    (1) Leaf trait predictions in *.csv format.
    (2) The accuracy file that training model from one site and applying to another site.
    (3) The overall accuracy that training model from one site and applying to all the other sites.
    (4) PLSR coefficients in *.csv format.
    (5) PLSR VIP metric in *.csv format.
    (6) Benchmark accuracy that train and test the model in the same site.
    """
    res = pd.DataFrame(np.zeros(shape = (0,8)),columns = ['training_site','testing_site','train samples','test samples','R2','RMSE','NRMSE','NSE'])
    sites = []
    for i in y['Site ID'].unique():
        df = y[y['Site ID'] == i]
        if len(df) > 100:
            sites.append(i)
    sites = np.array(sites)
    
    PRESS = []
    accu = []
    RMSE = []
    NRMSE = []
    NSE = []
    
    vip_score = pd.DataFrame(np.zeros(shape = (len(sites),X.shape[1])),index = sites,columns = X.columns)
    plsr_coef = pd.DataFrame(np.zeros(shape = (len(sites),X.shape[1])),index = sites,columns = X.columns)
    k = 0
    
    loo = LeaveOneOut()
    for i in np.arange(10,X.shape[1]):
        press = []
        for test_index, train_index in loo.split(sites):
            y_test = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
            train_sites = sites[train_index]
            test_sites = sites[test_index]
            
            for j in test_sites:
                df_test = y[y['Site ID']== j]
                y_test = pd.concat([y_test,df_test])
                
            y_train = y[y['Site ID']== train_sites[0]]
            X_train = X.iloc[y_train.index]
            X_test = X.iloc[y_test.index]
    
            pls = PLSRegression(n_components=i)
            pls.fit(X_train, y_train[tr])
            pred = pls.predict(X_test)

            aa= np.array(pred.reshape(-1,).tolist())
            bb = np.array(y_test[tr].tolist())

            score = mean_squared_error(aa,bb)
            press.append(score)
        press_mean = np.mean(press)
        PRESS.append(press_mean)         
    n_components = PRESS.index(min(PRESS))+10
    print(tr, 'model: Site_LOO CV_n_components:',n_components)
    
    var = True
    loo = LeaveOneOut()
    for test_index, train_index in loo.split(sites):
        y_test = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
        train_sites = sites[train_index]
        test_sites = sites[test_index]
        
        for j in test_sites:
            df_test = y[y['Site ID']== j]
            y_test = pd.concat([y_test,df_test])
        
        y_train = y[y['Site ID']== train_sites[0]]
        X_train = X.iloc[y_train.index]
        X_test = X.iloc[y_test.index]
        
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_train, y_train[tr])
        
        vvv = vip(X_train, y_train[tr], pls)
        coef = pls.coef_.reshape(-1,)
        vip_score.iloc[k] = vvv
        plsr_coef.iloc[k] = coef
        k = k+1
        
        pred = pls.predict(X_test)
        pred = pd.DataFrame(pred,columns = ['pred'])
        pred.reset_index(drop = True, inplace = True)
        y_test.reset_index(drop = True, inplace = True)
        new_df = pd.concat([y_test,pred],axis = 1)
    
        new_df['train_site'] = train_sites[0]
        new_df['test_sites'] = new_df['Site ID']
    
        if var:
            df = new_df
            var = False
        else:
            df = pd.concat([df,new_df],axis = 0)
        
        a = new_df['pred']
        b = new_df[tr]
        
        R2 = rsquared(a,b)
        rmse = np.sqrt(mean_squared_error(a,b))
        nrmse = np.sqrt(mean_squared_error(a,b))/(b.max()-b.min())
        N_SE = nse(a, b)
        
        accu.append(R2)
        RMSE.append(rmse)
        NSE.append(N_SE)
        NRMSE.append(nrmse)
        
        for kk in new_df['Site ID'].unique():
            data = new_df[new_df['Site ID'] == kk]

            R_2 = rsquared(data['pred'],data[tr])
            r_mse = np.sqrt(mean_squared_error(data['pred'],data[tr]))
            n_rmse = np.sqrt(mean_squared_error(data['pred'],data[tr]))/(data[tr].max()-data[tr].min())
            N_S_E = nse(data['pred'],data[tr])
            
            temp = pd.DataFrame(np.array([y_train['Site ID'].unique()[0],kk,len(y_train),len(data),R_2,r_mse,n_rmse,N_S_E]).reshape(1,8),columns = ['training_site','testing_site','train samples','test samples','R2','RMSE','NRMSE','NSE'])
            res = pd.concat([res,temp])

    performance = pd.DataFrame([accu,RMSE, NRMSE, NSE]).T
    performance.columns = ['R2','RMSE','NRMSE','NSE']
    performance.index = sites
    
    prefix = f"../2_results/{tr}/{tr}_LOO_site_"
    df.to_csv(f'{prefix}df.csv',index = False)
    res.to_csv(f'{prefix}accuracy.csv',index = False)
    performance.to_csv(f'{prefix}overall_accuracy.csv')
    vip_score.to_csv(f'{prefix}vip_score.csv')
    plsr_coef.to_csv(f'{prefix}coefficients.csv')
    
    ## generate the benchmark accuracy (train and test at same sites)
    accu = []
    RMSE = []
    NRMSE = []
    NSE = []
    for i in sites:
        y1 = y[y['Site ID']==i]
        X1 = X.loc[y1.index]
        
        press = []
        for i in np.arange(10,X1.shape[1]):
            pls = PLSRegression(n_components=i)
            pls.fit(X1, y1[tr])
            pred = pls.predict(X1)
            aa = np.array(pred.reshape(-1,).tolist())
            bb = np.array(y1[tr].tolist())
            score = np.sum((aa - bb) ** 2)
            press.append(score)
        n_components = press.index(min(press))+10
        
        pls = PLSRegression(n_components=n_components)
        pls.fit(X1, y1[tr])
        pred = pls.predict(X1)
        a = np.array(pred.reshape(-1,).tolist())
        b = np.array(y1[tr].tolist())

        R2 = rsquared(a,b)
        rmse = np.sqrt(mean_squared_error(a,b))
        nrmse = np.sqrt(mean_squared_error(a,b))/(b.max()-b.min())
        N_SE = nse(a, b)

        accu.append(R2)
        RMSE.append(rmse)
        NSE.append(N_SE)
        NRMSE.append(nrmse)
    res = pd.DataFrame(np.array([accu,RMSE,NRMSE,NSE])).T
    res.columns = ['R2','RMSE','NRMSE','NSE']
    res.index = sites
    res.to_csv(f'{prefix}benchmark_accuracy.csv')
    return

def site_extrapolation(X,y,tr):
    """Exploring site extrapolation ability of PLSR models. Select one sites as the testing sites and iteratively increase the number of sites used for training.
    Parameters:
    -----------
    X (numpy array): leaf spectra data used for PLSR modeling
    y (numpy array): leaf trait data used for PLSR modeling
    tr (str): trait name ("Chla+b", "Ccar", "EWT" or "LMA")

    Output files:
    -----------
    Standard summary statistics include R2, RMSE, NRMSE and NSE for site extrapolation.
    """
    s = pd.DataFrame(y['Site ID'].value_counts())
    s = s[s['Site ID']>30]
    s = s.index.tolist()
    s.reverse()
    s = np.array(s)
    
    R2_ex_frame = pd.DataFrame(np.zeros(shape = (len(s)-1,0)))
    RMSE_ex_frame = pd.DataFrame(np.zeros(shape = (len(s)-1,0)))
    nrmse_ex_frame = pd.DataFrame(np.zeros(shape = (len(s)-1,0)))
    nse_ex_frame = pd.DataFrame(np.zeros(shape = (len(s)-1,0)))
    
    for j in s:
        target_site = j
        source_site = s[s!=target_site]
    
        accu = []
        RMSE = []
        NRMSE = []
        NSE = []
    
        for k in range(1,len(source_site)+1):
            y_train = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
            for i in range(k):
                df_train = y[y['Site ID']== source_site[i]]
                y_train = pd.concat([y_train,df_train])

            y_test = y[y['Site ID']== target_site]

            X_train = X.iloc[y_train.index]
            X_test = X.iloc[y_test.index]

            pls = PLSRegression(n_components=30)
            pls.fit(X_train, y_train[tr])
            pred = pls.predict(X_test)

            a = np.array(pred.reshape(-1,).tolist())
            b = np.array(y_test[tr].tolist())

            R2 = rsquared(a,b)
            rmse = np.sqrt(mean_squared_error(a,b))
            nrmse = np.sqrt(mean_squared_error(a,b))/(b.max()-b.min())
            N_SE = nse(a, b)

            accu.append(R2)
            RMSE.append(rmse)
            NSE.append(N_SE)
            NRMSE.append(nrmse)
        
        accu = pd.DataFrame(accu,columns = [j])
        RMSE = pd.DataFrame(RMSE,columns = [j])
        NRMSE = pd.DataFrame(NRMSE,columns = [j])
        NSE = pd.DataFrame(NSE,columns = [j])
        
        R2_ex_frame = pd.concat([R2_ex_frame,accu],axis = 1)
        RMSE_ex_frame = pd.concat([RMSE_ex_frame,RMSE],axis = 1)
        nrmse_ex_frame = pd.concat([nrmse_ex_frame,NRMSE],axis = 1)
        nse_ex_frame = pd.concat([nse_ex_frame,NSE],axis = 1)
    
    R2_ex_frame.to_csv(f'../2_results/{tr}/{tr}_R2 site_extrapolation.csv',index = False)
    RMSE_ex_frame.to_csv(f'../2_results/{tr}/{tr}_RMSE site_extrapolation.csv',index = False)
    nrmse_ex_frame.to_csv(f'../2_results/{tr}/{tr}_nrmse site_extrapolation.csv',index = False)
    nse_ex_frame.to_csv(f'../2_results/{tr}/{tr}_nse site_extrapolation.csv',index = False)
    return[R2_ex_frame,RMSE_ex_frame,nrmse_ex_frame,nse_ex_frame]

def cross_PFT_CV(X,y,tr,n_splits,n_iterations):
    """cross PFTs validation of PLSR model for estimating leaf traits.
    Parameters:
    -----------
    X (numpy array): leaf spectra data used for PLSR modeling
    y (numpy array): leaf trait data used for PLSR modeling
    tr (str): trait name ("Chla+b", "Ccar", "EWT" or "LMA")
    n_splits (int): number of splits to divide the input data.
    n_iterations (int): How many iterations to train the PLSR model.

    Output files:
    -----------
    (1) Trained PLSR models in *.json format.
    (2) Leaf trait predictions in *.csv format.
    (3) PLSR VIP metric in *.csv format.
    (4) PLSR coefficients in *.csv format.
    (5) sample size files for each fold in *.csv format.
    """
    PRESS = []
    vip_score = pd.DataFrame(np.zeros(shape = (n_splits, X.shape[1])),columns = X.columns)
    plsr_coef = pd.DataFrame(np.zeros(shape = (n_splits, X.shape[1])),columns = X.columns)
    sample_size = pd.DataFrame(np.zeros(shape = (n_splits, 4)),columns = ['train_PFTs','test_PFTs','train_samples','test_samples'])
    
    k = 0
    saved_models = {"Trait_name": tr, "Units":trait_units(tr), "Type": "PLSR","Wavelength_units": "nanometers", "Wavelengths":np.arange(450,2401,10).tolist(), "Models": {},"Description":"coefficients, mean, std and itercept need to be converted to np.array() for model implementing, then employ the following equation for trait estimation: ((leaf spectra-Mean)/Std)@Coefficients + Intercept"}
    pfts = y['PFT'].unique()
    pfts = np.array([i for i in pfts if pd.isnull(i) == False and i != 'nan'])
    kf = KFold(n_splits=n_splits)
    
    for i in np.arange(10,X.shape[1]):
        press = []
        for train_index, test_index in kf.split(pfts):
            train_pft = pfts[train_index]
            test_pft = pfts[test_index]
            
            y_train = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
            y_test = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)

            for j in train_pft:
                temp = y[y['PFT'] == j]
                y_train = pd.concat([y_train,temp])
            for j in test_pft:
                temp = y[y['PFT'] == j]
                y_test = pd.concat([y_test,temp])
                
            X_train = X.iloc[y_train.index]
            X_test =  X.iloc[y_test.index]

            pls = PLSRegression(n_components=i)
            pls.fit(X_train, y_train[tr])
            pred = pls.predict(X_test)

            aa= np.array(pred.reshape(-1,).tolist())
            bb = np.array(y_test[tr].tolist())

            score = mean_squared_error(aa,bb)
            press.append(score)
        press_mean = np.mean(press)
        PRESS.append(press_mean)         
    n_components = PRESS.index(min(PRESS))+10
    print(tr, 'model: PFTs CV_n_components:',n_components)
    
    var = True
    for train_index, test_index in kf.split(pfts):
        train_pft = pfts[train_index]
        test_pft = pfts[test_index]
        
        y_train = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
        y_test = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
    
        for i in train_pft:
            temp = y[y['PFT'] == i]
            y_train = pd.concat([y_train,temp])
        for i in test_pft:
            temp = y[y['PFT'] == i]
            y_test = pd.concat([y_test,temp])
            
        X_train = X.iloc[y_train.index]
        X_test =  X.iloc[y_test.index]
        
        sample_size.iloc[k] = [train_pft,test_pft,len(X_train),len(X_test)]
        
        n_iterations = n_iterations
        var_start = True
        for iteration in range(n_iterations):
            XX_train, XX_test, yy_train, yy_test = train_test_split(X_train, y_train, test_size=0.3, random_state=iteration)
            pls = PLSRegression(n_components=n_components)
            pls.fit(XX_train, yy_train[tr])

            x_mean, x_std = pls._x_mean, pls._x_std
            coef, intercept = pls.coef_, pls._y_mean

            pred = pls.predict(X_test)
            pred = pd.DataFrame(pred,columns = [f'iteration_{iteration+1}'])
            vvv = vip(XX_train, yy_train[tr], pls)

            if var_start:
                iterative_pred = pred
                x_mean_, x_std_ = x_mean, x_std
                coefficients, intercept_, vip_ = coef, intercept, vvv
                var_start = False
            else:
                iterative_pred = pd.concat([iterative_pred,pred],axis = 1)
                x_mean_, x_std_, coefficients, intercept_, vip_ = x_mean_+x_mean, x_std_+x_std, coefficients+coef, intercept_+intercept, vip_+vvv

        final_mean, final_std = x_mean_/n_iterations, x_std_/n_iterations
        final_coef, final_intercept, final_vip = coefficients/n_iterations, intercept_/n_iterations, vip_/n_iterations
        final_model = {f"fold {k+1}({train_pft.tolist()} trained)": {"Coefficients": final_coef.tolist(), "Mean":final_mean.tolist(), "Std": final_std.tolist(), "Itercept": final_intercept.tolist()}}
        saved_models["Models"].update(final_model)

        y_test.reset_index(drop = True, inplace = True)
        final_pred = ((X_test-final_mean)/final_std)@final_coef + final_intercept

        final_pred.columns = ['final_model_result']
        final_pred.reset_index(drop = True, inplace = True)
        res = pd.concat([y_test,final_pred,iterative_pred],axis = 1)
        res['fold'] = f'fold{k+1}'

        if var:
            df = res
            var = False
        else:
            df = pd.concat([df,res],axis = 0)

        vip_score.iloc[k] = final_vip
        plsr_coef.iloc[k] = final_coef.reshape(-1,)
        k = k+1

    with open(f'../2_results/{tr}/0_saved_models/{tr}_{n_splits} fold PFTs CV_saved_models.json', 'w') as json_file:
        json.dump(saved_models, json_file)

    prefix = f"../2_results/{tr}/{tr}_{n_splits}fold PFTs CV_"
    df.to_csv(f"{prefix}df.csv", index=False)
    vip_score.to_csv(f"{prefix}VIP.csv", index=False)
    plsr_coef.to_csv(f"{prefix}coefficients.csv", index=False)
    sample_size.to_csv(f"{prefix}sample_size.csv", index=False)
    return

def leave_one_PFT_out(X,y,tr):
    """Leave one PFT out for PLSR model training for estimating leaf traits.
    Parameters:
    -----------
    X (numpy array): leaf spectra data used for PLSR modeling
    y (numpy array): leaf trait data used for PLSR modeling
    tr (str): trait name ("Chla+b", "Ccar", "EWT" or "LMA")

    Output files:
    -----------
    (1) Leaf trait predictions in *.csv format.
    (2) The accuracy file that training model from one PFT and applying to another PFT.
    (3) The overall accuracy that training model from one PFT and applying to all the other PFTs.
    (4) PLSR coefficients in *.csv format.
    (5) PLSR VIP metric in *.csv format.
    (6) Benchmark accuracy that train and test the model in the same site.
    """

    res = pd.DataFrame(np.zeros(shape = (0,8)),columns = ['training_PFT','testing_PFTs','train samples','test samples','R2','RMSE','NRMSE','NSE'])
    PFTs = y['PFT'].unique().tolist()
    PFTs = np.array([i for i in PFTs if pd.isnull(i) == False and i != 'nan'])
    
    PRESS = []
    accu = []
    RMSE = []
    NRMSE = []
    NSE = []
    
    vip_score = pd.DataFrame(np.zeros(shape = (len(PFTs),X.shape[1])),index = PFTs,columns = X.columns)
    plsr_coef = pd.DataFrame(np.zeros(shape = (len(PFTs),X.shape[1])),index = PFTs,columns = X.columns)
    k = 0
    
    loo = LeaveOneOut()
    for i in np.arange(10,X.shape[1]):
        press = []
        for test_index, train_index in loo.split(PFTs):
            y_test = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
            train_sets = PFTs[train_index]
            test_sets = PFTs[test_index]
            
            for j in test_sets:
                df_test = y[y['PFT']== j]
                y_test = pd.concat([y_test,df_test])
                
            y_train = y[y['PFT']== train_sets[0]]
            X_train = X.iloc[y_train.index]
            X_test = X.iloc[y_test.index]
    
            pls = PLSRegression(n_components=i)
            pls.fit(X_train, y_train[tr])
            pred = pls.predict(X_test)

            aa= np.array(pred.reshape(-1,).tolist())
            bb = np.array(y_test[tr].tolist())

            score = mean_squared_error(aa,bb)
            press.append(score)
        press_mean = np.mean(press)
        PRESS.append(press_mean)         
    n_components = PRESS.index(min(PRESS))+10
    print(tr, 'model: PFTs_LOO CV_n_components:',n_components)    
    
    var = True
    loo = LeaveOneOut()
    for test_index, train_index in loo.split(PFTs):
        y_test = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
        train_sets = PFTs[train_index]
        test_sets = PFTs[test_index]
        
        for j in test_sets:
            df_test = y[y['PFT']== j]
            y_test = pd.concat([y_test,df_test])
            
        y_train = y[y['PFT']== train_sets[0]]
        X_train = X.iloc[y_train.index]
        X_test = X.iloc[y_test.index]
        
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_train, y_train[tr])
        
        vvv = vip(X_train, y_train[tr], pls)
        coef = pls.coef_.reshape(-1,)
        vip_score.iloc[k] = vvv
        plsr_coef.iloc[k] = coef
        k = k+1
        
        pred = pls.predict(X_test)
        
        pred = pd.DataFrame(pred,columns = ['pred'])
        pred.reset_index(drop = True, inplace = True)
        y_test.reset_index(drop = True, inplace = True)
        new_df = pd.concat([pred,y_test],axis = 1)

        new_df['train_PFT'] = train_sets[0]
        new_df['test_PFTs'] = new_df['PFT']
    
        if var:
            df = new_df
            var = False
        else:
            df = pd.concat([df,new_df],axis = 0)
                
        a = new_df['pred']
        b = new_df[tr]
        
        R2 = rsquared(a,b)
        rmse = np.sqrt(mean_squared_error(a,b))
        nrmse = np.sqrt(mean_squared_error(a,b))/(b.max()-b.min())
        N_SE = nse(a, b)
        
        accu.append(R2)
        RMSE.append(rmse)
        NSE.append(N_SE)
        NRMSE.append(nrmse)
        
        for kk in new_df['PFT'].unique():
            data = new_df[new_df['PFT'] == kk]

            R_2 = rsquared(data['pred'],data[tr])
            r_mse = np.sqrt(mean_squared_error(data['pred'],data[tr]))
            n_rmse = np.sqrt(mean_squared_error(data['pred'],data[tr]))/(data[tr].max()-data[tr].min())
            N_S_E = nse(data['pred'],data[tr])
            
            temp = pd.DataFrame(np.array([y_train['PFT'].unique()[0],kk,len(y_train),len(data),R_2,r_mse,n_rmse,N_S_E]).reshape(1,8),columns = ['training_PFT','testing_PFTs','train samples','test samples','R2','RMSE','NRMSE','NSE'])
            res = pd.concat([res,temp])

    performance = pd.DataFrame([accu,RMSE, NRMSE, NSE]).T
    performance.columns = ['R2','RMSE','NRMSE','NSE']
    performance.index = PFTs
    
    prefix = f"../2_results/{tr}/{tr}_LOO_PFT_"
    df.to_csv(f'{prefix}df.csv',index = False)
    res.to_csv(f'{prefix}accuracy.csv',index = False)
    performance.to_csv(f'{prefix}overall_accuracy.csv')
    vip_score.to_csv(f'{prefix}vip_score.csv')
    plsr_coef.to_csv(f'{prefix}coefficients.csv')
    
    ## generate the benchmark accuracy (train and test at same sites)
    accu = []
    RMSE = []
    NRMSE = []
    NSE = []
    for i in PFTs:
        y1 = y[y['PFT']==i]
        X1 = X.loc[y1.index]
        
        press = []
        for i in np.arange(10,X1.shape[1]):
            pls = PLSRegression(n_components=i)
            pls.fit(X1, y1[tr])
            pred = pls.predict(X1)
            aa = np.array(pred.reshape(-1,).tolist())
            bb = np.array(y1[tr].tolist())
            score = np.sum((aa - bb) ** 2)
            press.append(score)
        n_components = press.index(min(press))+10

        pls = PLSRegression(n_components=n_components)
        pls.fit(X1, y1[tr])
        pred = pls.predict(X1)

        a = np.array(pred.reshape(-1,).tolist())
        b = np.array(y1[tr].tolist())

        R2 = rsquared(a,b)
        rmse = np.sqrt(mean_squared_error(a,b))
        nrmse = np.sqrt(mean_squared_error(a,b))/(b.max()-b.min())
        N_SE = nse(a, b)

        accu.append(R2)
        RMSE.append(rmse)
        NSE.append(N_SE)
        NRMSE.append(nrmse)
    res = pd.DataFrame(np.array([accu,RMSE,NRMSE,NSE])).T
    res.columns = ['R2','RMSE','NRMSE','NSE']
    res.index = PFTs
    res.to_csv(f'{prefix}benchmark_accuracy.csv')
    return

def cross_sites_PFTs(X,y,tr):
    """The transferability of PLSR model for the same PFT across different sites:
    Parameters:
    -----------
    X (numpy array): leaf spectra data used for PLSR modeling
    y (numpy array): leaf trait data used for PLSR modeling
    tr (str): trait name ("Chla+b", "Ccar", "EWT" or "LMA")

    Output files:
    -----------
    The accuracy file contained R2, RMSE, NEMSE and NSE metrics in *.csv file.
    """
    PFTs = y['PFT'].unique().tolist()
    PFTs = np.array([i for i in PFTs if pd.isnull(i) == False and i != 'nan'])
    col = ['PFTs','R2','RMSE','NSE','NRMSE']
    total_accuracy = pd.DataFrame(np.zeros(shape = (0,len(col))),columns = col)

    for pfts in PFTs:
        y_pft = y[y['PFT'] == pfts]
        X_pft = X.loc[y_pft.index]
        y_pft.reset_index(drop = True, inplace = True)
        X_pft.reset_index(drop = True, inplace = True)

        sites = []
        for i in y_pft['Site ID'].unique():
            df = y_pft[y_pft['Site ID'] == i]
            if len(df) > 20:
                sites.append(i)
        sites = np.array(sites)

        if len(sites)>1:
            accu = []
            RMSE = []
            NRMSE = []
            NSE = []

            loo = LeaveOneOut()
            for test_index, train_index in loo.split(sites):
                train_sites = sites[train_index]
                test_sites = sites[test_index]

                y_train = y_pft[y_pft['Site ID']== train_sites[0]]
                X_train = X_pft.iloc[y_train.index]

                pls = PLSRegression(n_components=20)
                pls.fit(X_train, y_train[tr])

                accu1 = []
                RMSE1 = []
                NRMSE1 = []
                NSE1 = []

                for i in test_sites:
                    y_test = y_pft[y_pft['Site ID']== i]
                    X_test = X_pft.iloc[y_test.index]

                    pred = pls.predict(X_test)
                    pred = pd.DataFrame(pred,columns = ['pred'])
                    pred.reset_index(drop = True, inplace = True)
                    y_test.reset_index(drop = True, inplace = True)
                    new_df = pd.concat([pred,y_test],axis = 1)

                    a = new_df['pred']
                    b = new_df[tr]

                    R2 = rsquared(a,b)
                    rmse = np.sqrt(mean_squared_error(a,b))
                    nrmse = np.sqrt(mean_squared_error(a,b))/(b.max()-b.min())
                    N_SE = nse(a, b)

                    accu1.append(R2)
                    RMSE1.append(rmse)
                    NSE1.append(N_SE)
                    NRMSE1.append(nrmse)

                accu.extend(accu1)
                RMSE.extend(RMSE1)
                NSE.extend(NSE1)
                NRMSE.extend(NRMSE1)

            a = pd.DataFrame(np.array(accu),columns = ['R2'])
            b = pd.DataFrame(np.array(RMSE),columns = ['RMSE'])
            c = pd.DataFrame(np.array(NSE),columns = ['NSE'])
            d = pd.DataFrame(np.array(NRMSE),columns = ['NRMSE'])
            temp = pd.concat([a,b,c,d],axis = 1)
            temp['PFTs'] = pfts
            total_accuracy = pd.concat([total_accuracy,temp])
    total_accuracy.reset_index(drop = True, inplace = True)
    total_accuracy.to_csv(f'../2_results/{tr}/{tr}_Cross sites for each PFT.csv',index = False)
    return

def PFT_extrapolation(X,y,tr):
    """Exploring PFT extrapolation ability of PLSR models. Select one PFT as the testing set and iteratively increase the number of PFTs used for training.
    Parameters:
    -----------
    X (numpy array): leaf spectra data used for PLSR modeling
    y (numpy array): leaf trait data used for PLSR modeling
    tr (str): trait name ("Chla+b", "Ccar", "EWT" or "LMA")

    Output files:
    -----------
    Standard summary statistics include R2, RMSE, NRMSE and NSE for PFT extrapolation.
    """
    s = pd.DataFrame(y['PFT'].value_counts())
    s = s[s['PFT']>2]
    s = s.index.tolist()
    s.reverse()
    s = np.array(s)
    
    R2_ex_frame = pd.DataFrame(np.zeros(shape = (len(s)-1,0)))
    RMSE_ex_frame = pd.DataFrame(np.zeros(shape = (len(s)-1,0)))
    nrmse_ex_frame = pd.DataFrame(np.zeros(shape = (len(s)-1,0)))
    nse_ex_frame = pd.DataFrame(np.zeros(shape = (len(s)-1,0)))
    
    for j in s:
        target_site = j
        source_site = s[s!=target_site]
    
        accu = []
        RMSE = []
        NRMSE = []
        NSE = []
    
        for k in range(1,len(source_site)+1):
            y_train = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
            for i in range(k):
                df_train = y[y['PFT']== source_site[i]]
                y_train = pd.concat([y_train,df_train])

            y_test = y[y['PFT']== target_site]

            X_train = X.iloc[y_train.index]
            X_test = X.iloc[y_test.index]

            pls = PLSRegression(n_components=20)
            pls.fit(X_train, y_train[tr])
            pred = pls.predict(X_test)

            a = np.array(pred.reshape(-1,).tolist())
            b = np.array(y_test[tr].tolist())

            R2 = rsquared(a,b)
            rmse = np.sqrt(mean_squared_error(a,b))
            nrmse = np.sqrt(mean_squared_error(a,b))/(b.max()-b.min())
            N_SE = nse(a, b)

            accu.append(R2)
            RMSE.append(rmse)
            NSE.append(N_SE)
            NRMSE.append(nrmse)
        
        accu = pd.DataFrame(accu,columns = [j])
        RMSE = pd.DataFrame(RMSE,columns = [j])
        NRMSE = pd.DataFrame(NRMSE,columns = [j])
        NSE = pd.DataFrame(NSE,columns = [j])
        
        R2_ex_frame = pd.concat([R2_ex_frame,accu],axis = 1)
        RMSE_ex_frame = pd.concat([RMSE_ex_frame,RMSE],axis = 1)
        nrmse_ex_frame = pd.concat([nrmse_ex_frame,NRMSE],axis = 1)
        nse_ex_frame = pd.concat([nse_ex_frame,NSE],axis = 1)

    R2_ex_frame.to_csv(f'../2_results/{tr}/{tr}_R2 PFT_extrapolation.csv',index = False)
    RMSE_ex_frame.to_csv(f'../2_results/{tr}/{tr}_RMSE PFT_extrapolation.csv',index = False)
    nrmse_ex_frame.to_csv(f'../2_results/{tr}/{tr}_nrmse PFT_extrapolation.csv',index = False)
    nse_ex_frame.to_csv(f'../2_results/{tr}/{tr}_nse PFT_extrapolation.csv',index = False)
    return[R2_ex_frame,RMSE_ex_frame,nrmse_ex_frame,nse_ex_frame]

def random_temporal_CV(X,y,tr,n_splits,dataset_num, n_iterations):
    """Random cross-validation of PLSR model for estimating leaf traits.
    Parameters:
    -----------
    X (numpy array): seasonal leaf spectra data used for PLSR modeling
    y (numpy array): seasonal leaf trait data used for PLSR modeling
    tr (str): trait name ("Chla+b", "Ccar", "EWT" or "LMA")
    n_splits (int): number of splits to divide the input data.
    dataset_num (str): dataset ID in the compiled dataset that contains the seasonal measurements ("Dataset#3", "Dataset#4", "Dataset#8")
    n_iterations (int): How many iterations to train the PLSR model.

    Output files:
    -----------
    (1) Trained PLSR models in *.json format.
    (2) Leaf trait predictions in *.csv format.
    (3) PLSR VIP metric in *.csv format.
    (4) PLSR coefficients in *.csv format.
    (5) sample size files for each fold in *.csv format.
    """
    PRESS = []
    accu = []
    RMSE = []
    NRMSE = []
    NSE = []
    pred_list = []
    test_list = []
    
    vip_score = pd.DataFrame(np.zeros(shape = (n_splits, X.shape[1])),columns = X.columns)
    plsr_coef = pd.DataFrame(np.zeros(shape = (n_splits, X.shape[1])),columns = X.columns)
    sample_size = pd.DataFrame(np.zeros(shape = (n_splits, 2)),columns = ['train_samples','test_samples'])
    
    k = 0
    saved_models = {"Trait_name": tr, "Units":trait_units(tr), "Type": "PLSR","Wavelength_units": "nanometers", "Wavelengths":np.arange(450,2401,10).tolist(), "Models": {},"Description":"coefficients, mean, std and itercept need to be converted to np.array() for model implementing, then employ the following equation for trait estimation: ((leaf spectra-Mean)/Std)@Coefficients + Intercept"}
    kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
    for i in np.arange(10,X.shape[1]):
        press = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            pls = PLSRegression(n_components=i)
            pls.fit(X_train, y_train[tr])

            pred = pls.predict(X_test)
            aa = np.array(pred.reshape(-1,).tolist())
            bb = np.array(y_test[tr].tolist())

            score = np.sum((aa - bb) ** 2)
            press.append(score)
        press_mean = np.mean(press)
        PRESS.append(press_mean)
    n_components = PRESS.index(min(PRESS))+10
    print(tr, 'model: random temporal CV_n_components:',n_components)
    
    var = True
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        sample_size.iloc[k] = [len(X_train),len(X_test)]
        
        n_iterations = n_iterations
        var_start = True
        for iteration in range(n_iterations):
            XX_train, XX_test, yy_train, yy_test = train_test_split(X_train, y_train, test_size=0.3, random_state=iteration)
            pls = PLSRegression(n_components=n_components)
            pls.fit(XX_train, yy_train[tr])

            x_mean, x_std = pls._x_mean, pls._x_std
            coef, intercept = pls.coef_, pls._y_mean

            pred = pls.predict(X_test)
            pred = pd.DataFrame(pred,columns = [f'iteration_{iteration+1}'])
            vvv = vip(XX_train, yy_train[tr], pls)

            if var_start:
                iterative_pred = pred
                x_mean_, x_std_ = x_mean, x_std
                coefficients, intercept_, vip_ = coef, intercept, vvv
                var_start = False
            else:
                iterative_pred = pd.concat([iterative_pred,pred],axis = 1)
                x_mean_, x_std_, coefficients, intercept_, vip_ = x_mean_+x_mean, x_std_+x_std, coefficients+coef, intercept_+intercept, vip_+vvv

        final_mean, final_std = x_mean_/n_iterations, x_std_/n_iterations
        final_coef, final_intercept, final_vip = coefficients/n_iterations, intercept_/n_iterations, vip_/n_iterations
        final_model = {f"fold {k+1}": {"Coefficients": final_coef.tolist(), "Mean":final_mean.tolist(), "Std": final_std.tolist(), "Itercept": final_intercept.tolist()}}
        saved_models["Models"].update(final_model)

        y_test.reset_index(drop = True, inplace = True)
        final_pred = ((X_test-final_mean)/final_std)@final_coef + final_intercept

        final_pred.columns = ['final_model_result']
        final_pred.reset_index(drop = True, inplace = True)
        res = pd.concat([y_test,final_pred,iterative_pred],axis = 1)
        res['fold'] = f'fold{k+1}'

        if var:
            df = res
            var = False
        else:
            df = pd.concat([df,res],axis = 0)

        vip_score.iloc[k] = final_vip
        plsr_coef.iloc[k] = final_coef.reshape(-1,)
        k = k+1

    with open(f'../2_results/{tr}/0_saved_models/{dataset_num}_{tr}_{n_splits} fold temporal_random CV_saved_models.json', 'w') as json_file:
        json.dump(saved_models, json_file)

    prefix = f"../2_results/{tr}/{dataset_num}_{tr}_{n_splits}fold temporal_random CV_"
    df.to_csv(f"{prefix}df.csv", index=False)
    vip_score.to_csv(f"{prefix}VIP.csv", index=False)
    plsr_coef.to_csv(f"{prefix}coefficients.csv", index=False)
    sample_size.to_csv(f"{prefix}sample_size.csv", index=False)
    return

def temporal_CV(X,y,tr,n_splits,dataset_num,n_iterations):
    """temporal cross-validation of PLSR model for estimating leaf traits.
    Parameters:
    -----------
    X (numpy array): seasonal leaf spectra data used for PLSR modeling
    y (numpy array): seasonal leaf trait data used for PLSR modeling
    tr (str): trait name ("Chla+b", "Ccar", "EWT" or "LMA")
    n_splits (int): number of splits to divide the input data.
    dataset_num (str): dataset ID in the compiled dataset that contains the seasonal measurements ("Dataset#3", "Dataset#4", "Dataset#8")
    n_iterations (int): How many iterations to train the PLSR model.

    Output files:
    -----------
    (1) Trained PLSR models in *.json format.
    (2) Leaf trait predictions in *.csv format.
    (3) PLSR VIP metric in *.csv format.
    (4) PLSR coefficients in *.csv format.
    (5) sample size files for each fold in *.csv format.
    """
    PRESS = []
    vip_score = pd.DataFrame(np.zeros(shape = (n_splits, X.shape[1])),columns = X.columns)
    plsr_coef = pd.DataFrame(np.zeros(shape = (n_splits, X.shape[1])),columns = X.columns)
    sample_size = pd.DataFrame(np.zeros(shape = (n_splits, 4)),columns = ['train_time','test_time','train_samples','test_samples'])
    
    k = 0
    saved_models = {"Trait_name": tr, "Units":trait_units(tr), "Type": "PLSR","Wavelength_units": "nanometers", "Wavelengths":np.arange(450,2401,10).tolist(), "Models": {},"Description":"coefficients, mean, std and itercept need to be converted to np.array() for model implementing, then employ the following equation for trait estimation: ((leaf spectra-Mean)/Std)@Coefficients + Intercept"}
    time = y['Sample date'].unique()
    kf = KFold(n_splits=n_splits)
    
    for i in np.arange(10,X.shape[1]):
        press = []
        for train_index, test_index in kf.split(time):
            train_time = time[train_index]
            test_time = time[test_index]
            
            y_train = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
            y_test = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)

            for j in train_time:
                temp = y[y['Sample date'] == j]
                y_train = pd.concat([y_train,temp])
            for j in test_time:
                temp = y[y['Sample date'] == j]
                y_test = pd.concat([y_test,temp])
                
            X_train = X.iloc[y_train.index]
            X_test =  X.iloc[y_test.index]

            pls = PLSRegression(n_components=i)
            pls.fit(X_train, y_train[tr])
            pred = pls.predict(X_test)

            aa= np.array(pred.reshape(-1,).tolist())
            bb = np.array(y_test[tr].tolist())

            score = mean_squared_error(aa,bb)
            press.append(score)
        press_mean = np.mean(press)
        PRESS.append(press_mean)         
    n_components = PRESS.index(min(PRESS))+10
    print(tr, 'model: temporal CV_n_components:',n_components)
    
    var = True
    for train_index, test_index in kf.split(time):
        train_time = time[train_index]
        test_time = time[test_index]
        
        y_train = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
        y_test = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
    
        for i in train_time:
            temp = y[y['Sample date'] == i]
            y_train = pd.concat([y_train,temp])
        for i in test_time:
            temp = y[y['Sample date'] == i]
            y_test = pd.concat([y_test,temp])
            
        X_train = X.iloc[y_train.index]
        X_test =  X.iloc[y_test.index]
        
        sample_size.iloc[k] = [train_time,test_time,len(X_train),len(X_test)]
        
        n_iterations = n_iterations    ##########
        var_start = True
        for iteration in range(n_iterations):
            XX_train, XX_test, yy_train, yy_test = train_test_split(X_train, y_train, test_size=0.3, random_state=iteration)
            pls = PLSRegression(n_components=n_components)
            pls.fit(XX_train, yy_train[tr])

            x_mean, x_std = pls._x_mean, pls._x_std
            coef, intercept = pls.coef_, pls._y_mean

            pred = pls.predict(X_test)
            pred = pd.DataFrame(pred,columns = [f'iteration_{iteration+1}'])
            vvv = vip(XX_train, yy_train[tr], pls)

            if var_start:
                iterative_pred = pred
                x_mean_, x_std_ = x_mean, x_std
                coefficients, intercept_, vip_ = coef, intercept, vvv
                var_start = False
            else:
                iterative_pred = pd.concat([iterative_pred,pred],axis = 1)
                x_mean_, x_std_, coefficients, intercept_, vip_ = x_mean_+x_mean, x_std_+x_std, coefficients+coef, intercept_+intercept, vip_+vvv

        final_mean, final_std = x_mean_/n_iterations, x_std_/n_iterations
        final_coef, final_intercept, final_vip = coefficients/n_iterations, intercept_/n_iterations, vip_/n_iterations
        final_model = {f"fold {k+1}({train_time.tolist()} trained)": {"Coefficients": final_coef.tolist(), "Mean":final_mean.tolist(), "Std": final_std.tolist(), "Itercept": final_intercept.tolist()}}
        saved_models["Models"].update(final_model)

        y_test.reset_index(drop = True, inplace = True)
        final_pred = ((X_test-final_mean)/final_std)@final_coef + final_intercept

        final_pred.columns = ['final_model_result']
        final_pred.reset_index(drop = True, inplace = True)
        res = pd.concat([y_test,final_pred,iterative_pred],axis = 1)
        res['fold'] = f'fold{k+1}'

        if var:
            df = res
            var = False
        else:
            df = pd.concat([df,res],axis = 0)

        vip_score.iloc[k] = final_vip
        plsr_coef.iloc[k] = final_coef.reshape(-1,)
        k = k+1

    with open(f'../2_results/{tr}/0_saved_models/{dataset_num}_{tr}_{n_splits} fold temporal CV_saved_models.json', 'w') as json_file:
        json.dump(saved_models, json_file)

    prefix = f"../2_results/{tr}/{dataset_num}_{tr}_{n_splits}fold temporal CV_"
    df.to_csv(f"{prefix}df.csv", index=False)
    vip_score.to_csv(f"{prefix}VIP.csv", index=False)
    plsr_coef.to_csv(f"{prefix}coefficients.csv", index=False)
    sample_size.to_csv(f"{prefix}sample_size.csv", index=False)
    return

def leave_one_season_out_CV(X,y,tr,dataset_num):
    """Leave one season out for PLSR model training for estimating leaf traits.
    Parameters:
    -----------
    X (numpy array): leaf spectra data used for PLSR modeling
    y (numpy array): leaf trait data used for PLSR modeling
    tr (str): trait name ("Chla+b", "Ccar", "EWT" or "LMA")
    dataset_num (str): dataset ID in the compiled dataset that contains the seasonal measurements ("Dataset#3", "Dataset#4", "Dataset#8")

    Output files:
    -----------
    (1) Leaf trait predictions in *.csv format.
    (2) The accuracy file that training model from one season and applying to another season.
    (3) The overall accuracy that training model from one season and applying to all the other seasons.
    (4) PLSR coefficients in *.csv format.
    (5) PLSR VIP metric in *.csv format.
    """
    res = pd.DataFrame(np.zeros(shape = (0,8)),columns = ['training_season','testing_season','train samples','test samples','R2','RMSE','NRMSE','NSE'])
    time = y['season'].unique()
    
    PRESS = []
    accu = []
    RMSE = []
    NRMSE = []
    NSE = []
    
    vip_score = pd.DataFrame(np.zeros(shape = (len(time),X.shape[1])),index = time,columns = X.columns)
    plsr_coef = pd.DataFrame(np.zeros(shape = (len(time),X.shape[1])),index = time,columns = X.columns)
    k = 0
    
    loo = LeaveOneOut()
    for i in np.arange(10,X.shape[1]):
        press = []
        for test, train in loo.split(time):
            y_test = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
            train_time = time[train]
            test_time = time[test]
            
            for j in test_time:
                df_test = y[y['season'] == j]
                y_test = pd.concat([y_test,df_test])
                
            y_train = y[y['season']== train_time[0]]
            X_train = X.iloc[y_train.index]
            X_test =  X.iloc[y_test.index]
    
            pls = PLSRegression(n_components=i)
            pls.fit(X_train, y_train[tr])
            pred = pls.predict(X_test)

            aa= np.array(pred.reshape(-1,).tolist())
            bb = np.array(y_test[tr].tolist())

            score = mean_squared_error(aa,bb)
            press.append(score)
        press_mean = np.mean(press)
        PRESS.append(press_mean)         
    n_components = PRESS.index(min(PRESS))+10
    print(tr, 'model: time_LOO CV_n_components:',n_components)    
    
    var = True
    loo = LeaveOneOut()
    for test, train in loo.split(time):
        y_test = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
        train_time = time[train]
        test_time = time[test]

        for j in test_time:
            df_test = y[y['season'] == j]
            y_test = pd.concat([y_test,df_test])
        
        y_train = y[y['season']== train_time[0]]
        X_train = X.iloc[y_train.index]
        X_test =  X.iloc[y_test.index]
        
        pls = PLSRegression(n_components= n_components)
        pls.fit(X_train, y_train[tr])
        
        vvv = vip(X_train, y_train[tr], pls)
        coef = pls.coef_.reshape(-1,)
        vip_score.iloc[k] = vvv
        plsr_coef.iloc[k] = coef
        k = k+1
        
        pred = pls.predict(X_test)
        
        pred = pd.DataFrame(pred,columns = ['pred'])
        pred.reset_index(drop = True, inplace = True)
        y_test.reset_index(drop = True, inplace = True)
        new_df = pd.concat([pred,y_test],axis = 1)
        
        new_df['train_season'] = train_time[0]
        new_df['test_seasons'] = new_df['season']
        
        if var:
            df = new_df
            var = False
        else:
            df = pd.concat([df,new_df],axis = 0)
    
        a = new_df['pred']
        b = new_df[tr]
        
        R2 = rsquared(a,b)
        rmse = np.sqrt(mean_squared_error(a,b))
        nrmse = np.sqrt(mean_squared_error(a,b))/(b.max()-b.min())
        N_SE = nse(a, b)
        
        accu.append(R2)
        RMSE.append(rmse)
        NSE.append(N_SE)
        NRMSE.append(nrmse)
        
        for kk in new_df['season'].unique():
            data = new_df[new_df['season'] == kk]

            R_2 = rsquared(data['pred'],data[tr])
            r_mse = np.sqrt(mean_squared_error(data['pred'],data[tr]))
            n_rmse = np.sqrt(mean_squared_error(data['pred'],data[tr]))/(data[tr].max()-data[tr].min())
            N_S_E = nse(data['pred'],data[tr])
            
            temp = pd.DataFrame(np.array([y_train['season'].unique()[0],kk,len(y_train),len(data), R_2,r_mse,n_rmse,N_S_E]).reshape(1,8),columns = ['training_season','testing_season','train samples','test samples','R2','RMSE','NRMSE','NSE'])
            res = pd.concat([res,temp])

    performance = pd.DataFrame([accu,RMSE, NRMSE, NSE]).T
    performance.columns = ['R2','RMSE','NRMSE','NSE']
    performance.index = time

    df.to_csv('../2_results/'+tr+'/'+tr+'_'+dataset_num+'_LOO_season_df.csv',index = False)
    res.to_csv('../2_results/'+tr+'/'+tr+'_'+dataset_num+'_LOO_season_accuracy.csv',index = False)
    performance.to_csv('../2_results/'+tr+'/'+tr+'_'+dataset_num+'_LOO_season_overall_accuracy.csv')
    vip_score.to_csv('../2_results/'+tr+'/'+tr+'_'+dataset_num+'_LOO_season_vip_score.csv')
    plsr_coef.to_csv('../2_results/'+tr+'/'+tr+'_'+dataset_num+'_LOO_season_coefficients.csv')
    return

def across_sensors(X,y,tr,PFT):
    """Test the sensors effects on model transferability (ensure the same PFT, sites, and location. Only sensors different.).
    Parameters:
    -----------
    X (numpy array): leaf spectra data used for PLSR modeling
    y (numpy array): leaf trait data used for PLSR modeling
    tr (str): trait name ("Chla+b", "Ccar", "EWT" or "LMA")
    PFT (str): The PFT of the selected data (Grasslands, Croplands, Shrublands, Deciduous broadleaf forests, Deciduous needleleaf forests, Evergreen broadleaf forests or Evergreen needleleaf forests)

    Output files:
    -----------
    (1) Leaf trait predictions in *.csv format.
    (2) The accuracy file that remaining one sensor as validation set, the other sensors as calibration set
    """
    sensor = y['Instruments'].unique()
    PRESS = []
    accu = []
    RMSE = []
    NRMSE = []
    NSE = []

    k = 0
    loo = LeaveOneOut()
    for i in np.arange(10,X.shape[1]):
        press = []
        for train, test in loo.split(sensor):
            y_train = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
            train_sensor = sensor[train]
            test_sensor = sensor[test]

            for j in train_sensor:
                df_train = y[y['Instruments'] == j]
                y_train = pd.concat([y_train,df_train])

            y_test = y[y['Instruments']== test_sensor[0]]
            X_train = X.iloc[y_train.index]
            X_test =  X.iloc[y_test.index]

            pls = PLSRegression(n_components=i)
            pls.fit(X_train, y_train[tr])
            pred = pls.predict(X_test)

            aa= np.array(pred.reshape(-1,).tolist())
            bb = np.array(y_test[tr].tolist())

            score = mean_squared_error(aa,bb)
            press.append(score)
        press_mean = np.mean(press)
        PRESS.append(press_mean)         
    n_components = PRESS.index(min(PRESS))+10
    print(tr, 'model: sensor_LOO CV_n_components:',n_components)
    
    var = True
    loo = LeaveOneOut()
    for train, test in loo.split(sensor):
        y_train = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
        train_sensor = sensor[train]
        test_sensor = sensor[test]

        for j in train_sensor:
            df_train = y[y['Instruments'] == j]
            y_train = pd.concat([y_train,df_train])

        y_test = y[y['Instruments']== test_sensor[0]]
        X_train = X.iloc[y_train.index]
        X_test =  X.iloc[y_test.index]

        print("train_sensors:",train_sensor, "length:",len(X_train), "PFT:",PFT)
        print("test_sensors:",test_sensor, "length:",len(X_test),"PFT:",PFT)

        pls = PLSRegression(n_components= n_components)
        pls.fit(X_train, y_train[tr])
        k = k+1

        pred = pls.predict(X_test)

        pred = pd.DataFrame(pred,columns = ['pred'])
        pred.reset_index(drop = True, inplace = True)
        y_test.reset_index(drop = True, inplace = True)
        new_df = pd.concat([pred,y_test],axis = 1)

        new_df['train_sensors'] = str(train_sensor.tolist())
        new_df['test_sensors'] = new_df['Instruments']

        if var:
            df = new_df
            var = False
        else:
            df = pd.concat([df,new_df],axis = 0)


        a = new_df['pred']
        b = new_df[tr]

        R2 = rsquared(a,b)
        rmse = np.sqrt(mean_squared_error(a,b))
        nrmse = np.sqrt(mean_squared_error(a,b))/(b.max()-b.min())
        N_SE = nse(a, b)

        accu.append(R2)
        RMSE.append(rmse)
        NSE.append(N_SE)
        NRMSE.append(nrmse)

    performance = pd.DataFrame([accu,RMSE, NRMSE, NSE]).T
    performance.columns = ['R2','RMSE','NRMSE','NSE']
    performance.index = sensor

    df.to_csv('../2_results/'+tr+'/'+tr+'_'+PFT+'_LOO_sensor_df.csv',index = False)
    performance.to_csv('../2_results/'+tr+'/'+tr+'_'+PFT+'_LOO_sensor_overall_accuracy.csv')
    return
    
