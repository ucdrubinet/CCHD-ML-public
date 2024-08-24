def ml_cchd_detection(data_path, printFull = True, col=0,predictProb = 0.5,ADA_est = 300, RF_seed = 7, isStd = True, is48orLess = True, modelName = 'RF', train_data_path = '/home/sample_features.csv'):
    ##load old and new data
    data = pd.read_csv(train_data_path)
    
    eval_data = pd.read_csv(data_path)
    
    data.drop(columns='Unnamed: 0',inplace=True)
    eval_data.drop(columns='Unnamed: 0',inplace=True)
    
    ##set columns to use, we remove columns like filename, Coarc, stuff that our model won't use
    min_features_to_select = 1
    columns_to_use_eval = list(set(eval_data.columns).difference(['Case', 'ID','take']))
    columns_to_use = list(set(data.columns).difference(['Case', 'ID','take','diagnosis']))
    
    ##chose scaler
    if isStd:
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()
        
    
    ##drop Non-Critical and Unlabeled Values
    data.drop(data.loc[data['diagnosis']=='Non-Critical'].index,inplace=True)
    data.drop(data.loc[data['diagnosis']=='Unlabeled'].index,inplace=True)

    data.replace({'diagnosis': {'Healthy': 0, 'CCHD':1, 'coarc':1}},inplace=True)
    
    y_train = data['diagnosis'].reset_index(drop=True)
    
    X_train = data[columns_to_use]
    
    X_test = eval_data[columns_to_use_eval]
    
    ## saves dataframe for column inspection
    Xpd = X_train
        
    ## scale it, label
    X_train = scaler.fit_transform(X_train)
    
    X_test = scaler.fit_transform(X_test)
    
    
    label_encoder = LabelEncoder()
    y_train = y_train.to_numpy()
    y_train = label_encoder.fit_transform(y_train)
    
    ##choose model
    if modelName == 'RF':
        print("Initializing CCHD Detection Random Forest")
        model = RandomForestClassifier(random_state=RF_seed)
    elif modelName == 'LR':
        print("Initializing CCHD Detection Logistic Regression")
        model = LogisticRegression(random_state=0)
    elif modelName == 'GB':
        print("Initializing CCHD Detection Gradient Boosting")
        model = GradientBoostingClassifier(random_state=0)
    elif modelName == 'DT':
        print("Initializing CCHD Detection Decision Tree")
        model = tree.DecisionTreeClassifier()
    elif modelName == 'AB':
        print("Initializing CCHD Detection ADA Boost")
        model = AdaBoostClassifier(n_estimators=ADA_est)
    else:
        print("Picked an invalid model name. Available models are:",'\n','RF, LR, GB, DT, AB')
        print("Initializing default CCHD Detection Random Forest")
        model = RandomForestClassifier(random_state=RF_seed)
    
    ## perform Recursive Feature Elimination
    columns_to_bool = None
    if isinstance(col, int):
        print("Performing CCHD Detection Recursive Feature Elimination")
        rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state = 0),
                  scoring='recall',
                  min_features_to_select=4)
        rfecv.fit(X_train, y_train)
        columns_to_bool = rfecv.support_
        print("CCHD Detection RFE done")
    else:
        print("Importing previous CCHD Detection  RFE Results")
        columns_to_bool = col
    ## applies columns
    X_train = X_train[:,columns_to_bool]
    X_test = X_test[:,columns_to_bool]
    
    ## print the columns that were used
    if printFull:
        print("Features Used by CCHD Detection  model: ")
        for j in range(len(columns_to_bool)):
            if columns_to_bool[j]:
                print(Xpd.columns[j])
                    
    ##do model fitting
    model.fit(X_train, y_train)
    
    ##lines for AUC, here there is no change of prediction probability
    predictions = model.predict_proba(X_test)

    ##lines for Metrics, here there is a change in prediction
    y_pred = (model.predict_proba(X_test)[:,1] >= predictProb).astype(bool)
    
    cchd_gt = []
    for prediction in y_pred:
        if prediction:
            cchd_gt.append("CCHD")
        else:
            cchd_gt.append("Healthy")
            
    eval_data['cchd_gt'] = cchd_gt
    
    ##finalize function, return values of spec, sens and aucs for user to tune model accordingly
    return eval_data

def read_diagnosis(evaled_data):
    for i in range(len(evaled_data)):
        case_name = evaled_data.ID[i]
        diagnosis = evaled_data.cchd_gt[i]
        print("Case ", case_name, "is diagnosed as ", diagnosis)