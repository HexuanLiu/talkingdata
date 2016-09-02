base_models={}

def fit_predict(base_models, stacker, X, y, T):
    #X = np.array(X)
    #y = np.array(y)
    #T = np.array(T)

    folds = list(StratifiedKFold(y, n_folds=5, shuffle=True, random_state=2016))

    S_train = np.zeros((X.shape[0], len(base_models)*12))
    S_test = np.zeros((T.shape[0], len(base_models)*12))
    
    # for each base model perform 5_fold #
    for i, clf in enumerate(base_models):
        print(i)
        S_test_i = np.zeros((T.shape[0], len(folds)))

        # for each fold train and test #
        for j, (train_idx, test_idx) in enumerate(folds):
            print(j)
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_holdout = X[test_idx]
                
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_holdout)[:]
            S_train[test_idx, i*12:i*12+12] = y_pred
            S_test_i_j = clf.predict_proba(T)[:]
        
        # take the average of 5 #
        S_test[:, i*12:i*12+12] = np.mean(S_test_i_j, axis=0)

    # fit the stacker model #
    stacker.fit(S_train, y)
    y_pred = stacker.predict_proba(S_test)[:]
