params = {}
params['booster'] = 'gblinear'
params['objective'] = "multi:softprob"
params['eval_metric'] = 'mlogloss'
params['eta'] = 0.005
params['num_class'] = 12
params['lambda'] = 3
params['alpha'] = 2

folds = list(StratifiedKFold(y, n_folds=5, shuffle=True, random_state=2016))

S_train = np.zeros((Xtrain.shape[0], 12))
S_test = np.zeros((Xtest.shape[0], 12))
Stest={}

# for each fold train and test #
for j, (train_idx, test_idx) in enumerate(folds):
    print(j)
    X_train = Xtrain[train_idx]
    y_train = y[train_idx]
    X_holdout = Xtrain[test_idx]
    y_holdout=y[test_idx]
                
    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_holdout, label=y_holdout)

    watchlist = [(d_train, 'train'), (d_valid, 'eval')]

    clf = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=25)

    y_pred = clf.predict(xgb.DMatrix(X_holdout))

    S_train[test_idx, 0:12] = y_pred
    Stest[j] = clf.predict(xgb.DMatrix(Xtest))
        
# take the average of 5 #
S_test = np.mean([Stest[0], Stest[1], Stest[2], Stest[3], Stest[4]], axis=0)

pd.DataFrame(S_train).to_csv('strain1.csv')
pd.DataFrame(S_test).to_csv('stest1.csv')
