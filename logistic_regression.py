Xtrain = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label, Xtr_time), format='csr')
Xtest =  hstack((Xte_brand, Xte_model, Xte_app, Xte_label, Xte_time), format='csr')

targetencoder = LabelEncoder().fit(gatrain.group)
y = targetencoder.transform(gatrain.group)
nclasses = len(targetencoder.classes_)

def fit_predict(clf, X, y, T):
    folds = list(StratifiedKFold(y, n_folds=5, shuffle=True, random_state=2016))

    S_train = np.zeros((X.shape[0], 12))
    S_test = np.zeros((T.shape[0], 12))
    Stest={}

    # for each fold train and test #
    for j, (train_idx, test_idx) in enumerate(folds):
        print(j)
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_holdout = X[test_idx]
                
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_holdout)[:]
        S_train[test_idx, 0:12] = y_pred
        Stest[j] = clf.predict_proba(T)[:]
        
    # take the average of 5 #
    S_test = np.mean([Stest[0], Stest[1],Stest[2], Stest[3], Stest[4]], axis=0)

    pd.DataFrame(S_train).to_csv('strain.csv')
    pd.DataFrame(S_test).to_csv('stest.csv')

clf1=LogisticRegression(C=0.02, multi_class='multinomial',solver='newton-cg')
fit_predict(clf1, Xtrain, y, Xtest)
