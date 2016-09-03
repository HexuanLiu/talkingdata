def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(150, input_dim=Xtrain.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(50, input_dim=Xtrain.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(12, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
    return model

def baseline_model2():
    # create model
    model = Sequential()
    model.add(Dense(50, input_dim=Xtrain.shape[1], init='normal', activation='tanh'))
    model.add(Dropout(0.6))
    model.add(Dense(12, init='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
    return model
    
folds = list(StratifiedKFold(y, n_folds=5, shuffle=True, random_state=2016))

S_train = np.zeros((Xtrain.shape[0], 12))
S_test = np.zeros((Xtest.shape[0], 12))
Stest={}

# for each fold train and test #
for j, (train_idx, test_idx) in enumerate(folds):
    print(j)
    X_train = Xtrain[train_idx]
    y_train = dummy_y[train_idx]
    X_holdout = Xtrain[test_idx]
    y_holdout=dummy_y[test_idx]
    model=None
    model=baseline_model()
    fit= model.fit_generator(generator=batch_generator(X_train, y_train, 400, True),
                         nb_epoch=15,
                         samples_per_epoch=69984,
                         validation_data=(X_holdout.todense(), y_holdout), verbose=2
                         )
    scores_val = model.predict_generator(generator=batch_generatorp(X_holdout, 400, False), val_samples=X_holdout.shape[0])
    scores = model.predict_generator(generator=batch_generatorp(Xtest, 800, False), val_samples=Xtest.shape[0])

    S_train[test_idx, 0:12] = scores_val
    Stest[j] = scores
        
# take the average of 5 #
S_test = np.mean([Stest[0], Stest[1], Stest[2], Stest[3], Stest[4]], axis=0)

pd.DataFrame(S_train).to_csv('strain2.csv')
pd.DataFrame(S_test).to_csv('stest2.csv')
