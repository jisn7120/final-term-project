# final_term_project
y_pred = np.zeros_like(y_test) # y_test 기반으로 y_pred를 만듬
perc = sklearn.linear_model.Perceptron(penalty=None, alpha=0.00001, fit_intercept=True, max_iter=100, tol=0.1, shuffle=True, verbose=0, eta0=1.0, n_jobs=None, random_state=0, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False)  # 각각의 변수를 다음과 같이 설정하여 train을 시켜봄

X_train2 = pow(X_train,1.09) 
X_train3 = pow(X_train,0.76) #train2 trian3 정의
X_train_new = []
y_train_new = []
for i in range(len(X_train)):
    X_train_new.append(X_train[i])
    X_train_new.append(X_train2[i])
    X_train_new.append(X_train3[i])
    y_train_new.append(y_train[i])
    y_train_new.append(y_train[i])
    y_train_new.append(y_train[i])
X_train_new = np.array(X_train_new)
y_train_new = np.array(y_train_new)

y_pred = np.zeros_like(y_test)
perc = sklearn.linear_model.Perceptron(penalty=None, alpha=0.00001, fit_intercept=True, max_iter=100, tol=0.1, shuffle=True, verbose=0, eta0=1.0, n_jobs=None, random_state=0, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False)
#y_test를 이용하여 y_pred를 다음과 같은 변수로 설정 (prec 이용)

perc.fit(X_train_new, y_train_new)


y_pred = perc.predict(X_test)
