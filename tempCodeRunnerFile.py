print("Nghi thuc kiem tra Hold_out\n")
max = 0;
max_index = 0;
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3.0, random_state = 10)
for i in range(1, 11):
	Tree = DecisionTreeClassifier(criterion="entropy", random_state=10, max_depth=i+5, min_samples_leaf=i+1)
	Tree.fit(X_train, y_train)
	y_pred = Tree.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	if (max <= acc):
		max = acc
		max_index = i
	print ("Lan lap ", i, " Do chinh xac =", round(acc*100, 2))