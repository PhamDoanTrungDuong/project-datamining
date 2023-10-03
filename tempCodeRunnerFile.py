# Đưa dữ liệu về đúng kiểu của nó
# def to_typedata(dataset_name, features, typedata):
#     dataset_name[features] = dataset_name[features].astype(typedata)

# # Đổi dữ liệu cột Object về dữ liệu kiểu số
# def transfer_to_numberic_data(dataset_name, features):
#     for col in features:
#         i = 0
#         unique_values = np.unique(dataset_name[col])
#         for value in unique_values:
#             dataset_name[col] = np.where(dataset_name[col] == value, i, dataset_name[col])
#             i+=1
#     return dataset_name

# # Đổi dữ liệu cột String về dữ liệu kiểu số (float)
# def transfer_string_to_float_data(dataset_name, features):
#     for col in features:
#         col_values = dataset_name[col]
#         for value in col_values:
#             if isinstance(value, str):
#                 dataset_name[col] = float(value.replace(',', '.'))
#             else:
#                 dataset_name[col] = float(value)
#     return dataset_name

# #------Đổi dữ liệu String sang dữ liệu kiểu số---------
# columns = ['Class']
# data = transfer_to_numberic_data(data, columns)

# junk_column = ['Compactness', 'ShapeFactor3']
# data = transfer_string_to_float_data(data, junk_column)

# scaler = StandardScaler();

# X = data.iloc[:,0:17]
# # Chuẩn hoá dữ liệu
# # X = scaler.fit_transform(X)

# # Chuyển đổi dữ liệu kiểu số để thư viện sklearn nhận diện
# y = data.Class
# y = y.astype('int')
# print(data.dtypes)

# # Kiểm tra dữ liệu isnull?
# print("\n")
# print("Kiem tra xem du lieu co bi thieu (NULL) khong?")
# print(data.isnull().sum())

# # Chuyển đổi kiểu đối tượng
# features = ['Class']
# to_typedata(data, features, 'int64')

# # features = ['Compactness', 'ShapeFactor3', 'ConvexArea', 'Area']
# # for i in features:
# #     to_typedata(data, i, 'float64')
    
# # ----------------------------Nghi Thức HOLD_OUT----------------------------
# print("Nghi thuc kiem tra Hold_out\n")
# max = 0;
# max_index = 0;
# X_train, X_test, y_train, y_test = train_test_split(scaler.fit_transform(X), y, test_size=1/3.0, random_state = 10)
# for i in range(1, 11):
# 	Tree = DecisionTreeClassifier(criterion="entropy", random_state=10, max_depth=i+5, min_samples_leaf=i+1)
# 	Tree.fit(X_train, y_train)
# 	y_pred = Tree.predict(X_test)
# 	acc = accuracy_score(y_test, y_pred)
# 	if (max <= acc):
# 		max = acc
# 		max_index = i
# 	print ("Lan lap ", i, " Do chinh xac =", round(acc*100, 2))

# # # ----------------------------Nghi Thức K_FOLD----------------------------
# kf = KFold(n_splits=10, shuffle = True)

# print("\nNghi thuc kiem tra K-fold\n")

# KNN = KNeighborsClassifier(n_neighbors = 10)

# Bayes = GaussianNB()

# Tree = DecisionTreeClassifier(criterion="entropy", random_state=10, max_depth=max_index, min_samples_leaf=max_index)

# total_acc_tree = 0
# total_acc_knn = 0
# total_acc_bayes = 0
# arrTree = []
# arrKNN = []
# arrBayes = []

# i=1

# for train_index, test_index in kf.split(X):
# 	#-------------Split Data--------------
# 	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
# 	y_train, y_test = y.iloc[train_index], y.iloc[test_index]
 
# 	print("=============================")
# 	print("\nLan lap thu", i, "")
# 	i = i + 1

# 	#--------DecisionTree----------
# 	Tree.fit(X_train, y_train)
# 	y_pred = Tree.predict(X_test)
# 	acc_tree = accuracy_score(y_test, y_pred) * 100
# 	total_acc_tree += acc_tree
# 	matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
# 	print("\nConfusion Matrix cua Decision Tree")
# 	print(matrix)

# 	#-------------KNN--------------
# 	KNN.fit(X_train, y_train)
# 	y_pred = KNN.predict(X_test)
# 	acc_knn = accuracy_score(y_test, y_pred) * 100
# 	total_acc_knn += acc_knn

# 	#------------Bayes-------------
# 	Bayes.fit(X_train, y_train)
# 	y_pred = Bayes.predict(X_test)
# 	acc_bayes = accuracy_score(y_test, y_pred) * 100
# 	total_acc_bayes += acc_bayes

# 	arrTree.append(round(acc_tree, 2))
# 	arrKNN.append(round(acc_knn, 2))
# 	arrBayes.append(round(acc_bayes, 2))

# 	print("Do chinh xac Tree: ", round(acc_tree, 2), "%", "\nDo chinh xac KNN: ", round(acc_knn, 2), "%", "\nDo chinh xac Bayes: ", round(acc_bayes, 2), "%\n")

# print("Tree", arrTree)
# print("KNN", arrKNN)
# print("Bayes", arrBayes)

# print("\nDo chinh xac TB:\nTree : ", round(float(total_acc_tree/10), 2), "%\nKNN : ", round(total_acc_knn/10, 2), "%\nBayes : ", round(total_acc_bayes/10, 2), "%")