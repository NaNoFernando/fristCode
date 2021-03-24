from sklearn import tree
x=[[2,1.65,40],[1,1.66,45],[30,1.50,35],[45,1.60,33],[1,1.8,45],[1,1.7,45],[50,1.65,35],[2,1.8,33],[30,1.7,40]]
y=['M','M','F','F','M','M','F','F','M']
clasificador=tree.DecisionTreeClassifier(criterion="gini")#el criterio gini=c45, hay mas criterios y cada uno dependera de los datos que usemos, el gini es por defecto
clasificador.fit(x,y)
prediccion=clasificador.predict([[10,1.8,33],[41,1.7,40]])
print(prediccion)
print("==================================================================================================================")
from sklearn import datasets
iris=datasets.load_iris()
x=iris.data
y=iris.target
clasificador.fit(x,y)
yp=[[6.3,3.0,5.2,1],[5.5,3,1.4,1],[5.0,2,2,0.4]]
prediccion=clasificador.predict(yp)
print(prediccion)

from sklearn import model_selection
from sklearn.metrics import confusion_matrix
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2)
print("X_train datos****************")
print(len(x_train))
print(x_train[:2])

clasificador.fit(x_train,y_train)
prediccion=clasificador.predict(x_test)
print(prediccion)
print(confusion_matrix(y_test,prediccion))
print("X_test datos****************")
print(len(x_test))
print(x_test[:2])
