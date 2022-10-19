
from collections import Counter
import random
import pandas as pd
import matplotlib.pyplot as plt
import copy
from decision_tree import DecisionTree
from sklearn import metrics

df = pd.read_excel("./datos_de_entrenamiento.xlsx")
df2 = pd.read_excel("./datos_de_prueba.xlsx")
df3 = pd.read_excel("./ssss.xlsx")
##data transformada
data_classification = df.to_numpy()
##sin transformar
classification_data = pd.read_excel("./datos_de_entrenamiento.xlsx")
test_data_1 = df2.to_numpy()
data = df3.to_numpy()
attributes = {
    'account_balance': {1: {1: 0,
              2: 0,
              3: 0,
              4: 0
              },
          0: {
        1: 0,
        2: 0,
        3: 0,
        4: 0



    }},
    'marital_status': {1: {1: 0,
              2: 0,
              3: 0,
              4: 0
              },
          0: {
        1: 0,
        2: 0,
        3: 0,
        4: 0

    }},
    'mv_avaiable_asset': {1: {1: 0,
              2: 0,
              3: 0,
              4: 0
              },
          0: {
        1: 0,
        2: 0,
        3: 0,
        4: 0


    }},
    'type_of_apartament': {1: {1: 0,
              2: 0,
              3: 0,

              },
          0: {
        1: 0,
        2: 0,
        3: 0,


    }},
    'concurrent_credits': {1: {1: 0,
              2: 0,
              3: 0,

              },
          0: {
        1: 0,
        2: 0,
        3: 0,


    }}
}
example = {
    'pronostico': {1: {'soleado': 0,
                       'nublado': 0,
                       'lluvioso': 0,

                       },
                   0: {
        "soleado": 0,
        "nublado": 0,
        "lluvioso": 0,




    }},
    'temperatura': {1: {"calido": 0,
                        "templado": 0,
                        "frio": 0,

                        },
                    0: {"calido": 0,
                        "templado": 0,
                        "frio": 0,

                        }},
    'humedad': {1: {"alta": 0,
                    "normal": 0,

                    },
                0: {"alta": 0,
                    "normal": 0,

                    }


                },
    'viento': {1: {'debil': 0,
                   'fuerte': 0,


                   },
               0: {'debil': 0,
                   'fuerte': 0,


                   }}
}


def calculate_metrics(arr_true_class, arr_2):
        tp, tn, fp, fn = 0, 0, 0, 0
        result = {}
        for i in range(len(arr_true_class)):
            if (arr_true_class[i] == 1 and arr_2[i] == 1):
                tp += 1
            elif (arr_true_class[i] == 0 and arr_2[i] == 0):
                tn += 1
            elif ((arr_true_class[i] == 0 and arr_2[i] == 1)):
                fp += 1
            elif ((arr_true_class[i] == 1 and arr_2[i] == 0)):
                fn += 1
        result['accuracy'] = (tp+tn)/(tp+tn+fp+fn)
        result['precision'] = tp/(tp+fp)
        result['recall'] = tp/(tp+fn)
        result['f1_score'] = (2*result['precision']*result['recall']) / \
            (result['precision'] + result['recall'])
        result['tp_rate'] = tp/(tp+fn)
        result['fp_rate'] = fp/(tn+fp)
        result['tp'] = tp
        result['tn'] = tn
        result['fn'] = fn
        result['fp'] = fp
        return result

def get_cred_Ability_column(arr):
        return list(map(lambda element: element[0], arr))
    
    
def random_forest(_list):
    arr=copy.deepcopy(list(_list))
    length_data_test = 201
    data_test_random = []
    for i in range(length_data_test):
        data_test_random.append(arr.pop(random.randint(0,len(arr)-1)))
    return arr, data_test_random

def random_forest(dataset,numbero_de_arboles):
    # Lista para ir guardando cada árbol
    trees_random_forest = []
    
    # For para la cantidad de arboles a crear
    for i in range(numbero_de_arboles):
        # Crea los datos aleatorios del bootstraping
        # De esta forma usa datos random del conjunto
        data_bootstrap = dataset.sample(frac=1,replace=True)
        # Se agrega a la lista el arbol que se crea con el conjunto de datos 
        ##Lista de atributos sin la clase primaria
        tree = DecisionTree(attributes,data_bootstrap.to_numpy())
        tree.fit()
        trees_random_forest.append(tree)
    
    return trees_random_forest
        
def random_forest_predict(test_data,random_forest):    
    predictions = []
    #Para cada arbol
    for tree in random_forest:
        #Obtenemos los valores de cada test
        predictions.append(tree.test(test_data))
    
    
    
    #Se agrupan los valores
    group_values = []
 
    for i in range(len(test_data)):
        rows = []
        for row in predictions:
            rows.append(row[i])
        group_values.append(rows)
    
 
    result = []
    #Se saca el maximo valor de cada atributo
    for row in group_values:
        data = Counter(row)
        result.append(max(data, key=data.get))
                      
    

    return result



tree = DecisionTree(attributes, data_classification)
tree.fit()



print("RESULTADO DATOS DE PRUEBA: ")

result_test_data = tree.test(test_data_1)
arr_true_class_test_data = get_cred_Ability_column(test_data_1) 
test_data_metrics= calculate_metrics(arr_true_class_test_data, result_test_data)
confusion_matrix = metrics.confusion_matrix(arr_true_class_test_data, result_test_data)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.title("RESULTADO DATOS DE PRUEBA")
plt.ylabel('Clase resultado')
plt.xlabel('Clase verdadera')
plt.show()
print("ACCURACY: ", test_data_metrics['accuracy'])
print("PRECISION: ", test_data_metrics['precision'])
print("")
print("RESULTADO DATOS DE ENTRENAMIENTO: ")

result_data_classification = tree.test(data_classification)
arr_true_class_data = get_cred_Ability_column(data_classification) 
train_data_metrics= calculate_metrics(arr_true_class_data, result_data_classification)
confusion_matrix = metrics.confusion_matrix(arr_true_class_data, result_data_classification)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.title("RESULTADO DATOS DE ENTRENAMIENTO")
plt.ylabel('Clase resultado')
plt.xlabel('Clase verdadera')
plt.show()
print("ACCURACY: ", train_data_metrics['accuracy'])
print("PRECISION: ", train_data_metrics['precision'])



##RANDOM FOREST
random_forest_result = random_forest(classification_data, 5)

result_random_forest_predict_test_data =random_forest_predict(test_data_1, random_forest_result)
result_random_forest_predict_train_data = random_forest_predict(data_classification, random_forest_result)
print("")
print("")
print("PUNTO F - RESULTADO DATOS DE PRUEBA: ")

arr_true_class_test_data = get_cred_Ability_column(test_data_1) 
test_data_metrics= calculate_metrics(arr_true_class_test_data, result_random_forest_predict_test_data)
confusion_matrix = metrics.confusion_matrix(arr_true_class_test_data, result_random_forest_predict_test_data)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.title("PUNTO F - RESULTADO DATOS DE PRUEBA")
plt.ylabel('Clase resultado')
plt.xlabel('Clase verdadera')
plt.show()
print("ACCURACY: ", test_data_metrics['accuracy'])
print("PRECISION: ", test_data_metrics['precision'])
print("")
print("PUNTO F - RESULTADO DATOS DE ENTRENAMIENTO: ")

result_data_classification = tree.test(data_classification)
arr_true_class_data = get_cred_Ability_column(data_classification) 

train_data_metrics= calculate_metrics(arr_true_class_data, result_random_forest_predict_train_data)
confusion_matrix = metrics.confusion_matrix(arr_true_class_data, result_random_forest_predict_train_data)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.title("PUNTO F - RESULTADO DATOS DE ENTRENAMIENTO")
plt.ylabel('Clase resultado')
plt.xlabel('Clase verdadera')
plt.show()
print("ACCURACY: ", train_data_metrics['accuracy'])
print("PRECISION: ", train_data_metrics['precision'])


