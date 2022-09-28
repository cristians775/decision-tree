
import random
import pandas as pd
import matplotlib.pyplot as plt
import copy
from decision_tree import DecisionTree
df = pd.read_excel("./datos_de_entrenamiento.xlsx")
df2 = pd.read_excel("./datos_de_prueba.xlsx")
df3 = pd.read_excel("./ssss.xlsx")
data_classification = df.to_numpy()
test_data_1 = df2.to_numpy()
data = df3.to_numpy()
attributes = {
    'a': {1: {1: 0,
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
    'b': {1: {1: 0,
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
    'c': {1: {1: 0,
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
    'd': {1: {1: 0,
              2: 0,
              3: 0,

              },
          0: {
        1: 0,
        2: 0,
        3: 0,


    }},
    'e': {1: {1: 0,
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

def roc(fp_rate_list,tp_rate_list):
    plt.figure()
    plt.title('Curva ROC')
    plt.plot(fp_rate_list, tp_rate_list, marker=".", label="MLP")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.xlabel('Tasa de falsos positivos')
    plt.show()

tree = DecisionTree(attributes, data_classification)
tree.fit()
 
result = tree.test(test_data_1)

arr_true_class = get_cred_Ability_column(test_data_1) 


data_classification_1, data_test_1 = random_forest(data_classification)
data_classification_2, data_test_2 = random_forest(data_classification)
data_classification_3, data_test_3 = random_forest(data_classification)
data_classification_4, data_test_4 = random_forest(data_classification)
data_classification_5, data_test_5 = random_forest(data_classification)

tree_1 = DecisionTree(attributes, data_classification_1)
tree_2 = DecisionTree(attributes, data_classification_2)
tree_3 = DecisionTree(attributes, data_classification_3)
tree_4 = DecisionTree(attributes, data_classification_4)
tree_5 = DecisionTree(attributes, data_classification_5)

tree_1.fit()
tree_2.fit()
tree_3.fit()
tree_4.fit()
tree_5.fit()


result_1 = tree_1.test(test_data_1)
result_2 = tree_2.test(test_data_1)
result_3 = tree_3.test(test_data_1)
result_4 = tree_4.test(test_data_1)
result_5 = tree_5.test(test_data_1)

metrics = calculate_metrics(arr_true_class, result)
metrics_1 = calculate_metrics(arr_true_class, result_1)
metrics_2 = calculate_metrics(arr_true_class, result_2)
metrics_3 = calculate_metrics(arr_true_class, result_3)
metrics_4 = calculate_metrics(arr_true_class, result_4)
metrics_5 = calculate_metrics(arr_true_class, result_5)

print(metrics)
print(metrics_1)
print(metrics_2)
print(metrics_3)
print(metrics_4)
print(metrics_5)

""" tree2 = DecisionTree(example,data)
tree2.fit()
fp_rate_list = [metrics['fp_rate'], metrics_1['fp_rate'], metrics_2['fp_rate'], metrics_3['fp_rate'], metrics_4['fp_rate'],
                metrics_5['fp_rate']]
tp_rate_list = [metrics['tp_rate'], metrics_1['tp_rate'], metrics_2['tp_rate'], metrics_3['tp_rate'], metrics_4['tp_rate'],
                metrics_5['tp_rate']]
roc(fp_rate_list, tp_rate_list)
 """