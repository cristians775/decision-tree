import copy
import json
import math
import operator



class DecisionTree:
    def __init__(self, attributes, classification_data,) -> None:
        self.classification_data = classification_data
        self.attributes = attributes
        self.tree = {}
        self.fields = {1:"a",2:"b",3:"c", 4:"d", 5:"e"} 

    def calculate_general_entropy(self, data):
        #por cada valor del nuevo atributo a evaluar se calcula la entropia
        sum_cred1 = 0
        sum_cred0 = 0
        data_classification_size = len(data)
        if data_classification_size==0:
            return 0
        for i in range(len(data)):
            if (data[i][0] == 1):
                sum_cred1 += 1
            else:
                sum_cred0 += 1

        prob_cred_1 = sum_cred1 / data_classification_size
        prob_cred_0 = sum_cred0 / data_classification_size
        result_1 = (- prob_cred_1 * math.log2(prob_cred_1)) if prob_cred_1 != 0 else 0
        result_0 = (- prob_cred_0 * math.log2(prob_cred_0)) if prob_cred_0 != 0 else 0
        return result_1 + result_0

    def fit(self):
        self.generate_desision_tree()

    def generate_desision_tree(self):
        attributes = copy.deepcopy(self.attributes)
        data = copy.deepcopy(self.classification_data)

        #ganancy = self.calculate_attributes_ganancy(attributes, data)
        #print("attributes ganancy", ganancy)
        
        #max_key = max(ganancy.items(), key=operator.itemgetter(1))[0]
        #tree = self.create_tree(max_key)
        # print("Tree",tree)
        #current_index = list(attributes).index(max_key)
        #print("Win", list(attributes).index(max_key))
        #del attributes[max_key]

        # print(tree[max_key])
        tree = self.make_tree(data, attributes)
        print("TrEee", json.dumps(tree, sort_keys=True, indent=4))
        self.tree = tree
        

    def best_attribute(self, data, attr):
        return max(self.calculate_attributes_ganancy(attr, data).items(), key=operator.itemgetter(1))[0]
    
    def test(self, list):
      
      list_transformed=self.transform_list(list)
      return self.get_values(self.tree, list_transformed)
      
    def make_tree(self, data, attr):
        truth = False

        tree = {}
        truth, record = self.unique(data)
        if(truth):
            tree = record
            
        elif len(attr) == 0:
            tree = self.maximum_value(data)
            
        else:
            
            A = self.best_attribute(data, attr)
            del attr[A]
            tree = {A: {}}
            
            values = list(self.attributes[A][0])
            
            for vi in values:
                index = list(self.attributes).index(A)+1
                examples = list(filter(lambda row: row[index] == vi, data))
                tree[A][vi] = self.make_tree(
                    examples, attr)

        return tree

    def maximum_value(self, data):
        #se devuelve el creditability que se repite mas veces
        #row 0 = creditability --> row[0] es el valor de la columna 0
        #count diccionario con las dos clases para contar los valores
        count = {0: 0, 1: 0}
        for row in data:
            count[row[0]] += 1
        return 1 if count[1] > count[0] else 0

    def unique(self, data):
        #se devuelve los dato si son todos negativos o positivos sino nada
        all_positive = all(
            list(map(lambda row: row[0] == 1  , data)))
        all_negative = all(
            list(map(lambda row: row[0] == 0  , data)))

            
        if all_positive and len(data)!=0:
            return True, 1
        elif all_negative and len(data)!=0:
            return True, 0
        else:
            return False, 0

    

    def calculate_attributes_ganancy(self, attributes, data):
        #se recorren las key de los atributos y se obtiene la frecuencia
        dic_result = {}

        for key in attributes:
            dic_result[key] = self.calculate_attribute_ganancy(key, data)
        return dic_result

    """ def all(self, _class, index, value, data):
        # print("##############################")
        #print("row ", index+1,"value", value)
        #print(all(map(lambda row: row[0] == _class and row[index+1] == value, data)))
        # for i in data:
        # print(i)
        # print(data)
        # print("##############################")
        return all(map(lambda row: row[0] == _class and row[index+1] == value, data)) """

    def calculate_attribute_ganancy(self, attr_key, data):
        #se obtiene el index del atributo(se le suma +1 para evitar el creditability)
        index = list(self.attributes).index(attr_key)+1
        #se calcula la entropia del conjunto S, por ejemplo el conjunto de los dias soleados
        general_entropy = self.calculate_general_entropy(data)
        attributes_entropies = {}
        attribute_dic = copy.deepcopy(self.attributes[attr_key])
        frequency_attributes_values = self.calculate_frequency(
            attribute_dic, index, data)
        #Se calcula la entropia de cada atributo a evaluar ver formula entropia
        for attribute in frequency_attributes_values[1]:
            attribute_class_1 = frequency_attributes_values[1][attribute]
            attribute_class_0 = frequency_attributes_values[0][attribute]
           
            attributes_entropies[attribute] = (-attribute_class_1*math.log2(
            attribute_class_1)if(attribute_class_1>0) else 0) +(- attribute_class_0*math.log2(attribute_class_0)if(attribute_class_0>0)else 0)
        result = 0
        ##print("GENERAL: ", general_entropy)
        #Se hace el calculo de la ganancia ver formula ganancia
        for attribute in attribute_dic[1]:
            attr_counter_class_1 = attribute_dic[1][attribute]
            attr_counter_class_0 = attribute_dic[0][attribute]
            sum_attr = attr_counter_class_0+attr_counter_class_1
            result += (-(sum_attr/len(data)) * attributes_entropies[attribute]) if sum_attr > 0 else 0
        
        return general_entropy+result

    def calculate_frequency(self, attribute, index, data):
       
        attributeAux = {0: {}, 1: {}}
        
        #se cuentan las veces que se repiten los atributos
        for row in range(len(data)):
            value = data[row][index]
            class_value = data[row][0]
            attribute[class_value][value] += 1
        #se calcula la frecuencia de cada atributo por cada clase
        for class_key in attribute:
            for attr_key in attribute[class_key]:
                    divisor = attribute[0][attr_key]+attribute[1][attr_key]
                    attributeAux[class_key][attr_key] = attribute[class_key][attr_key]/divisor if divisor > 0 else 0
                
        return attributeAux
      
    def transform_attribute(self,attribute):
        result = []
        for i in range(len(attribute)):
            if(i!= 0):
                result.append({'value':attribute[i], 'key':self.fields[i]})
        return result

    def transform_list(self,list):
        result = []
        for attribute in list:
            result.append(self.transform_attribute(attribute))
        return result


    def get_values(self,tree, attributes):
        result = []
        for attr in attributes:
            result.append(self.get_value(tree,attr))
    
        return result


    def get_value(self,tree,attribute):
       
        if(isinstance(tree,int)):
            return tree
        for attr in attribute:
            key=attr['key']
            value = attr['value']

            if(key in tree):
                result = tree[key][value]
                
               
        return self.get_value(result, attribute)
  

