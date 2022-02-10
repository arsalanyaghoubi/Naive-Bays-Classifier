import os
import numpy as np
from math import log
from collections import defaultdict,Counter
class NaiveBayes():

    def __init__(self):
        self.class_dict = {'neg': 0, 'pos': 1}
        self.prior = defaultdict(int)
        self.likelihood = defaultdict(dict)
        self.class_document_count = defaultdict(int)
        self.total_document = 0
        self.class_feature_count = defaultdict(Counter)
        self.feature_count = defaultdict(int)
        self.prior_array = np.zeros(len(self.class_dict), )
        self.likelihood_array = []
        self.feature_index = defaultdict(int)
        self.class_count= defaultdict(int)

    def train(self, train_set):
        for root, dirs, files in os.walk(train_set):
            for name in files:
                class_name = root.split('\\')[-1]
                self.class_document_count[class_name] += 1
                self.total_document += 1
                with open(os.path.join(root, name)) as f:
                    text = f.read()
                    list_text = text.split()
                    for word in list_text:
                        self.class_count[class_name]+=1
                        self.class_feature_count[class_name][word]+= 1
                        self.feature_count[word]+=1
        feature_set_list = list(self.feature_count.keys())
        self.likelihood_array = np.zeros((len(self.class_dict), len(self.feature_count)))
        for clas in self.class_dict:
            self.prior[clas]= log(self.class_document_count[clas]/ self.total_document)
            if clas == 'neg':
                self.prior_array[0]= self.prior[clas]
            elif clas == 'pos':
                self.prior_array[1]= self.prior[clas]

            for feature in self.feature_count:
                self.feature_index[feature] = feature_set_list.index(feature)
                self.likelihood[clas][feature] = log((self.class_feature_count[clas][feature] + 1)/
                                                  (self.class_count[clas]+ len(self.feature_count)))
                self.likelihood_array[self.class_dict[clas]][self.feature_index[feature]]= self.likelihood[clas][feature]

    def test(self, dev_set):
        results = defaultdict(dict)
        for root, dirs, files in os.walk(dev_set):
            for name in files:
                class_name = root.split('\\')[-1]
                feature_vector = np.zeros((len(self.feature_count),))
                with open(os.path.join(root, name)) as f:
                    text = f.read()
                    list_text= text.split()
                    for token in list_text:
                        if token in self.feature_count:
                            word_index = self.feature_index[token]
                            feature_vector[word_index] += 1

                    result_matrix = np.dot(self.likelihood_array,feature_vector)
                    final_matrix = np.add(result_matrix, self.prior_array)
                    max_index = np.argmax(final_matrix)
                    if max_index==0:
                        prediction = 'neg'
                    else:
                        prediction = 'pos'
                    results[name]['correct']= class_name
                    results[name]['predicted'] = prediction
        return results


    def evaluate(self, results):
        confusion_matrix = np.zeros((len(self.class_dict),len(self.class_dict)))
        precision = defaultdict(float)
        recall = defaultdict(float)
        F1 = defaultdict(float)
        for items in results:
            if results[items]['correct'] == results[items]['predicted']:
                if results[items]['predicted'] == 'neg':
                    confusion_matrix[0][0] += 1
                if results[items]['predicted'] == 'pos':
                    confusion_matrix[1][1] += 1
            if results[items]['correct'] != results[items]['predicted']:
                if results[items]['predicted'] == 'neg':
                    confusion_matrix[0][1] += 1
                if results[items]['predicted'] == 'pos':
                    confusion_matrix[1][0] += 1

        print(confusion_matrix)
        print(" ")
        for clas in self.class_dict:
            if clas == 'neg':
                precision[clas] = confusion_matrix[0][0] / (
                            confusion_matrix[0][0] + confusion_matrix[0][1])
                recall[clas] = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
                F1[clas] = (2 * precision[clas] * recall[clas]) / (precision[clas] + recall[clas])
                print("Precision for ", clas, " is ", precision[clas])
                print("Recall for ", clas, " is ", recall[clas])
                print("F1 for ", clas, " is ", F1[clas])
            else:
                precision[clas] = confusion_matrix[1][1] / (
                            confusion_matrix[1][1] + confusion_matrix[1][0])
                recall[clas] = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])
                F1[clas] = (2 * precision[clas] * recall[clas]) / (precision[clas] + recall[clas])
                print("Precision for ", clas, " is ", precision[clas])
                print("Recall for ", clas, " is ", recall[clas])
                print("F1 for ", clas, " is ", F1[clas])

        accuracy = ((confusion_matrix[0][0] + confusion_matrix[1][1]) / (
                confusion_matrix[0][0] + confusion_matrix[0][1] +
                confusion_matrix[1][0] + confusion_matrix[1][1]))
        print('Overal Accuracy: ', accuracy)

if __name__ == '__main__':
    nb = NaiveBayes()
    nb.train("INSERT THE LINK TO THE TRAINING DATASET")
    results = nb.test("INSERT THE LINK TO THE TEST DATASET")
    nb.evaluate(results)
