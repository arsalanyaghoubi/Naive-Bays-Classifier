# Naive-Bays-Classifier
Application of the NB classifier (BoW faeture and add-1 smoothing) for a task of sentiment analysis (+ or -) 
Dataset: NLTK movie review corpus
Overall Accuracy: 75
Precision, Recall and F1 score for + : 74-77-75
Precision, Recall and F1 score for - : 75-72-74

To calculate the matrices for this simple task you could use the matricies as below:


def matrices(predicted,gold):
    predicted = [tensor.cpu() for tensor in predicted]
    results = classification_report(gold, predicted)
    print(results)
    print()
    cm = confusion_matrix(gold, predicted)
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()
