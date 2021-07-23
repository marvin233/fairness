def precision(true, predict):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(len(predict)):
        if true[i] == 0 and predict[i] == 0:
            tp += 1
        elif true[i] == 0 and predict[i] == 1:
            fn += 1
        elif true[i] == 1 and predict[i] == 0:
            fp += 1
        elif true[i] == 1 and predict[i] == 1:
            tn += 1
    return tp/(tp+fp)


def accuracy(true, predict):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(len(predict)):
        if true[i] == 0 and predict[i] == 0:
            tp += 1
        elif true[i] == 0 and predict[i] == 1:
            fn += 1
        elif true[i] == 1 and predict[i] == 0:
            fp += 1
        elif true[i] == 1 and predict[i] == 1:
            tn += 1
    return (tp+tn) / (tp+tn+fn+fp)


def recall(true, predict):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(len(predict)):
        if true[i] == 0 and predict[i] == 0:
            tp += 1
        elif true[i] == 0 and predict[i] == 1:
            fn += 1
        elif true[i] == 1 and predict[i] == 0:
            fp += 1
        elif true[i] == 1 and predict[i] == 1:
            tn += 1
    return tp / (tp + fn)


def f1_score(true, predict):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(len(predict)):
        if true[i] == 0 and predict[i] == 0:
            tp += 1
        elif true[i] == 0 and predict[i] == 1:
            fn += 1
        elif true[i] == 1 and predict[i] == 0:
            fp += 1
        elif true[i] == 1 and predict[i] == 1:
            tn += 1
    return (2*tp) / (2*tp+fp+fn)