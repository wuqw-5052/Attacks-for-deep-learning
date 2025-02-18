
def evaluation(prediction, y):
    TP, FP, FN, TN = 0, 0, 0, 0
    for i in range(len(y)):
        Act, Pre = y[i], prediction[i]
        if Act == 0 and Pre == 0: TN += 1
        if Act == 0 and Pre != 0: FP += 1
        if Act != 0 and Pre == 0: FN += 1
        if Act != 0 and Pre != 0: TP += 1

    ## print result
    Acc = round(float(TP + TN) / float(TP + TN + FN + FP) * 100, 3)
    TPR = round(float(TP) / float(TP + FN) * 100, 3)
    TNR = round(float(TN) / float(FP+TN) * 100, 3)

    if (TP + FP)==0:
        Prec =0
    else:
        Prec = round(float(TP) / float(TP + FP) * 100, 2)
    if (TP + FN )==0:
        Recll =0
    else:
        Recll = round(float(TP) / float(TP + FN ) * 100, 2)
    if (Prec + Recll )==0:
        F1 =0
    else:
        F1 = round(2 * Prec * Recll / (Prec + Recll ) , 2)
    # print('TN:{}, FP: {}, FN: {}, TP: {}'.format(TN, FP, FN, TP))
    # print('TPR: {}, TNR: {}, Acc: {}, Prec: {}, F1: {}'.format(TPR, TNR, Acc, Prec, F1))
    gate_num = TN + FP + FN + TP
    lst1 = [gate_num, TN, FP, FN, TP,TPR, TNR, Acc, Prec, F1]
    lst2 = [Recll, Prec, F1, Acc]
    return lst2







