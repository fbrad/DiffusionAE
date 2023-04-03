import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt
from tadpak import pak

def get_fp_tp_rate(predict, actual):
    tn, fp, fn, tp = confusion_matrix(actual, predict, labels=[0, 1]).ravel()

    true_pos_rate = tp/(tp+fn)
    false_pos_rate = fp/(fp+tn)

    return false_pos_rate, true_pos_rate


def pak_protocol(scores, labels, threshold, max_k=100):
    f1s = []
    ks = [k/100 for k in range(0, max_k + 1)]
    fprs = []
    tprs = []
    preds = []

    for k in range(max_k +1):
        adjusted_preds = pak.pak(scores, labels, threshold, k=k)
        f1 = f1_score(labels, adjusted_preds)
        fpr, tpr = get_fp_tp_rate(adjusted_preds, labels)
        fprs.append(fpr)
        tprs.append(tpr)
        #print(f1)   
        #print(k)
        f1s.append(f1)
        preds.append(adjusted_preds)

    area_under_f1 = auc(ks, f1s)
    max_f1_k = max(f1s)
    k_max = f1s.index(max_f1_k)
    preds_for_max = preds[f1s.index(max_f1_k)]
    # import matplotlib.pyplot as plt
    # plt.cla()
    # plt.plot(ks, f1s)
    # plt.savefig('DiffusionAE/plots/PAK_PROTOCOL')
    #print(f'AREA UNDER CURVE {area}')
    return area_under_f1, max_f1_k, k_max, preds_for_max, fprs, tprs


def evaluate(score, label, validation_thresh=None):
    if len(score) != len(label):
        score = score[:len(score) - (len(score) - len(label))]
    false_pos_rates = []
    true_pos_rates = []
    f1s = []
    max_f1s_k = []
    preds = []
    #thresholds = np.arange(0, score.max(), min(0.001, score.max()/50))#0.001
    thresholds = np.arange(0, score.max(), score.max()/50)#0.001

    max_ks = []
    pairs = []


    for thresh in thresholds:
        f1, max_f1_k, k_max, best_preds, fprs, tprs = pak_protocol(score, label, thresh)
        max_f1s_k.append(max_f1_k)
        max_ks.append(k_max)
        preds.append(best_preds)
        false_pos_rates.append(fprs)
        true_pos_rates.append(tprs)
        f1s.append(f1)
        pairs.extend([(thresh, i) for i in range(101)])
    
    if validation_thresh:
        f1, max_f1_k, max_k, best_preds, _, _ = pak_protocol(score, label, validation_thresh)
    else:    
        f1 = max(f1s)
        max_possible_f1 = max(max_f1s_k)
        max_idx = max_f1s_k.index(max_possible_f1)
        max_k = max_ks[max_idx]
        thresh_max_f1 = thresholds[max_idx]
        best_preds = preds[max_idx]
        best_thresh = thresholds[f1s.index(f1)]
    
    roc_max = auc(np.transpose(false_pos_rates)[max_k], np.transpose(true_pos_rates)[max_k])
    #np.save('/root/Diff-Anomaly/DiffusionAE/plots_for_paper/fprs_diff_score_pa.npy', np.transpose(false_pos_rates)[0])
    #np.save('/root/Diff-Anomaly/DiffusionAE/plots_for_paper/tprs_diff_score_pa.npy', np.transpose(true_pos_rates)[0])
    
    false_pos_rates = np.array(false_pos_rates).flatten()
    true_pos_rates = np.array(true_pos_rates).flatten()

    sorted_indexes = np.argsort(false_pos_rates) 
    false_pos_rates = false_pos_rates[sorted_indexes]
    true_pos_rates = true_pos_rates[sorted_indexes]
    pairs = np.array(pairs)[sorted_indexes]
    roc_score = auc(false_pos_rates, true_pos_rates)

    #np.save('/root/Diff-Anomaly/DiffusionAE/plots_for_paper/tprs_diff_score.npy', true_pos_rates)
    #np.save('/root/Diff-Anomaly/DiffusionAE/plots_for_paper/fprs_diff_score.npy', false_pos_rates)
    #np.save('/root/Diff-Anomaly/DiffusionAE/plots_for_paper/pairs_diff_score.npy', pairs)
    #preds = predictions[f1s.index(f1)]
    if validation_thresh:
        return {
            'f1': f1,   # f1_k(area under f1) for validation threshold
            'ROC/AUC': roc_score, # for all ks and all thresholds obtained on test scores
            'f1_max': max_f1_k, # best f1 across k values
            'preds': best_preds, # corresponding to best k 
            'k': max_k, # the k value correlated with the best f1 across k=1,100
            'thresh_max': validation_thresh,
            'roc_max': roc_score,
        }
    else:
        return {
            'f1': f1,
            'ROC/AUC': roc_score,
            'threshold': best_thresh,
            'f1_max': max_possible_f1, 
            'roc_max': roc_max,
            'thresh_max': thresh_max_f1, 
            'preds': best_preds,
            'k': max_k,
        }, false_pos_rates, true_pos_rates
