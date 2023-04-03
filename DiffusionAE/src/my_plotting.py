import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statistics
import os, torch
import numpy as np

#plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2

os.makedirs('plots', exist_ok=True)

def smooth(y, box_pts=1):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
 
# def plotter(ground_truth, prediction, ascore, labels, preds=False):
# 	dim = 0
# 	y_t, y_p, l, a_s = ground_truth[0:1000, dim], prediction[0:1000, dim], labels[0:1000, dim], ascore[0:1000, dim]
# 	plt.clf()
# 	fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# 	ax1.set_ylabel('Value')
# 	ax1.set_title(f'Dimension = {dim}')
# 	ax1.plot(smooth(y_t), linewidth=0.2, label='True')
# 	ax1.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.3, label='Predicted')
# 	ax3 = ax1.twinx()
# 	ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
# 	ax3.fill_between(np.arange(l.shape[0]), l, color='yellow', alpha=0.3)
# 	#ax4 = ax2.twinx()
# 	#ax4.plot(pred, '--', linewidth=0.3, alpha=0.5)
# 	#ax4.fill_between(np.arange(l.shape[0]), pred, color='yellow', alpha=0.3)
# 	ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
# 	ax2.plot(smooth(a_s), linewidth=0.2, color='g')
# 	ax2.set_xlabel('Timestamp')
# 	ax2.set_ylabel('Anomaly Score')
# 	plt.savefig('/root/Diff-Anomaly/DiffusionAE/plots/AAAAAA.jpg')
# 	plt.close()


def plotter(model, dataset, ground_truth, anomaly_score, labels, results=None, ae_recon=None, diff_sample=None, preds=None, dim=0, plot_test=True, epoch=0, set='test'):
    if ground_truth.shape[-1] < 6:
        timestamps = 4000
    else:
        timestamps = len(labels)
    print(timestamps)
    #timestamps = len(ground_truth)
    gt = ground_truth[0:timestamps, dim]
    labels = labels[0:timestamps]
    score = anomaly_score[0:timestamps]
    preds = results['preds']
    preds = preds[0:timestamps]
    trained_ae = True
    trained_diff = True
    thresh = results['thresh_max']
    text = f"{model},{dataset}\nROC_k = %.2f, F1_k = %.2f, ROC_max = %.2f, F1_max = %.2f\n best_k = %i, best_th = %.4f"%(results['ROC/AUC'], results['f1'], results['roc_max'], results['f1_max'], results['k'], results['thresh_max'])
    TP = [1 if preds[i] and labels[i] else 0 for i in range(0, timestamps)]
    FP = [1 if preds[i] and not labels[i] else 0 for i in range(0, timestamps)]
    FN = [1 if not preds[i] and labels[i] else 0 for i in range(0, timestamps)]

    try:
        ae = ae_recon[0:timestamps, dim]
    except:
        trained_ae = False

    try:
        diff = diff_sample[0:timestamps, dim]
    except:
        trained_diff = False

    if trained_ae and trained_diff:
        if plot_test:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
            # ax1.set_ylabel('Series')
            # ax1.set_title(f'Dimension = {dim}')
            ax1.set_title(text, fontsize=7)
            ax1.plot(gt, linewidth=0.2)
            ax1.set_ylim(0, 1)
            #ax1.plot(preds, linewidth=0.2, color='orange')
            ax1.plot(labels, '--', linewidth=0.3, color='red')

            ax2.set_ylim(0,1)
            ax2.plot(ae, linewidth=0.2)
            # ax2.plot(TP, '--', linewidth=0.3, color='green', label='TP')
            # ax2.plot(FP, '--', linewidth=0.3, color='orange', label = 'FP')
            # ax2.plot(FN, '--', linewidth=0.3, color='blue', label='FN')
            ax2.legend(loc=(-0.1, -0.2), borderaxespad=0, fontsize='xx-small')
             
            ax3.set_ylim()
            ax3.plot(diff, linewidth=0.2, label='diff')
            # ax3.plot(TP, '--', linewidth=0.3, color='green')
            # ax3.plot(FP, '--', linewidth=0.3, color='orange')
            # ax3.plot(FN, '--', linewidth=0.3, color='blue')
            ax2.fill_between(np.arange(labels.shape[0]), TP, color='green', alpha=0.2, linestyle='dashed', linewidth=0.3, label='TP')
            ax2.fill_between(np.arange(labels.shape[0]), FP, color='orange', alpha=0.3, linestyle='dashed', linewidth=0.3, label='FP')
            ax2.fill_between(np.arange(labels.shape[0]), FN, color='blue', alpha=0.2, linestyle='dashed', linewidth=0.3, label='FN')

            #ax4.fill_between(np.arange(l.shape[0]), pred, color='yellow', alpha=0.3)

            # ax2.plot(smooth(gt), linewidth=0.2, label='True')
            # ax2.plot(smooth(diff), linewidth=0.2, label='Diff')
            th = [thresh] * timestamps

            ax4.plot(score, linewidth=0.2)
            ax4.set_xlabel('Timestamp')
            ax4.set_ylabel('Score')
            ax4.plot(th, '--', linewidth=0.2, alpha=0.5)
            
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
            ax2.set_ylim(min(ae),max(ae))
            ax3.set_ylim(min(diff),max(diff))
            ax1.plot(gt, linewidth=0.2, label='True')
            ax2.plot(ae, linewidth=0.2, label='AE')
            ax3.plot(diff, linewidth=0.2, label='diff')
            #ax1.set_ylim(0,1)
    else:
        fig, (ax1, ax2, ax4) = plt.subplots(3, 1, sharex=True)
        # ax1.set_ylabel('Series')
        # ax1.set_title(f'Dimension = {dim}')
        ax1.set_title(text, fontsize=7)
        ax1.plot(gt, linewidth=0.2)
        ax1.set_ylim(0, 1)
        #ax1.plot(preds, linewidth=0.2, color='orange')
        #ax1.plot(labels, '--', linewidth=0.3, color='red')
        ax1.fill_between(np.arange(labels.shape[0]), labels, color='red', alpha=0.2, linestyle='dashed', linewidth=0.3)

        ax2.set_ylim(0,1)
        if trained_ae:
            ax2.plot(ae, linewidth=0.2)
        else:
            ax2.plot(diff, linewidth=0.2)
        # ax2.plot(TP, '--', linewidth=0.3, color='green', label='TP')
        # ax2.plot(FP, '--', linewidth=0.3, color='orange', label = 'FP')
        # ax2.plot(FN, '--', linewidth=0.3, color='blue', label='FN')
        ax2.fill_between(np.arange(labels.shape[0]), TP, color='green', alpha=0.2, linestyle='dashed', linewidth=0.3, label='TP')
        ax2.fill_between(np.arange(labels.shape[0]), FP, color='orange', alpha=0.3, linestyle='dashed', linewidth=0.3, label='FP')
        ax2.fill_between(np.arange(labels.shape[0]), FN, color='blue', alpha=0.2, linestyle='dashed', linewidth=0.3, label='FN')
        ax2.legend(loc=(-0.15, -0.2), borderaxespad=0, fontsize='xx-small')
            
        # ax3.plot(TP, '--', linewidth=0.3, color='green')
        # ax3.plot(FP, '--', linewidth=0.3, color='orange')
        # ax3.plot(FN, '--', linewidth=0.3, color='blue')

        #ax4.fill_between(np.arange(l.shape[0]), pred, color='yellow', alpha=0.3)

        # ax2.plot(smooth(gt), linewidth=0.2, label='True')
        # ax2.plot(smooth(diff), linewidth=0.2, label='Diff')
        th = [thresh] * timestamps

        ax4.plot(score, linewidth=0.2)
        ax4.set_xlabel('Timestamp')
        ax4.set_ylabel('Score')
        ax4.plot(th, '--', linewidth=0.2, alpha=0.5)

    
    if dataset:
        folder = f'../../../../plots/plots3/{model}_{dataset}'
    else:
        folder = f'../../../../plots/plots3/{model}'
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f'{folder}/dim_{dim}_epoch_{epoch}.jpg')
    plt.close()
    return fig
