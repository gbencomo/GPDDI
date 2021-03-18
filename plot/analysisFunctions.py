import matplotlib.pyplot as plt
import numpy as np

# Reciever Operating Characteristic
def ROC(dat, bins=41):
    dat.sort(key=lambda tup: tup[0][1])

    probability_thresholds = np.linspace(0, 1, num=bins)
    tpr, fpr = [], []

    for j in range(0, len(probability_thresholds)):
        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(0, len(dat)):
            c = (dat[i][0][1] > probability_thresholds[j])

            if dat[i][2] == c == 1:
                tp += 1
            if c == 1 and dat[i][2] != c:
                fp += 1
            if dat[i][2] == c == 0:
                tn += 1
            if c == 0 and dat[i][2] != c:
                fn += 1

        if tp != 0:
            tpr.append(tp / (tp + fn))
        else:
            tpr.append(0.0)
        if fp != 0:
            fpr.append(fp / (fp + tn))
        else:
            fpr.append(0.0)

    # Simple Riemann Sum to calculate integral for AUC
    bin_width = 1 / (bins - 1)

    auc = 0
    for i in tpr:
        auc += i * bin_width


    plt.figure()
    plt.plot(fpr, tpr, 'b', [0,1],[0,1], '--k')
    plt.legend([f'GP Classifier (AUC = {auc})', 'Random Classification'])
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    return auc


# 1st order analysis functions with data.npz
def plot_histo(filepath, params=[], bins=41):
    D = np.load(filepath, allow_pickle=True)['save_list']

    for i in range(0, len(params)):
        pos = D[1]['Positive DDIs'][i + 2][params[i]]
        neg = D[2]['All Negative DDIs'][i + 2][params[i]]

        pos_prop = []
        neg_prop = []
        x = []

        bins = np.linspace(0, 1, 41)

        for k in range(1, len(bins)):
            count_p = 0
            count_n = 0
            for j in range(0, len(pos)):
                if pos[j] >= bins[k - 1] and pos[j] < bins[k]:
                    count_p += 1
            for j in range(0, len(neg)):
                if neg[j] >= bins[k - 1] and neg[j] < bins[k]:
                    count_n += 1


            pos_prop.append(count_p / len(pos))
            neg_prop.append(count_n / len(neg))
            x.append((bins[k - 1] + bins[k]) / 2)

        mean_p = sum(pos) / len(pos)
        mean_n = sum(neg) / len(neg)
        print(mean_p)
        print(mean_n)

        plt.figure()
        plt.plot(x, pos_prop, 'r', x, neg_prop, 'b')
        plt.vlines(mean_p, 0, max(pos_prop), 'r', linestyles='dotted')
        plt.vlines(mean_n, 0, max(neg_prop), 'b', linestyles='dotted')
        plt.title(params[i])
        plt.legend(['DDI +', 'DDI -', '+ Mean ({:.3f})'.format(mean_p), '- Mean ({:.3f})'.format(mean_n)])
        plt.xlabel(params[i])
        plt.ylabel('Proportion')


#plot_histo('/Users/gianlucabencomo/PycharmProjects/DDI/data.npz', params=['Structural Similarity', 'Genotypic Similarity', 'Therapeutic Similarity', 'Phenotypic Similarity'])
#plt.show()