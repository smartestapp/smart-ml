import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# df1 = pd.read_excel('~/Desktop/smarttest-aws/data/btnx_field_labels.xlsx') ## NOTE: Configure the sheet_name if necessary (i.e. more than one sheet)
## df1 = pd.read_excel('~/Desktop/smarttest-aws/data/try.xlsx', sheet_name='consensus_labels')
# df2 = pd.read_csv('btnx_mayo_eval.csv')
# OUTPUT_ID = 'btnx_mayo_eval_all'
# NUM_ZONES = 3

# df1 = pd.read_excel('~/Desktop/smarttest-aws/data/try.xlsx', sheet_name='consensus_labels') ## NOTE: Configure the sheet_name if necessary (i.e. more than one sheet)
# df2 = pd.read_csv('btnx_mayo1_eval.csv')
# OUTPUT_ID = 'btnx_mayo1_eval_all'
# NUM_ZONES = 3

# df1 = pd.read_excel('~/Desktop/smarttest-aws/data/deepblue_labels.xlsx') ## NOTE: Configure the sheet_name if necessary (i.e. more than one sheet)
# df2 = pd.read_csv('deepblue_eval.csv')
# OUTPUT_ID = 'deepblue_eval_all'
# NUM_ZONES = 2

# df1 = pd.read_excel('~/Desktop/smarttest-aws/data/labels.xlsx') ## NOTE: Configure the sheet_name if necessary (i.e. more than one sheet)
# df2 = pd.read_csv('paper_test_eval.csv')
# OUTPUT_FILEPATH = 'paper_test_eval_all.csv'
# NUM_ZONES = 3

# df1 = pd.read_excel('~/Desktop/smarttest-aws/data/btnx_validation_labels.xlsx')
# df2 = pd.read_csv('validation_eval.csv')
# OUTPUT_FILEPATH = 'validation_eval_all.csv'
# NUM_ZONES = 3

# df1 = pd.read_excel('~/Desktop/smarttest-aws/data/aconab_labels.xlsx')
# df2 = pd.read_csv('aconab_eval.csv')
# OUTPUT_ID = 'aconab_eval_all'
# NUM_ZONES = 3

# df1 = pd.read_excel('~/Desktop/smarttest-aws/data/aconag_labels.xlsx')
# df2 = pd.read_csv('aconag_eval.csv')
# OUTPUT_ID = 'aconag_eval_all'
# NUM_ZONES = 2

# df1 = pd.read_excel('~/Desktop/smarttest-aws/data/rapidconnect_labels.xlsx')
# df2 = pd.read_csv('rapidconnectab_eval.csv')
# OUTPUT_ID = 'rapidconnectab_eval_all'
# NUM_ZONES = 3

# df1 = pd.read_excel('~/Desktop/smarttest-aws/data/paramountag_labels.xlsx')
# df2 = pd.read_csv('paramountag_eval.csv')
# OUTPUT_ID = 'paramountag_eval_all'
# NUM_ZONES = 2

# df1 = pd.read_excel('~/Desktop/smarttest-aws/data/quidelag_labels.xlsx')
# df2 = pd.read_csv('quidelagrsv_eval.csv')
# OUTPUT_ID = 'quidelagrsv_eval_all'
# NUM_ZONES = 2

# df1 = pd.read_excel('~/Desktop/smarttest-aws/data/quidelag_labels.xlsx')
# df2 = pd.read_csv('quidelagsars_eval.csv')
# OUTPUT_ID = 'quidelagsars_eval_all'
# NUM_ZONES = 2

# df1 = pd.read_excel('~/Desktop/smarttest-aws/data/quidelag_labels.xlsx')
# df2 = pd.read_csv('quidelag_eval.csv')
# OUTPUT_ID = 'quidelag_eval_all'
# NUM_ZONES = 2

# df1 = pd.read_excel('~/Desktop/smarttest-aws/data/accessbio_labels_visual_based.xlsx')
# df1 = pd.read_excel('~/Desktop/smarttest-aws/data/accessbio_labels_conc_based.xlsx')
# df1 = pd.read_excel('~/Desktop/smarttest-aws/data/accessbio_labels_visual_based_wo_1ng.xlsx')
# df2 = pd.read_csv('accessbio_eval.csv')
# OUTPUT_ID = 'accessbio_eval_visual_wo_1ng_tr_all'
# OUTPUT_ID = 'accessbio_eval_conc_all'
# NUM_ZONES = 2

df1 = pd.read_excel('~/Desktop/smarttest-aws/data/sialab_quidelag_labels.xlsx')
df2 = pd.read_csv('sialab_quidelag_V2_TR_eval.csv')
OUTPUT_ID = 'sialab_quidelag_TR_eval_all'
NUM_ZONES = 2

OUTPUT_CSV_FILEPATH = '%s.csv' % OUTPUT_ID
OUTPUT_TXT_FILEPATH = '%s.txt' % OUTPUT_ID
OUTPUT_TXT = []
df2['Ground Truth'] = None 
df2['Membrane Result'] = None

print('----------------------------------------------------------------')
print(df1.head())
print(df2.head())
print('----------------------------------------------------------------')

class PerformanceMetrics(object):
    def __init__(self, true_positive, false_positive, true_negative, false_negative):
        self.true_positive = true_positive if true_positive > 0 else 1e-15
        self.false_positive = false_positive
        self.true_negative = true_negative
        self.false_negative = false_negative

    def accuracy(self):
        numerator = self.true_positive + self.true_negative
        denominator = self.true_positive + self.true_negative + self.false_positive + self.false_negative
        return numerator / denominator

    def precision(self):
        numerator = self.true_positive
        denominator = self.true_positive + self.false_positive
        return numerator / denominator

    def recall(self):
        numerator = self.true_positive
        denominator = self.true_positive + self.false_negative
        return numerator / denominator

    def f1(self):
        numerator = 2 * self.precision() * self.recall()
        denominator = self.precision() + self.recall()
        return numerator / denominator


NUM_ZONE_CORRECT, NUM_ZONE_TOTAL, NUM_MEMBRANE_CORRECT, NUM_MEMBRANE_TOTAL, NUM_SKIPPED, NUM_THRESHOLD_REJECTED = 0, 0, 0, 0, 0, 0
TP, FP, TN, FN = 0, 0, 0, 0

incorrect_confidence_scores = []
incorrect_confidence_scores_full = []
correct_confidence_scores = []
correct_confidence_scores_full = []

THRESHOLD_REJECTED_IDS = []

for index, row2 in tqdm(df2.copy().iterrows(), desc='Checking Agreement'):
    if 'btnx' in OUTPUT_ID:
        sample_id = row2['Sample ID'] # [:-4] ## DO THIS TO GET RID OF .JPG or .PNG at the end, have to do -5 for .JPEG
    else:
        sample_id = row2['Sample ID'][:-4] ## DO THIS TO GET RID OF .JPG or .PNG at the end, have to do -5 for .JPEG

    # sample_id = row2['Sample ID']
    prediction2 = [row2['Pred_%d' % i] for i in range(1, NUM_ZONES + 1)]
    prediction2_confidence_scores = [eval(row2['Pred_%d_Confidence' % i]) for i in range(1, NUM_ZONES + 1)]

    if sample_id in df1['Sample ID'].values.tolist():
        row1 = df1.loc[df1['Sample ID'] == sample_id, :]

        # Get predictions based on the column names
        # prediction1 = [row1['Pred_%d' % i].values.tolist()[0] for i in range(1, NUM_ZONES + 1)]
        prediction1 = [row1['Zone %d' % i].values.tolist()[0] for i in range(1, NUM_ZONES + 1)]

        df2.loc[index, 'Ground Truth'] = str(prediction1)

        # Skip Invalids!
        if 99 in prediction1:
            continue
            # prediction1[0] = 1  ## NOTE: This was a quick and dirty way to get rid of 99s to evaluate correcty, the classifier always returns 1 for these!

        # Skip anything that is not in 'test' split
        if 'Dataset' in row1:
            if row1['Dataset'].values.tolist()[0].lower() != 'test':
                continue

        # Skip Threshold Rejected Examples!
        if -1 in prediction2:

            NUM_THRESHOLD_REJECTED += 1
            continue

        NUM_MEMBRANE_TOTAL += 1
        if prediction1 == prediction2:
            NUM_MEMBRANE_CORRECT += 1
            df2.loc[index, 'Membrane Result'] = 'CORRECT'
        else:
            df2.loc[index, 'Membrane Result'] = 'WRONG'
            print('BAD SAMPLE ID: ', sample_id, 'GROUND TRUTH: ', prediction1, 'PREDICTION: ', prediction2)
            print(prediction2_confidence_scores)

            OUTPUT_TXT.append('BAD SAMPLE ID: %s, GROUND TRUTH: %s, PREDICTION: %s' % (str(sample_id), str(prediction1), str(prediction2)))

        NUM_ZONE_TOTAL += NUM_ZONES
        for i in range(NUM_ZONES):
            if prediction1[i] == prediction2[i]:
                NUM_ZONE_CORRECT += 1
            if prediction1[i] == prediction2[i] and prediction2[i] == 1:
                TP += 1
                correct_confidence_scores.append(max(prediction2_confidence_scores[i]))
                correct_confidence_scores_full.append(prediction2_confidence_scores[i])

            elif prediction1[i] == prediction2[i] and prediction2[i] == 0:
                TN += 1
                correct_confidence_scores.append(max(prediction2_confidence_scores[i]))
                correct_confidence_scores_full.append(prediction2_confidence_scores[i])

            elif prediction1[i] != prediction2[i] and prediction2[i] == 0:
                FN += 1
                incorrect_confidence_scores.append(max(prediction2_confidence_scores[i]))
                incorrect_confidence_scores_full.append(prediction2_confidence_scores[i])

            elif prediction1[i] != prediction2[i] and prediction2[i] == 1:
                FP += 1
                incorrect_confidence_scores.append(max(prediction2_confidence_scores[i]))
                incorrect_confidence_scores_full.append(prediction2_confidence_scores[i])

    else:
        NUM_SKIPPED += 1
        print('Skipping, sample ID not found: %s' % sample_id)


print('NUM. TOTAL: ', NUM_MEMBRANE_TOTAL)
OUTPUT_TXT.append('NUM. TOTAL: %d' % NUM_MEMBRANE_TOTAL)
print('NUM. SKIPPED MEMBRANES: ', NUM_SKIPPED)
OUTPUT_TXT.append('NUM. SKIPPED MEMBRANES: %d' % NUM_SKIPPED)
print('NUM. THRESHOLD REJECTED MEMBRANES: ', NUM_THRESHOLD_REJECTED)
OUTPUT_TXT.append('NUM. THRESHOLD REJECTED MEMBRANES: %d' % NUM_THRESHOLD_REJECTED)

print('Zone Accuracy: ', NUM_ZONE_CORRECT/NUM_ZONE_TOTAL)
OUTPUT_TXT.append('Zone Accuracy: %s' % str(NUM_ZONE_CORRECT/NUM_ZONE_TOTAL))
print('Membrane Accuracy: ', NUM_MEMBRANE_CORRECT/NUM_MEMBRANE_TOTAL)
OUTPUT_TXT.append('Membrane Accuracy: %s' % str(NUM_MEMBRANE_CORRECT/NUM_MEMBRANE_TOTAL))

metrics = PerformanceMetrics(true_positive=TP, false_positive=FP, true_negative=TN, false_negative=FN)

print('TP: %d \t FP: %d \t TN: %d \t FN: %d' % (TP, FP, TN, FN))
OUTPUT_TXT.append('TP: %d \t FP: %d \t TN: %d \t FN: %d' % (TP, FP, TN, FN))
print('Accuracy: %0.4f \t Recall: %0.4f \t Precision: %0.4f \t F1 Score: %0.4f' % (metrics.accuracy(), metrics.recall(), metrics.precision(), metrics.f1()))
OUTPUT_TXT.append('Accuracy: %0.4f \t Recall: %0.4f \t Precision: %0.4f \t F1 Score: %0.4f' % (metrics.accuracy(), metrics.recall(), metrics.precision(), metrics.f1()))
print('----------------------------------------------------------------------------------')

df2.to_csv(OUTPUT_CSV_FILEPATH)

with open(OUTPUT_TXT_FILEPATH, 'w') as f:
    for item in OUTPUT_TXT:
        f.write("%s\n" % item)


## HISTOGRAMS ##

hist = plt.hist(incorrect_confidence_scores, bins=20)
for i in range(20):
    plt.text(hist[1][i], hist[0][i], str(int(hist[0][i])))
plt.title('Incorrect Classification (Total: %d)' % len(incorrect_confidence_scores))
plt.grid()
plt.ylabel('Num. Zones')
plt.xlabel('Confidence Scores')
plt.xticks(np.arange(0.5, 1.1, 0.1))
plt.xlim(0.5, 1.0)
plt.savefig('incorrect_confidence_scores.png')
plt.show()

hist = plt.hist(correct_confidence_scores, bins=20)
for i in range(20):
    plt.text(hist[1][i], hist[0][i], str(int(hist[0][i])))
plt.title('Correct Classification (Total: %d)' % len(correct_confidence_scores))
plt.grid()
plt.ylabel('Num. Zones')
plt.xlabel('Confidence Scores')
plt.xticks(np.arange(0.5, 1.1, 0.1))
plt.xlim(0.5, 1.0)
plt.savefig('correct_confidence_scores.png')
plt.show()


THRESHOLD_ANALYSIS = False
if THRESHOLD_ANALYSIS:
    threshold_analysis_df = pd.DataFrame({}, columns=['threshold_score', 'TP', 'FP', 'TN', 'FN', 'num_rejected_zones', 'num_rejected_zones_percentage', 'Accuracy', 'Recall', 'Precision', 'F1 Score'])
    for k, threshold_score in enumerate(np.linspace(0.5, 1.00, 51)):
        TP_cur, FP_cur, TN_cur, FN_cur, num_rejected = TP, FP, TN, FN, 0
        for c in correct_confidence_scores_full:
            if max(c) < threshold_score:
                num_rejected += 1
                if c.index(max(c)) == 0:
                    TN_cur -= 1
                elif c.index(max(c)) == 1:
                    TP_cur -= 1

        for i in incorrect_confidence_scores_full:
            if max(i) < threshold_score:
                num_rejected += 1
                if i.index(max(i)) == 0:
                    FN_cur -= 1
                elif i.index(max(i)) == 1:
                    FP_cur -= 1


        metrics = PerformanceMetrics(true_positive=TP_cur, false_positive=FP_cur, true_negative=TN_cur, false_negative=FN_cur)
        threshold_analysis_df.loc[k] = ['%0.2f' % threshold_score, TP_cur, FP_cur, TN_cur, FN_cur, num_rejected, '%0.2f' % ((num_rejected / NUM_ZONE_TOTAL) * 100), '%0.4f' % metrics.accuracy(), '%0.4f' % metrics.recall(), '%0.4f' % metrics.precision(), '%0.4f' % metrics.f1()]

    threshold_analysis_df.to_csv(OUTPUT_ID + '_threshold_analysis.csv')

