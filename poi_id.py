
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("C:/Users/Bash/Desktop/Udacity/2_Data_Analysis/P5-ML/ud120-projects/tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data,load_classifier_and_data,test_classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFECV, SelectFdr
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score


### Task 1: Select what features you'll use.
### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Task 4: Try a varity of classifiers
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.

features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 
                 'total_stock_value', 'expenses', 'exercised_stock_options', 
                 'other', 'long_term_incentive', 'restricted_stock', 
                 'director_fees','to_messages', 
                 'from_poi_to_this_person', 'from_messages', 
                 'from_this_person_to_poi', 'shared_receipt_with_poi']
                 

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

## Recursive feature elimination
def feature_select(labels,features,alfa=0.4):
    dct = DecisionTreeClassifier(random_state=42)
    rfecv1 = RFECV(estimator=dct, step=1, cv=StratifiedKFold(labels, n_folds=6,shuffle=True,random_state=42),
                  scoring='recall')
    rfecv2 = RFECV(estimator=dct, step=1, cv=StratifiedKFold(labels, n_folds=6,shuffle=True,random_state=42),
              scoring='precision')
    rfecv1.fit(features,labels)     
    rfecv2.fit(features,labels)
    print("Optimal number of features - Recall : %d" % rfecv1.n_features_) 
    print("Optimal number of features - Precision : %d" % rfecv2.n_features_)
 
    BestFeatures = SelectFdr(score_func=f_classif,alpha=alfa)
    BestFeatures.fit_transform(features,labels)
#    BestFeatures = SelectKBest(score_func=f_classif,k=numbest)
#    BestFeatures.fit_transform(features,labels)
    feature_scores = BestFeatures.scores_
    feature_pvalues = BestFeatures.pvalues_
    best_feat_indices = BestFeatures.get_support(indices=True)
    
    best_list = []
    for i in range(len(best_feat_indices)):
        best_list.append(features_list[best_feat_indices[i]+1])
        
    print 'Best features:', best_list
    feat_ctr=-1
    for index in best_feat_indices:
        feat_ctr+=1
        print best_list[feat_ctr],'Score:',feature_scores[index],'P-value:',feature_pvalues[index]
    
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Recall & Precision")
    plt.plot(range(1, len(rfecv1.grid_scores_) + 1), rfecv1.grid_scores_,
             label='Recall',color='blue')
    plt.plot(range(1, len(rfecv2.grid_scores_) + 1), rfecv2.grid_scores_,
             label='Precision',color='green')
    plt.legend()
    plt.show()

feature_select(labels,features)

# At this point we can see that no individual feature has great power over discovery
# of POI in the dataset. However, it's likely that the financial data are highly covariate
# and that even the highest-rated poi-related email variables are covariate. We ought to 
# consider PCA or at least reduced feature selection in order to eliminate covariance.
# First, we'll try to curate a set of features that are representative of categories of features,
# as follows:
# 1. Communication with POI - shared receipt
# 2. Stock - exercised stock
# 3. Regular Pay - bonus - salary isn't as good b/c it might not vary as much for POI

features_list = ['poi','shared_receipt_with_poi','exercised_stock_options','bonus']

# Re-load data with new set of features
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Let's check a couple of plots to see how POI info differs from non-POI and see whether
# outliers will be an issue

def analyze_feature(feature_list,feature_num,labels,features):
    poi_feature_ob =[]
    npoi_feature_ob =[]
    
    for i in range(len(labels)):
        if labels[i] == 0:
            npoi_feature_ob.append(features[i][feature_num])
        if labels[i] == 1:
            poi_feature_ob.append(features[i][feature_num])
    
    poi_vs_npoi = [npoi_feature_ob,poi_feature_ob]
    plot_labels = ['non-POI','POI']
    
    plt.figure()           
    plt.boxplot(poi_vs_npoi,showmeans=True)
    plt.ylabel(feature_list[feature_num])
    plt.xticks([1,2],plot_labels)
    plt.show()

def analyze_features(feature_list,labels,features):
    relevant_features = feature_list[1:]
    
    for feat in range(len(relevant_features)):
        analyze_feature(relevant_features,feat,labels,features)
        
analyze_features(features_list,labels,features)

del my_dataset['TOTAL']

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

analyze_features(features_list,labels,features)

features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 
                 'total_stock_value', 'expenses', 'exercised_stock_options', 
                 'other', 'long_term_incentive', 'restricted_stock', 
                 'director_fees','to_messages', 
                 'from_poi_to_this_person', 'from_messages', 
                 'from_this_person_to_poi', 'shared_receipt_with_poi']
                 
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

feature_select(labels,features,alfa=0.01)

# Create added feature of all POI-related email variables

for entry in my_dataset:
    poi_track = 0
    if my_dataset[entry]['from_poi_to_this_person'] != 'NaN':
        poi_track = my_dataset[entry]['from_poi_to_this_person']
    if my_dataset[entry]['from_this_person_to_poi'] != 'NaN':
        poi_track += my_dataset[entry]['from_this_person_to_poi']
    if my_dataset[entry]['shared_receipt_with_poi'] != 'NaN':
        poi_track += my_dataset[entry]['shared_receipt_with_poi']
    my_dataset[entry]['poi_email_contact'] = poi_track
    
    
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 
                 'total_stock_value', 'expenses', 'exercised_stock_options', 
                 'other', 'long_term_incentive', 'restricted_stock', 
                 'director_fees','poi_email_contact']
                 
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

feature_select(labels,features,alfa=0.01)


features_list = ['poi','salary','bonus','deferred_income','total_stock_value',
                'exercised_stock_options','long_term_incentive']
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
        
analyze_features(features_list,labels,features)
  
PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"   
   
clf = GaussianNB()
clf2 = GradientBoostingClassifier(random_state=42)
clf3 = AdaBoostClassifier(random_state=42)
clf4 = RandomForestClassifier(random_state=42)
clf_list = [('gnb',clf),('grdbst',clf2),('adabst',clf3),('rdmfst',clf4)]

clfvote = VotingClassifier(estimators=clf_list,voting='soft')

cv = StratifiedShuffleSplit(labels, 1000, random_state = 24)
true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0
clf_f1 = []
clf2_f1 = []
clf3_f1 = []
clf4_f1 = []
clfvote_f1 = []

for train_idx, test_idx in cv: 
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )
    

    clf.fit(features_train, labels_train)
    clf2.fit(features_train, labels_train)
    clf3.fit(features_train, labels_train)
    clf4.fit(features_train, labels_train)
    clfvote.fit(features_train, labels_train)
    
    predictions1 = clf.predict(features_test)
    predictions2 = clf2.predict(features_test)
    predictions3 = clf3.predict(features_test)
    predictions4 = clf4.predict(features_test)
    predictions = clfvote.predict(features_test)
 
    clf_f1.append(f1_score(labels_test,predictions1))
    clf2_f1.append(f1_score(labels_test,predictions2))
    clf3_f1.append(f1_score(labels_test,predictions3))
    clf4_f1.append(f1_score(labels_test,predictions4))
    clfvote_f1.append(f1_score(labels_test,predictions))
    
    # Added after GaussianNB() known to be best clf to evaluate
    for prediction, truth in zip(predictions1, labels_test):
        if prediction == 0 and truth == 0:
            true_negatives += 1
        elif prediction == 0 and truth == 1:
            false_negatives += 1
        elif prediction == 1 and truth == 0:
            false_positives += 1
        elif prediction == 1 and truth == 1:
            true_positives += 1
        else:
            print "Warning: Found a predicted label not == 0 or 1."
            print "All predictions should take value 0 or 1."
            print "Evaluating performance for processed predictions:"
            break    
    
print 'Gaussian Naive Bayes Avg F1:', np.mean(clf_f1)
print 'Gradient Boost Avg F1:', np.mean(clf2_f1)
print 'AdaBoost Avg F1:', np.mean(clf3_f1)
print 'Random Forest Avg F1:', np.mean(clf4_f1)
print 'Soft Vote Avg F1:', np.mean(clfvote_f1)

try:
    total_predictions = true_negatives + false_negatives + false_positives + true_positives
    accuracy = 1.0*(true_positives + true_negatives)/total_predictions
    precision = 1.0*true_positives/(true_positives+false_positives)
    recall = 1.0*true_positives/(true_positives+false_negatives)
    f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
    f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
    print clf
    print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
    print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
    print ""
except:
    print "Got a divide by zero when trying out:", clf
    print "Precision or recall may be undefined due to a lack of true positive predicitons."

CLF_PICKLE_FILENAME = "my_classifier.pkl"
DATASET_PICKLE_FILENAME = "my_dataset.pkl"
FEATURE_LIST_FILENAME = "my_feature_list.pkl"

dump_classifier_and_data(clf, my_dataset, features_list)

clf, dataset, feature_list = load_classifier_and_data()

test_classifier(clf,dataset,feature_list)