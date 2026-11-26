# Machine_Learning
A beginner in Data Analytics, passionate about learning, exploring, and solving problems with data.
Machine Learning Exam Project: Admissions Prediction and Zomato User Segmentation

End-to-end solution to a two-part machine learning exam using Python and scikit-learn:

1. Supervised learning: predict whether a student's profile is competitive for a US Master's admission.
2. Unsupervised learning: segment Zomato users by behaviour with PCA and K-Means.

Highlights

Part
Dataset
Result
1. Classification
US Master's Admission Competitiveness (5,200 students, 30 columns)
Best model: Logistic Regression + SMOTE (F1 0.681, recall 0.795, ROC-AUC 0.866)
2. Clustering
Zomato User Behaviour (3,640 rows, 22 columns, 3,587 after de-duplication)
4 user segments (k = 4 from the elbow method), PCA keeps 95% of variance with 13 components

Part 1: US Master's Admission Competitiveness

Goal: classify a profile as Competitive (1) or Non-competitive (0).

• EDA: the target is imbalanced (70% / 30%). UG university tier is the strongest categorical driver (Tier 1 is 48% competitive, Tier 3 is 12%). SOP quality, field alignment, LOR strength and internship relevance separate the classes. GRE and English scores barely differ between classes.
• Cleaning: median imputation for five score columns (4-5% missing), fitted on training data only to avoid leakage. student_id removed.
• Encoding and scaling: ordinal (university tier, ranking band), binary, one-hot for nominal columns; StandardScaler for Logistic Regression.
• Feature selection: Random Forest importance (SOP quality, field alignment and LOR strength lead).
• Models: Logistic Regression, Decision Tree, Random Forest, each with and without SMOTE (applied to the training set only).
Model
Setting
Accuracy
Precision
Recall
F1
ROC-AUC
Logistic Regression
Baseline
0.814
0.712
0.635
0.671
0.867
Decision Tree
Baseline
0.737
0.586
0.417
0.487
0.761
Random Forest
Baseline
0.800
0.774
0.471
0.586
0.847
Logistic Regression
SMOTE
0.777
0.596
0.795
0.681
0.866
Decision Tree
SMOTE
0.664
0.465
0.792
0.586
0.757
Random Forest
SMOTE
0.774
0.607
0.702
0.651
0.843

SMOTE Logistic Regression is chosen because it finds about 80% of truly competitive students while keeping the highest F1.

�

Load image

Part 2: Zomato User Behaviour Segmentation

• Data quality: "None" in subscription_status means no Gold plan (a real category, not missing). 53 duplicate rows removed; remaining blanks filled with the median (numeric) or Unknown (cuisine).
• Outliers: detected with the IQR rule and retained, since they are genuine heavy and lapsed users.
• Correlation: orders, spend, app opens and session time are highly correlated (up to r = 0.94), which is why PCA is used before clustering.
• PCA: 11 components keep 90% of the variance, 13 keep 95% (13 used).
• K-Means: elbow at k = 4, supported by the silhouette score.
Cluster
Share
Orders/month
Spend/month
Description
Power Users
22%
18.1
Rs 7,653
Highest activity, recent orders
Steady Regulars
33%
8.9
Rs 2,821
Average, reliable core
Deal Hunters
26%
3.9
Rs 1,036
Heavy promo use, lots of browsing without ordering
Lapsed / At-risk
19%
1.3
Rs 398
Low activity, about 45 days since last order

�
�

Load image
Load image

Repository Structure

.
├── README.md
├── src/ml_exam_solution.py      # full code for Part 1 and Part 2
├── report/ML_Exam_Solved.docx   # solved exam report with charts and a plain-language summary
├── images/                      # charts used in this README
└── data/                        # place the CSV files here (not included)

How to Run

1. Install dependencies:
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
2. Put the two CSV files in data/:
   ◦ us_masters_admission_competitiveness_-_LMS.csv
   ◦ zomato_user_data_-_LMS.csv
3. In src/ml_exam_solution.py, set the D variable (data folder) and OUT variable (output folder) near the top of each section to your local paths.
4. Run the script. It writes the charts and result tables to the output folder.

Tech Stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE), Matplotlib, Seaborn.

Key Skills Demonstrated

Exploratory data analysis, missing value and duplicate handling, outlier detection, feature encoding and scaling, feature selection, classification, class imbalance (SMOTE), model evaluation, PCA, K-Means clustering and business interpretation of segments.

Author

Saniya BK: LinkedIn | GitHub
