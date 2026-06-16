Project Overview

This project was built for DataSprint 2026, a one-week data science hackathon organised by the Strathmore Data Community (SDC) in partnership with iLab Africa at Strathmore University, Nairobi.

The challenge was to analyse data from the 2024 FinAccess Household Survey, the most comprehensive survey of financial inclusion in Kenya, published in December 2024 by the Central Bank of Kenya, KNBS, and FSD Kenya — and build a machine learning model that predicts whether a Kenyan adult's financial situation has Improved, Stayed the same, or Worsened compared to the previous year.

Beyond just building a model, the project answers the guiding question:


Which factors most strongly predict financial deterioration among Kenyan adults, and what should policymakers, banks, or NGOs prioritise to improve financial wellbeing?




The Problem

Kenya has made significant strides in financial inclusion over the past decade. Mobile money has transformed how millions of Kenyans save, borrow, and transact. Yet the picture on the ground remains deeply uneven.

According to the 2024 FinAccess Household Survey:


9.9% of Kenyan adults remain fully excluded from financial services
52.6% of those surveyed reported that their financial situation worsened compared to the previous year
Only 20.5% reported improvement


Behind every data point is a real person — a smallholder farmer who could not repay a loan after a drought, a young woman in Turkana with no national ID and therefore no bank account, a household in Nairobi spending more than it earns.

Understanding who is struggling and why is one of the most important questions facing Kenya's policymakers, banks, NGOs, and development partners today. This project uses machine learning to help answer it.

Problem Type: Multiclass Classification (3 classes)

ClassDescriptionProportionWorsenedFinancial situation deteriorated vs previous year52.6%Stayed the sameNo significant change26.9%ImprovedFinancial situation improved vs previous year20.5%


Dataset

Source: 2024 FinAccess Household Survey, curated by iLab Africa (Kevin Obote)

File: finaccess2024_datasprint.csv

Size: 20,871 rows × 28 columns — covering adults across all 47 counties of Kenya

Feature Categories

CategoryFeaturesDemographicscounty, location_type, Sex, Age, household_size, education_level, marital_status, has_disabilityLivelihoodmonthly_income, experienced_shockMobile & Digitalmobile_ownership_1, mobile_money_access, barriers_mobile_moneyFinancial BehaviourSavings_formal, Savings_informal, Loan_formal, Loan_informal, defaulted, formal_service_use, barriers_bank, prodsum1Financial Healthfl_score, nfhi_11, nfhi_12, nfhi_13, accessto_13k_1month, not_difficultTargetfinancial_status

Data Quality Issues Found & Fixed

ColumnIssueFix Appliedfinancial_statusLeading whitespace on ' Improved'str.strip()monthly_incomeMixed text and numeric values (e.g. 'household', 'Construction worker')Converted to numeric; non-numeric values replaced with median (KES 5,000)barriers_bank27.5% missing — these are respondents who already have a bank accountFilled with 0 meaning "No barrier"


Tools & Technologies

ToolPurposePython 3Core programming languagepandasData loading, cleaning, manipulationnumpyNumerical operationsmatplotlibData visualisationseabornHeatmaps and statistical chartsscikit-learnMachine learning models and evaluationGoogle ColabNotebook execution environmentMicrosoft PowerPointFinal presentation slidespython-pptx / pptxgenjsProgrammatic slide generation

scikit-learn Components Used


train_test_split — splitting data into training and test sets
LabelEncoder — encoding categorical variables
DecisionTreeClassifier — baseline interpretable model
RandomForestClassifier — primary high-performance model
f1_score, classification_report, confusion_matrix — evaluation



Methodology

This project followed the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework, the industry standard used by 43% of data science practitioners.

Business Understanding → Data Understanding → Data Preparation → Modelling → Evaluation

Step 1 - Business Understanding

Read the problem statement carefully before touching any data. Understood that:

This is a multiclass classification problem (not binary, not regression)
The primary metric is Weighted F1-Score — NOT accuracy
Class imbalance is a critical challenge (Worsened = 52.6% majority)
The output must be explainable to non-technical policymakers


Step 2 - Data Understanding (EDA)

Loaded and inspected all 28 columns
Identified data types, missing values, and outliers
Visualised distributions and relationships before any modelling
Produced 5+ labelled charts to tell the data story


Step 3 - Data Preparation


Stripped whitespace from financial_status
Converted monthly_income to fully numeric
Filled barriers_bank missing values with 0 (informed fill — not arbitrary)
Applied LabelEncoder to all remaining categorical columns (county, Sex)
Split data: 80% training / 20% test with stratify=y to preserve class proportions


Step 4 - Modelling

Built a two-model strategy:

Model 1 - Decision Tree (Baseline)


Purpose: Explainability - every prediction can be traced step by step
max_depth=8 prevents overfitting
class_weight='balanced' compensates for the Worsened majority


Model 2 - Random Forest (Primary)


Purpose: Best performance
200 decision trees vote together — reduces variance and overfitting
max_depth=12, class_weight='balanced'
Built-in feature importance scores


Step 5 - Evaluation


Evaluated on the held-out test set (4,175 respondents)
Primary metric: Weighted F1-Score
Also reported: per-class precision, recall, F1; confusion matrix



Exploratory Data Analysis

Five key charts were produced during EDA:

Chart 1 - Target Distribution

Over half of respondents (52.6% = 10,981 people) reported their finances worsened. Only 20.5% improved. This class imbalance was central to every modelling decision.

Chart 2 - Financial Shock vs Outcome (Most Important Finding)

Adults who experienced a financial shock showed dramatically higher rates of worsened outcomes (59.6% Worsened) compared to those who did not (47.3% Worsened). Shocks - droughts, illness, job loss  are a key entry point into financial decline.

Chart 3 - Monthly Income Distribution

The income distributions for the three classes are clearly separated:


Improved respondents cluster at higher income levels
Worsened respondents are concentrated at very low incomes (under KES 5,000)
Median income for Improved is approximately 2× higher than Worsened


Chart 4 - Education Level vs Financial Status

Higher education levels are associated with better financial outcomes. University-educated adults show notably higher rates of improvement — education provides financial resilience and access to better-paying jobs.

Chart 5 - Financial Health Indicators

Three indicators were examined: food security (nfhi_11), managing non-food spending (nfhi_12), and being debt-stress free (nfhi_13). In all three cases, respondents who answered "Yes" showed substantially lower rates of financial deterioration.


Modelling

Why These Models?

Decision Tree was chosen as the baseline because:


Fully interpretable - every prediction can be explained
Works well with mixed data types
Fast to train
Judges can follow the logic


Random Forest was chosen as the primary model because:


Significantly outperforms a single tree
Handles class imbalance well with class_weight='balanced'
Provides reliable feature importance scores
Robust to overfitting through ensemble averaging


Why NOT just use accuracy?

With 52.6% of data labelled "Worsened", a model that predicts "Worsened" every single time would achieve 52.6% accuracy while being completely useless. Weighted F1-Score penalises this behaviour by measuring performance on each class separately, then averaging weighted by class size.

Handling Class Imbalance


Used class_weight='balanced' in both models - automatically adjusts weights so minority classes (Improved = 20.5%) are not ignored
Used stratify=y in train/test split to ensure all three classes are proportionally represented in both sets



Results & Performance

Model Comparison

ModelWeighted F1-ScoreDecision Tree (baseline)0.479Random Forest (primary)0.542Improvement+0.063

Random Forest - Full Classification Report

ClassPrecisionRecallF1-ScoreSupportImproved0.400.460.43856Stayed the same0.430.300.361,122Worsened0.650.720.682,197Weighted Average0.540.550.544,175

Confusion Matrix Insights (Random Forest)


Worsened is predicted best - 71.6% correctly identified (most policy-critical class)
Stayed the same is the hardest to predict — often confused with both other classes
Improved recall of 46% - the model catches nearly half of all people whose situation improved


Test Set Size


Total test respondents: 4,175
Training respondents: 16,696



Key Findings

Top 10 Feature Importances (Random Forest)

RankFeatureImportanceWhat It Measures1monthly_income11.8%Total income received in past month (KES)2county11.4%Which of 47 counties the respondent lives in3household_size7.6%Number of people in the household4prodsum17.0%Number of financial services/products used5Age6.8%Age group of respondent6education_level6.5%Highest education level completed7barriers_bank4.7%Main barrier to accessing a bank account8fl_score3.9%Financial literacy score9nfhi_113.8%Whether household was food secure in past year10marital_status3.6%Marital status of respondent

Summary of Key Insights

1. Income is the root driver
Monthly income (11.8% importance) is the single strongest predictor of financial status. Low-income earners are disproportionately represented in the "Worsened" class and rarely transition to "Improved" without structural support.

2. Geography determines opportunity
County (11.4%) is the second strongest predictor. Where a Kenyan adult lives has an enormous impact on their financial trajectory - reflecting differences in economic activity, infrastructure, market access, and public services between counties.

3. Large households are under pressure
Household size (7.6%) shows that more dependents per earner directly increases financial stress. This is especially critical in low-income households where a single earner supports many.

4. Financial product access is protective
prodsum1 (7.0%) — the number of financial products a person uses — is strongly associated with better outcomes. People using savings accounts, mobile money, loans, and insurance are more financially resilient.

5. Education buffers financial shocks
Education level (6.5%) correlates with improved outcomes, likely through access to better-paying, more stable employment and greater financial literacy.

6. Financial shocks are a critical trigger
While experienced_shock ranked 14th in feature importance, the EDA showed it dramatically shifts the probability of financial deterioration — from 47% Worsened (no shock) to 60% Worsened (shock experienced).

7. Literacy matters
fl_score (3.9%) — financial literacy — appears in the top 10. Adults who understand financial concepts make better decisions and are more likely to use protective financial tools.


Recommendations

Based on the model findings, the following actions are recommended for each stakeholder group:

For National Policymakers

Priority: Income support interventions
Monthly income is the #1 predictor of financial outcomes. Programmes that directly raise incomes — cash transfers, minimum wage enforcement, agricultural price support, and rural employment schemes — will have the highest population-wide impact on financial wellbeing.

For County Governments

Priority: Geographic resource allocation
County is the 2nd strongest predictor. This means some counties are systematically producing worse financial outcomes for residents. County-level financial deterioration maps (which this model can generate) should guide resource allocation. Counties with the highest proportion of "Worsened" respondents need priority investment in economic infrastructure.

For Banks and Microfinance Institutions (MFIs)

Priority: Expand product access to low-income earners
The prodsum1 finding shows that financial product use is protective. Banks and MFIs should:


Design affordable savings products for low-income earners
Offer microinsurance products to buffer financial shocks
Simplify account opening to reduce barriers_bank
Target mobile-first products since mobile_money_access is widespread


For NGOs and Development Partners

Priority: Shock resilience programmes
Financial shocks (drought, illness, job loss) dramatically increase the risk of financial deterioration. Development partners should:


Build emergency savings and credit facilities accessible within 30 days
Expand agricultural insurance in drought-prone counties
Invest in financial literacy programmes — fl_score appears in the top 10 drivers
Focus on food security interventions (nfhi_11 is also a key predictor)


For Researchers and Future Work


A county-level analysis would reveal which specific counties are most at risk
Time-series modelling would help track whether interventions actually improve outcomes year over year
SHAP values would give individual-level explanations (why did this specific person's finances worsen?)
Adding qualitative data (reasons for shocks, types of employment) could improve model performance



File Structure

DataSprint2026/
│
├── finaccess2024_datasprint.csv          # Cleaned dataset (20,871 rows × 28 cols)
├── DataSprint2026_FinancialStatus_Prediction.ipynb  # Full Jupyter notebook
├── DataSprint2026_Presentation.pptx      # 7-slide PowerPoint presentation
└── README.md                             # This file


How to Run

Option 1 — Google Colab (Recommended)


Go to colab.research.google.com
Click File → Upload notebook → upload DataSprint2026_FinancialStatus_Prediction.ipynb
In the left sidebar, click the Files icon (📁) → upload finaccess2024_datasprint.csv
Click Runtime → Run all
All cells will execute in order — total runtime approximately 3–5 minutes


Option 2 - VS Code


Install extensions: Python (Microsoft) + Jupyter (Microsoft)
Place both files in the same folder
Open the .ipynb file — it will render as a notebook
Install dependencies:


bash   pip install pandas numpy matplotlib seaborn scikit-learn


Click Run All at the top


Option 3 — Terminal / Command Line

bashpip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook DataSprint2026_FinancialStatus_Prediction.ipynb

Expected Output

Running all cells produces:


7 labelled charts saved inline in the notebook
Decision Tree Weighted F1 ≈ 0.479
Random Forest Weighted F1 ≈ 0.542
Full classification report per class
Side-by-side confusion matrices
Feature importance bar chart
Final results summary printed to console
