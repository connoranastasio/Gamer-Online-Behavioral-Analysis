# training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================
# Data Loading & Cleaning
# =====================
df = pd.read_csv('online_gaming_behavior_dataset.csv')
df = df.drop_duplicates()
df = df.dropna(subset=['EngagementLevel'])
if 'PlayerID' in df.columns:
    df = df.drop('PlayerID', axis=1)

# =====================
# Feature/Target Split & Encoding
# =====================
X = df.drop('EngagementLevel', axis=1)
y = df['EngagementLevel']
le = LabelEncoder()
y = le.fit_transform(y)

# =====================
# Column Identification
# =====================
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# =====================
# Preprocessing Pipelines
# =====================
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
preprocessor.set_output(transform="pandas")

# =====================
# Model Pipeline
# =====================
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# =====================
# Train/Test Split
# =====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================
# Model Training
# =====================
clf.fit(X_train, y_train)

# =====================
# Prediction & Evaluation
# =====================
y_pred = clf.predict(X_test)
y_test = np.array(y_test).flatten()
y_pred = np.array(y_pred).flatten()
print('Unique predictions:', np.unique(y_pred))
print('Class labels:', le.classes_)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))
print('y_test type/shape:', type(y_test), y_test.shape)
print('y_pred type/shape:', type(y_pred), y_pred.shape)

# =====================
# DataFrame Output (Optional)
# =====================
# Uncomment to view transformed features as DataFrame
# X_train_transformed = clf.named_steps['preprocessor'].transform(X_train)
# print(X_train_transformed.head())

# =====================
# EDA Visualizations
# =====================
# Histograms for numeric features
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
df[numeric_cols].hist(bins=30, figsize=(18, 12), layout=(-1, 3))
plt.suptitle('Histograms of Numeric Features', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# Bar plots for categorical features
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f'Count Plot of {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
corr = df.select_dtypes(include=['int64', 'float64']).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Heatmap of Numeric Features')
plt.tight_layout()
plt.show()

# Boxplots by EngagementLevel
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x='EngagementLevel', y=col)
    plt.title(f'Boxplot of {col} by EngagementLevel')
    plt.tight_layout()
    plt.show()

# Pairplot for selected features
selected_cols = numeric_cols[:3] + ['EngagementLevel'] if len(numeric_cols) >= 3 else numeric_cols + ['EngagementLevel']
sns.pairplot(df[selected_cols], hue='EngagementLevel', diag_kind='kde')
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.show()

# Countplot for EngagementLevel
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='EngagementLevel', order=df['EngagementLevel'].value_counts().index)
plt.title('Count Plot of EngagementLevel')
plt.tight_layout()
plt.show()

# Violin plot for PlayTimeHours by EngagementLevel
if 'PlayTimeHours' in df.columns:
    plt.figure(figsize=(8, 4))
    sns.violinplot(data=df, x='EngagementLevel', y='PlayTimeHours')
    plt.title('Violin Plot of PlayTimeHours by EngagementLevel')
    plt.tight_layout()
    plt.show()


