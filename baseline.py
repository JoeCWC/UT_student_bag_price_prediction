import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import xgboost as xgb




# Reading .csv data file

train_data = pd.read_csv("/home/joe/student_bag_price_prediction/playground-series-s5e2/train.csv")
test_data = pd.read_csv("/home/joe/student_bag_price_prediction/playground-series-s5e2/test.csv")
original_data = pd.read_csv('/home/joe/student_bag_price_prediction/playground-series-s5e2/Noisy_Student_Bag_Price_Prediction_Dataset.csv')

print(train_data.head())
train_data.info()

print(test_data.head())
test_data.info()

print(original_data.head())
original_data.info()

# Checking the number of rows and columns

num_train_rows, num_train_columns = train_data.shape

num_test_rows, num_test_columns = test_data.shape

num_original_rows, num_original_columns = original_data.shape

print("Training Data:")
print(f"Number of Rows: {num_train_rows}")
print(f"Number of Columns: {num_train_columns}\n")

print("Test Data:")
print(f"Number of Rows: {num_test_rows}")
print(f"Number of Columns: {num_test_columns}\n")

print("Original Data:")
print(f"Number of Rows: {num_original_rows}")
print(f"Number of Columns: {num_original_columns}")

# Creating a table for missing values, unique values and data types of the features

missing_values_train = pd.DataFrame({'Feature': train_data.columns,
                              '[TRAIN] No. of Missing Values': train_data.isnull().sum().values,
                              '[TRAIN] % of Missing Values': ((train_data.isnull().sum().values)/len(train_data)*100)})

missing_values_test = pd.DataFrame({'Feature': test_data.columns,
                             '[TEST] No.of Missing Values': test_data.isnull().sum().values,
                             '[TEST] % of Missing Values': ((test_data.isnull().sum().values)/len(test_data)*100)})

missing_values_original = pd.DataFrame({'Feature': original_data.columns,
                             '[ORIGINAL] No.of Missing Values': original_data.isnull().sum().values,
                             '[ORIGINAL] % of Missing Values': ((original_data.isnull().sum().values)/len(original_data)*100)})

unique_values = pd.DataFrame({'Feature': train_data.columns,
                              'No. of Unique Values[FROM TRAIN]': train_data.nunique().values})

feature_types = pd.DataFrame({'Feature': train_data.columns,
                              'DataType': train_data.dtypes})

merged_df = pd.merge(missing_values_train, missing_values_test, on='Feature', how='left')
merged_df = pd.merge(merged_df, missing_values_original, on='Feature', how='left')
merged_df = pd.merge(merged_df, unique_values, on='Feature', how='left')
merged_df = pd.merge(merged_df, feature_types, on='Feature', how='left')

merged_df

# Count duplicate rows in train_data
train_duplicates = train_data.duplicated().sum()

# Count duplicate rows in test_data
test_duplicates = test_data.duplicated().sum()

# Count duplicate rows in original_data
original_duplicates = original_data.duplicated().sum()

# Print the results
print(f"Number of duplicate rows in train_data: {train_duplicates}")
print(f"Number of duplicate rows in test_data: {test_duplicates}")
print(f"Number of duplicate rows in original_data: {original_duplicates}")

# Having a look at the description of all the numerical columns present in the dataset

train_data.describe().T

numerical_variables = ['Weight Capacity (kg)']
target_variable = 'Price' 
categorical_variables = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment','Waterproof', 'Style', 'Color']

# Analysis of all NUMERICAL features

# Define a custom color palette
custom_palette = ['#3498db', '#e74c3c','#2ecc71']

# Add 'Dataset' column to distinguish between train and test data
train_data['Dataset'] = 'Train'
test_data['Dataset'] = 'Test'
original_data['Dataset'] = 'Original'

variables = [col for col in train_data.columns if col in numerical_variables]


# Function to create and display a row of plots for a single variable

def create_variable_plots(variable):
    sns.set_style('whitegrid')

    # 加入 Dataset 標籤並重設索引，避免重複
    train_labeled = train_data.copy()
    train_labeled["Dataset"] = "Train"
    train_labeled = train_labeled.reset_index(drop=True)

    test_labeled = test_data.copy()
    test_labeled["Dataset"] = "Test"
    test_labeled = test_labeled.reset_index(drop=True)

    original_labeled = original_data.dropna().copy()
    original_labeled["Dataset"] = "Original"
    original_labeled = original_labeled.reset_index(drop=True)

    # 合併資料
    combined_data = pd.concat([train_labeled, test_labeled, original_labeled], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Box plot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=combined_data, x=variable, hue="Dataset", palette=custom_palette, dodge=False)
    plt.xlabel(variable)
    plt.title(f"Box Plot for {variable}")
    plt.legend().remove()

    # Separate Histograms
    plt.subplot(1, 2, 2)
    sns.histplot(data=train_labeled, x=variable, color=custom_palette[0], kde=True, bins=30, label="Train")
    sns.histplot(data=test_labeled, x=variable, color=custom_palette[1], kde=True, bins=30, label="Test")
    sns.histplot(data=original_labeled, x=variable, color=custom_palette[2], kde=True, bins=30, label="Original")
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(f"Histogram for {variable} [TRAIN, TEST & ORIGINAL]")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Perform univariate analysis for each variable
for variable in variables:
    create_variable_plots(variable)

# Drop the 'Dataset' column after analysis
train_data.drop('Dataset', axis=1, inplace=True)
test_data.drop('Dataset', axis=1, inplace=True)
original_data.drop('Dataset', axis=1, inplace=True)

pie_chart_palette = ['#33638d', '#28ae80', '#d3eb0c', '#ff9a0b', '#7e03a8', '#35b779', '#fde725', '#440154', '#90d743', '#482173', '#22a884', '#f8961e']

countplot_color = '#5C67A3'

# # Function to create and display a row of plots for a single categorical variable

def create_categorical_plots(variable):
    sns.set_style('whitegrid')

    # 加上 Dataset 標籤並重設索引
    train_labeled = train_data.copy()
    train_labeled["Dataset"] = "Train"
    train_labeled = train_labeled.reset_index(drop=True)

    test_labeled = test_data.copy()
    test_labeled["Dataset"] = "Test"
    test_labeled = test_labeled.reset_index(drop=True)

    original_labeled = original_data.dropna().copy()
    original_labeled["Dataset"] = "Original"
    original_labeled = original_labeled.reset_index(drop=True)

    combined_data = pd.concat([train_labeled, test_labeled, original_labeled], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Pie Chart (只看 train)
    plt.subplot(1, 2, 1)
    train_data[variable].value_counts().plot.pie(
        autopct='%1.1f%%', colors=pie_chart_palette, wedgeprops=dict(width=0.3), startangle=140
    )
    plt.title(f"Pie Chart for {variable}")

    # Bar Graph (合併後)
    plt.subplot(1, 2, 2)
    sns.countplot(
        data=combined_data,
        x=variable,
        hue="Dataset",   # 用顏色區分 Train/Test/Original
        palette=pie_chart_palette,
        alpha=0.8
    )
    plt.xlabel(variable)
    plt.ylabel("Count")
    plt.title(f"Bar Graph for {variable} [TRAIN, TEST & ORIGINAL]")

    plt.tight_layout()
    plt.show()

# Perform univariate analysis for each categorical variable
for variable in categorical_variables:
    create_categorical_plots(variable)


# Analysis of the TARGET feature (Continuous)

# Define a custom color palette
target_palette = ['#3498db', '#e74c3c']

# Add 'Dataset' column to distinguish between Train and Original data
train_data['Dataset'] = 'Train'
original_data['Dataset'] = 'Original'

# Function to create and display a row of plots for the target variable
def create_target_plots(target_variable):
    sns.set_style('whitegrid')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Box Plot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=pd.concat([train_data, original_data.dropna()]), x=target_variable, y="Dataset", palette=target_palette)
    plt.xlabel(target_variable)
    plt.title(f"Box Plot for Target Feature '{target_variable}'")

    # Histogram
    plt.subplot(1, 2, 2)
    sns.histplot(data=train_data, x=target_variable, color=target_palette[0], kde=True, bins=30, label="Train")
    sns.histplot(data=original_data.dropna(), x=target_variable, color=target_palette[1], kde=True, bins=30, label="Original")
    plt.xlabel(target_variable)
    plt.ylabel("Frequency")
    plt.title(f"Histogram for Target Feature '{target_variable}' [TRAIN & ORIGINAL]")
    plt.legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()

# Perform univariate analysis for the target variable
create_target_plots(target_variable)

# Drop the 'Dataset' column after analysis
train_data.drop('Dataset', axis=1, inplace=True)
original_data.drop('Dataset', axis=1, inplace=True)


variables = [col for col in train_data.columns if col in numerical_variables]

cat_variables_train = ['Compartments','Weight Capacity (kg)', 'Price']
cat_variables_test = ['Compartments','Weight Capacity (kg)']

# Adding variables to the existing list
train_variables = variables + cat_variables_train
test_variables = variables + cat_variables_test

# Calculate correlation matrices for train_data and test_data
corr_train = train_data[train_variables].corr()
corr_test = test_data[test_variables].corr()

# Create masks for the upper triangle
mask_train = np.triu(np.ones_like(corr_train, dtype=bool))
mask_test = np.triu(np.ones_like(corr_test, dtype=bool))

# Set the text size and rotation
annot_kws = {"size": 8, "rotation": 45}

# Generate heatmaps for train_data
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
ax_train = sns.heatmap(corr_train, mask=mask_train, cmap='viridis', annot=True,
                      square=True, linewidths=.5, xticklabels=1, yticklabels=1, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Train Data')

# Generate heatmaps for test_data
plt.subplot(1, 2, 2)
ax_test = sns.heatmap(corr_test, mask=mask_test, cmap='viridis', annot=True,
                     square=True, linewidths=.5, xticklabels=1, yticklabels=1, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Test Data')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()



# 🛠️ Data Preprocessing
# Drop null values from original_data
original_data = original_data.dropna()

# Print the count of null values in original_data
print(original_data.isnull().sum())

# Combine original_data with train_data
train_data = pd.concat([train_data, original_data], axis=0).reset_index(drop=True)

# Define imputation strategies
categorical_features = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
numerical_features = ["Weight Capacity (kg)"]


# Fill categorical missing values with mode (most frequent value)
# Categorical features
for col in categorical_features:
    train_data[col] = train_data[col].fillna(train_data[col].mode()[0])
    test_data[col] = test_data[col].fillna(test_data[col].mode()[0])

# Fill numerical missing values with median
# Numerical features
for col in numerical_features:
    train_data[col] = train_data[col].fillna(train_data[col].median())
    test_data[col] = test_data[col].fillna(test_data[col].median())


def perform_feature_engineering(df):
    # Brand Material Interaction - Certain materials may be common for specific brands
    df['Brand_Material'] = df['Brand'] + '_' + df['Material']

    # Brand & Size Interaction - Some brands may produce only specific sizes
    df['Brand_Size'] = df['Brand'] + '_' + df['Size']

    # Has Laptop Compartment - Convert Yes/No to 1/0 for easier analysis
    df['Has_Laptop_Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})

    # Is Waterproof - Convert Yes/No to 1/0 for easier analysis
    df['Is_Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})

    # Compartments Binning - Group compartments into categories
    df['Compartments_Category'] = pd.cut(df['Compartments'], bins=[0, 2, 5, 10, np.inf], labels=['Few', 'Moderate', 'Many', 'Very Many'])

    # Weight Capacity Ratio - Normalize weight capacity using the max value
    df['Weight_Capacity_Ratio'] = df['Weight Capacity (kg)'] / df['Weight Capacity (kg)'].max()

    # Interaction Feature: Weight vs. Compartments - Some bags may hold more with less compartments
    df['Weight_to_Compartments'] = df['Weight Capacity (kg)'] / (df['Compartments'] + 1)  # Avoid division by zero

    # Style and Size Interaction - Certain styles may correlate with sizes
    df['Style_Size'] = df['Style'] + '_' + df['Size']

    return df

# Apply the function to the training data
train_data = perform_feature_engineering(train_data)

# Apply the function to the test data
test_data = perform_feature_engineering(test_data)

id_test = test_data['id']

columns_to_drop = ['id']
train_data.drop(columns_to_drop, axis=1, inplace=True)
test_data.drop(columns_to_drop, axis=1, inplace=True)


columns_to_check = ['Weight Capacity (kg)','Weight_Capacity_Ratio','Weight_to_Compartments']

# Function to remove outliers using IQR and visualize
def remove_outliers_iqr_with_plot(data, column):
    Q1 = data[column].quantile(0.15)
    Q3 = data[column].quantile(0.85)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the data
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    # Calculate the number of rows deleted
    rows_deleted = len(data) - len(filtered_data)
    
    # Plot the distribution with outliers
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=data[column], color='lightblue', flierprops={'marker': 'o', 'markersize': 5, 'markerfacecolor': 'red'})
    
    # Highlight Q1 and Q3
    plt.axvline(Q1, color='green', linestyle='--', label='Q1 (10th Percentile)')
    plt.axvline(Q3, color='blue', linestyle='--', label='Q3 (90th Percentile)')
    
    # Highlight lower and upper bounds
    plt.axvline(lower_bound, color='red', linestyle='-', label='Lower Bound')
    plt.axvline(upper_bound, color='red', linestyle='-', label='Upper Bound')

    plt.title(f'Outlier Detection for {column}')
    plt.legend()
    plt.xlabel(column)
    plt.show()
    
    return filtered_data, rows_deleted

# Apply function to each numerical column and visualize
rows_deleted_total = 0

for column in columns_to_check:
    train_data, rows_deleted = remove_outliers_iqr_with_plot(train_data, column)
    rows_deleted_total += rows_deleted
    print(f"Rows deleted for {column}: {rows_deleted}")

print(f"Total rows deleted: {rows_deleted_total}")



y = train_data['Price']

# [FOR TRAIN]
# Identify features with skewness greater than 0.75
skewed_features = train_data[numerical_variables].skew()[train_data[numerical_variables].skew() > 0.75].index.values

# Print the list of variables to be transformed
print("Features to be transformed (skewness > 0.75):")
print(skewed_features)

# Plot skewed features before transformation
for feature in skewed_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(train_data[feature], bins=50, kde=True, color='blue')
    plt.title(f'Distribution of {feature} before log transformation')
    plt.show()

# Apply log1p transformation to skewed features
train_data[skewed_features] = np.log1p(train_data[skewed_features])

# Plot skewed features after transformation
for feature in skewed_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(train_data[feature], bins=50, kde=True, color='green')
    plt.title(f'Distribution of {feature} after log transformation')
    plt.show()


# [FOR TEST]
# Identify features with skewness greater than 0.75
skewed_features = test_data[numerical_variables].skew()[test_data[numerical_variables].skew() > 0.75].index.values

# Print the list of variables to be transformed
print("Features to be transformed (skewness > 0.75):")
print(skewed_features)

# Plot skewed features before transformation
for feature in skewed_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(test_data[feature], bins=50, kde=True, color='blue')
    plt.title(f'Distribution of {feature} before log transformation')
    plt.show()

# Apply log1p transformation to skewed features
test_data[skewed_features] = np.log1p(test_data[skewed_features])

# Plot skewed features after transformation
for feature in skewed_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(test_data[feature], bins=50, kde=True, color='green')
    plt.title(f'Distribution of {feature} after log transformation')
    plt.show()

# Selecting specific columns for encoding
columns_to_encode = ['Brand', 'Material', 'Size', 'Laptop Compartment','Waterproof', 'Style', 'Color','Brand_Material', 'Brand_Size', 'Has_Laptop_Compartment','Is_Waterproof', 'Compartments_Category', 'Style_Size']
train_data_to_encode = train_data[columns_to_encode]
test_data_to_encode = test_data[columns_to_encode]

# Dropping selected columns for scaling
train_data_to_scale = train_data.drop(columns_to_encode, axis=1)
test_data_to_scale = test_data.drop(columns_to_encode, axis=1)

train_data_encoded = pd.get_dummies(train_data_to_encode, columns=columns_to_encode, drop_first=True)
test_data_encoded = pd.get_dummies(test_data_to_encode, columns=columns_to_encode, drop_first=True)
print(train_data_encoded.head())
print(test_data_encoded.head())


# Feature Scaling
# Initialize MinMaxScaler
minmax_scaler = MinMaxScaler()

# Fit the scaler on the training data
minmax_scaler.fit(train_data_to_scale.drop(['Price'], axis=1))

# Scale the training data
scaled_data_train = minmax_scaler.transform(train_data_to_scale.drop(['Price'], axis=1))
scaled_train_df = pd.DataFrame(scaled_data_train, columns=train_data_to_scale.drop(['Price'], axis=1).columns)

# Scale the test data using the parameters from the training data
scaled_data_test = minmax_scaler.transform(test_data_to_scale)
scaled_test_df = pd.DataFrame(scaled_data_test, columns=test_data_to_scale.columns)
print(scaled_train_df.head())
print(scaled_test_df.head())

# Concatenate train datasets
train_data_combined = pd.concat([train_data_encoded.reset_index(drop=True), scaled_train_df.reset_index(drop=True)], axis=1)

# Concatenate test datasets
test_data_combined = pd.concat([test_data_encoded.reset_index(drop=True), scaled_test_df.reset_index(drop=True)], axis=1)
print(train_data_combined.head())
print(test_data_combined.head())



# 🏗️ Model Building & Evaluation

# Cross-Validation strategy
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# XGBoost parameters (roughly equivalent to your CatBoost params)
xgb_params = {
    "n_estimators": 300,
    "learning_rate": 0.1,
    "max_depth": 6,
    "early_stopping_rounds": 50,
    "random_state": 42,
    "verbosity": 0,
    "tree_method": "gpu_hist"   # 建議加速選項，可依 GPU/CPU 調整
}

# Lists to store results
rmse_scores = []
mae_scores = []
oof_preds = np.zeros(len(train_data_combined))
test_preds_xgb = np.zeros(len(test_data_combined))

# Store feature importances
feature_importance_list = np.zeros(train_data_combined.shape[1])

print("Training using Cross-Validation...")
for fold, (train_idx, val_idx) in enumerate(kf.split(train_data_combined)):
    print(f"\nTraining Fold {fold+1}...")

    X_train, X_val = train_data_combined.iloc[train_idx], train_data_combined.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Define model
    xgb_model = xgb.XGBRegressor(**xgb_params)

    # Train model
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Predict on validation set
    val_preds = xgb_model.predict(X_val)
    oof_preds[val_idx] = val_preds

    # Calculate and store scores
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    mae = mean_absolute_error(y_val, val_preds)
    rmse_scores.append(rmse)
    mae_scores.append(mae)

    print(f"Fold {fold+1} RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Accumulate feature importances
    feature_importance_list += xgb_model.feature_importances_ / kf.get_n_splits()

    # Predict on test data and average across folds
    test_preds_xgb += xgb_model.predict(test_data_combined) / kf.get_n_splits()

# Final evaluation

cv_rmse = np.mean(rmse_scores)
cv_mae = np.mean(mae_scores)

print("\nCross-Validation Results:")
print(f"Mean RMSE: {cv_rmse:.4f}")
print(f"Mean MAE: {cv_mae:.4f}")

# Plot RMSE per fold if needed
if len(rmse_scores) > 1:
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(rmse_scores) + 1), rmse_scores, marker='o', linestyle='--', color='b', label='RMSE per Fold')
    plt.axhline(y=cv_rmse, color='r', linestyle='-', label=f'Avg RMSE: {cv_rmse:.4f}')
    plt.xlabel('Fold')
    plt.ylabel('RMSE')
    plt.title('RMSE per Fold')
    plt.legend()
    plt.show()

cv_rmse = np.mean(rmse_scores)
cv_mae = np.mean(mae_scores)

print("\nCross-Validation Results:")
print(f"Mean RMSE: {cv_rmse:.4f}")
print(f"Mean MAE: {cv_mae:.4f}")

# Feature importance visualization
feature_importance_df = pd.DataFrame({
    'Feature': train_data_combined.columns,
    'Importance': feature_importance_list
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 6))
plt.barh(feature_importance_df['Feature'][:20], feature_importance_df['Importance'][:20], color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Top 20 Feature Importances - XGBoost')
plt.gca().invert_yaxis()
plt.show()


# Determine threshold for feature selection
median_importance = np.median(feature_importance_list)
threshold = max(median_importance, 0.05 * np.max(feature_importance_list))  # Keep features > 5% of max importance

selected_features = feature_importance_df[feature_importance_df['Importance'] >= threshold]['Feature'].tolist()

print(f"Selected {len(selected_features)} features out of {train_data_combined.shape[1]} using threshold: {threshold:.4f}")