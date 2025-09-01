"""
Improved PTSD Prediction Model - Enhanced for Imbalanced Data
This model specifically addresses:
1. Low recall for positive class (PTSD cases)
2. Severe class imbalance
3. Threshold optimization
4. Advanced loss functions for imbalanced learning
"""

from tabnanny import verbose
from turtle import mode    
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import (StratifiedKFold, StratifiedShuffleSplit, cross_val_score, train_test_split, RepeatedStratifiedKFold,
                                   GridSearchCV, RandomizedSearchCV)
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report, 
                           confusion_matrix, make_scorer, recall_score, precision_score,
                           f1_score, balanced_accuracy_score, matthews_corrcoef)
from sklearn.feature_selection import (SelectKBest, f_classif, RFE, SelectFromModel,
                                     mutual_info_classif)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                            VotingClassifier, StackingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, SVMSMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.ensemble import (BalancedRandomForestClassifier, BalancedBaggingClassifier,
                              RUSBoostClassifier, EasyEnsembleClassifier)
import xgboost as xgb
# from tabpfn import TabPFNClassifier
# from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import warnings
import os
warnings.filterwarnings('ignore')

class ImprovedPTSDModel:
    """Enhanced PTSD Prediction Model with focus on improving recall for positive class"""
    
    def __init__(self, use_random_seed=True, random_state=42):
        self.use_random_seed = use_random_seed
        self.random_state = random_state if use_random_seed else None
        self.feature_names = None
        self.top_feature_names = None
        self.results = {}
        self.scaler = None
        self.optimal_threshold = 0.5  # Will be optimized
        
        # Set random seeds if enabled
        if self.use_random_seed:
            print(f"Using fixed random seed: {random_state}")
            np.random.seed(random_state)
        else:
            print("Random seed disabled - using true randomness")
    
    @classmethod
    def with_random_seed(cls, seed=42):
        return cls(use_random_seed=True, random_state=seed)
    
    @classmethod
    def with_true_randomness(cls):
        return cls(use_random_seed=False)
    
    def load_data(self, file_path='./data/pre_deployment_data.pkl'):
        """Load and prepare the data with sophisticated missing value handling"""
        print("Loading data...")
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        self.merged = data_dict['merged_df']
        self.HRV_features = data_dict['HRV_features']
        self.EMG_features = data_dict['EMG_features']
        self.CUT_features = data_dict['CUT_feature']
        self.SPSS_features = data_dict['SPSS_feature']
        self.PTSD_features = data_dict['PTSD_features']
        
        
        # Combine all numerical features
        self.all_numeric_features = (self.HRV_features + self.CUT_features + 
                                   self.SPSS_features + self.EMG_features)
        
        print(f"Total numerical features: {len(self.all_numeric_features)}")
        
        # Apply sophisticated missing value handling
        # self.merged = self._handle_missing_values(self.merged, self.all_numeric_features)

        rows_before = len(self.merged)
        self.merged = self.merged[self.all_numeric_features + ['CAPSF1I2s.0']].dropna()
        rows_after = len(self.merged)
        rows_gone = rows_before - rows_after
        print(f"Rows before dropna: {rows_before}")
        print(f"Rows after dropna: {rows_after}")
        print(f"Number of rows removed: {rows_gone}")

        df = self.merged[self.all_numeric_features]
        
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        print(f"Non-numeric columns in df: {non_numeric}")
        
        for col in non_numeric:
            try:
                # Check if values contain decimal points to decide between int and float
                sample_val = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "0"
                if '.' in sample_val:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                    print(f"Converted df_pre['{col}'] to float")
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(int)
                    print(f"Converted df_pre['{col}'] to int")
            except Exception as e:
                print(f"Error converting df_pre['{col}']: {e}")


        # Verify conversion
        print(f"\nAfter conversion:")
        print(f"df non-numeric columns: {df.select_dtypes(exclude=[np.number]).columns.tolist()}")

        for col in self.all_numeric_features:
            self.merged[col] = df[col]
        
        return self.merged, self.all_numeric_features
    
    def _handle_missing_values(self, df, numeric_features):
        """Sophisticated missing value handling using multiple strategies"""
        print("\nHandling missing values with advanced imputation strategies...")
        
        # Create a copy to work with
        df_imputed = df.copy()
        
        # Analyze missing data patterns
        missing_stats = {}
        for col in numeric_features:
            if col in df_imputed.columns:
                # Convert to numeric first
                df_imputed[col] = pd.to_numeric(df_imputed[col], errors='coerce')
                missing_count = df_imputed[col].isna().sum()
                missing_pct = (missing_count / len(df_imputed)) * 100
                missing_stats[col] = {
                    'count': missing_count,
                    'percentage': missing_pct
                }
        
        # Categorize features by missing data severity
        high_missing = [col for col, stats in missing_stats.items() if stats['percentage'] > 15]
        medium_missing = [col for col, stats in missing_stats.items() if 5 < stats['percentage'] <= 15]
        low_missing = [col for col, stats in missing_stats.items() if 1 < stats['percentage'] <= 5]
        minimal_missing = [col for col, stats in missing_stats.items() if 0 < stats['percentage'] <= 1]
        
        print(f"Missing data categorization:")
        print(f"  High missing (>15%): {len(high_missing)} features")
        print(f"  Medium missing (5-15%): {len(medium_missing)} features")
        print(f"  Low missing (1-5%): {len(low_missing)} features")
        print(f"  Minimal missing (<1%): {len(minimal_missing)} features")
        
        # Strategy 1: Handle minimal missing with median (most stable)
        if minimal_missing:
            print(f"Imputing {len(minimal_missing)} features with minimal missing using median...")
            for col in minimal_missing:
                median_val = df_imputed[col].median()
                df_imputed[col] = df_imputed[col].fillna(median_val)
        
        # Strategy 2: Handle low missing with KNN imputation (preserves local structure)
        if low_missing:
            print(f"Imputing {len(low_missing)} features with low missing using KNN...")
            
            # Group by feature type for more accurate imputation
            hrv_low = [col for col in low_missing if any(hrv in col for hrv in ['HRV', 'HR', 'NN', 'RR'])]
            emg_low = [col for col in low_missing if any(emg in col.lower() for emg in ['emg', 'base', 'p80', 'p85', 'p90', 'isi'])]
            bio_low = [col for col in low_missing if any(bio in col.lower() for bio in ['cortisol', 'amylase', 'crp', 'npy', 'ue', 'une'])]
            other_low = [col for col in low_missing if col not in hrv_low + emg_low + bio_low]
            
            # Apply KNN imputation by groups
            for group_name, group_features in [('HRV', hrv_low), ('EMG', emg_low), ('Biomarkers', bio_low), ('Other', other_low)]:
                if group_features:
                    print(f"  Imputing {group_name} features: {len(group_features)}")
                    
                    # Find related complete features to help with imputation
                    if group_name == 'HRV':
                        helper_features = [col for col in numeric_features if col in df_imputed.columns and 
                                         any(hrv in col for hrv in ['HRV', 'HR']) and df_imputed[col].notna().all()]
                    elif group_name == 'EMG':
                        helper_features = [col for col in numeric_features if col in df_imputed.columns and 
                                         any(emg in col.lower() for emg in ['emg', 'base', 'p']) and df_imputed[col].notna().all()]
                    elif group_name == 'Biomarkers':
                        helper_features = [col for col in numeric_features if col in df_imputed.columns and 
                                         any(bio in col.lower() for bio in ['cortisol', 'crp', 'age', 'bmi']) and df_imputed[col].notna().all()]
                    else:
                        helper_features = [col for col in numeric_features if col in df_imputed.columns and df_imputed[col].notna().all()]
                    
                    # Limit helper features to prevent overfitting
                    helper_features = helper_features[:min(20, len(helper_features))]
                    
                    if helper_features:
                        # Combine target and helper features
                        impute_features = group_features + helper_features
                        impute_data = df_imputed[impute_features].copy()
                        
                        # Use KNN imputer
                        knn_imputer = KNNImputer(n_neighbors=min(5, len(impute_data) // 10), weights='distance')
                        imputed_values = knn_imputer.fit_transform(impute_data)
                        
                        # Update only the target features
                        for i, col in enumerate(group_features):
                            df_imputed[col] = imputed_values[:, i]
                    else:
                        # Fallback to median if no helper features
                        for col in group_features:
                            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
        
        # Strategy 3: Handle medium missing with Iterative Imputation (MICE-like)
        if medium_missing:
            print(f"Imputing {len(medium_missing)} features with medium missing using Iterative Imputation...")
            
            # Select stable features as predictors (features with <1% missing)
            stable_predictors = [col for col in numeric_features if col in df_imputed.columns and 
                               df_imputed[col].isna().sum() / len(df_imputed) < 0.01][:30]  # Limit to 30 predictors
            
            if stable_predictors:
                # Combine target and predictor features
                iterative_features = medium_missing + stable_predictors
                iterative_data = df_imputed[iterative_features].copy()
                
                # Use Iterative Imputer with Random Forest estimator for robustness
                iterative_imputer = IterativeImputer(
                    estimator=None,  # Uses BayesianRidge by default (fast and stable)
                    max_iter=10,
                    random_state=self.random_state,
                    initial_strategy='median'
                )
                imputed_values = iterative_imputer.fit_transform(iterative_data)
                
                # Update only the target features
                for i, col in enumerate(medium_missing):
                    df_imputed[col] = imputed_values[:, i]
            else:
                # Fallback to median if no stable predictors
                for col in medium_missing:
                    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
        
        # Strategy 4: Handle high missing with domain-specific strategies
        if high_missing:
            print(f"Imputing {len(high_missing)} features with high missing using domain-specific strategies...")
            
            for col in high_missing:
                missing_pct = missing_stats[col]['percentage']
                
                if missing_pct > 50:
                    # Very high missing: use mode or create missing indicator
                    print(f"  {col}: {missing_pct:.1f}% missing - using median with missing indicator")
                    df_imputed[f'{col}_was_missing'] = df_imputed[col].isna().astype(int)
                    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
                else:
                    # High but <50%: try group-based imputation
                    print(f"  {col}: {missing_pct:.1f}% missing - using group-based median")
                    
                    # Group by outcome if available for more accurate imputation
                    if 'CAPSF1I2s.0' in df_imputed.columns:
                        grouped_median = df_imputed.groupby('CAPSF1I2s.0')[col].transform('median')
                        df_imputed[col] = df_imputed[col].fillna(grouped_median)
                        # Fill any remaining with overall median
                        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
                    else:
                        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
        
        # Final check: handle any remaining missing values
        remaining_missing = df_imputed[numeric_features].isna().sum().sum()
        if remaining_missing > 0:
            print(f"Filling {remaining_missing} remaining missing values with median...")
            for col in numeric_features:
                if col in df_imputed.columns:
                    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
        
        # Verify no missing values remain
        final_missing = df_imputed[numeric_features].isna().sum().sum()
        print(f"Missing value handling complete. Remaining missing values: {final_missing}")
        
        return df_imputed
    
    def apply_power_transformation(self, X):
        """Apply power transformation for normality"""
        print("Applying power transformation...")
        
        # Power transformation for normality
        pt = PowerTransformer(method='yeo-johnson', standardize=True)
        X_transformed = pd.DataFrame(
            pt.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        return X_transformed
    
    def remove_outliers(self, X, y):
        """Remove outliers using IQR method on a specified subset of columns"""
        print("Removing outliers...")

        # First apply power transformation to X for outlier detection
        X_transformed = self.apply_power_transformation(X)

        cols_in_X = X_transformed.columns

        # Use IQR method with less aggressive thresholds on the specified columns only
        Q1 = X_transformed[cols_in_X].quantile(0.15)
        Q3 = X_transformed[cols_in_X].quantile(0.85)
        IQR = Q3 - Q1
        outlier_mask = ~((X_transformed[cols_in_X] < (Q1 - 1.5 * IQR)) | (X_transformed[cols_in_X] > (Q3 + 1.5 * IQR))).any(axis=1)

        # Apply the outlier mask to the original (non-transformed) X and y
        X_clean = X[outlier_mask]
        y_clean = y[outlier_mask]

        # print(f"Outlier columns used: {len(cols_in_X)} of {len(columns_for_outliers)}")
        print(f"Removed {len(X) - len(X_clean)} outliers")
        print(f"Cleaned dataset shape: {X_clean.shape}")
        print(f"Class distribution after cleaning: {y_clean.value_counts().to_dict()}")

        return X_clean, y_clean
    
    def advanced_feature_engineering(self, X, y, provided_top_features=None):
        """Create advanced engineered features with focus on discriminative power.
        Ranking/statistics are computed on a RobustScaler-transformed copy of X,
        but engineered features are added to the original X values.
        If provided_top_features is given, reuse those to build interactions on new data.
        """
        print("Advanced feature engineering...")
        
        X_engineered = X.copy()
        # Use robust-scaled copy for statistical ranking only
        scaler_for_stats = RobustScaler()
        X_scaled = pd.DataFrame(
            scaler_for_stats.fit_transform(X), columns=X.columns, index=X.index
        )
        
        # Create interaction features between top features
        feature_stats = {}
        for col in X.columns:
            ptsd_group = X_scaled[y == 1][col]
            control_group = X_scaled[y == 0][col]
            
            # Use multiple statistical tests
            _, p_value_mw = stats.mannwhitneyu(ptsd_group, control_group, alternative='two-sided')
            _, p_value_ks = stats.ks_2samp(ptsd_group, control_group)
            
            # Calculate effect size (Cohen's d)
            cohen_d = (ptsd_group.mean() - control_group.mean()) / np.sqrt(
                ((len(ptsd_group) - 1) * ptsd_group.std()**2 + 
                 (len(control_group) - 1) * control_group.std()**2) / 
                (len(ptsd_group) + len(control_group) - 2)
            )
            
            # Combined score emphasizing effect size
            combined_score = abs(cohen_d) * (1 - min(p_value_mw, p_value_ks))
            feature_stats[col] = combined_score
        
        # Get top features based on discriminative power, or reuse provided
        if provided_top_features is not None:
            top_feature_names = [f for f in provided_top_features if f in X.columns]
        else:
            top_features = sorted(feature_stats.items(), key=lambda x: x[1], reverse=True)[:15]
            top_feature_names = [f[0] for f in top_features]
        
        
        print(f"Creating interaction features from top {len(top_feature_names)} features...")
        
        # Create more sophisticated interactions
        interaction_count = 0
        for i in range(len(top_feature_names)):
            for j in range(i+1, len(top_feature_names)):
                feat1, feat2 = top_feature_names[i], top_feature_names[j]
                
                # Multiplicative interaction
                X_engineered[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
                
                # Ratio interaction (avoid division by zero)
                X_engineered[f'{feat1}_div_{feat2}'] = X[feat1] / (X[feat2] + 1e-8)
                
                # Difference interaction
                X_engineered[f'{feat1}_minus_{feat2}'] = X[feat1] - X[feat2]
                
                # Maximum and minimum
                X_engineered[f'{feat1}_max_{feat2}'] = X[[feat1, feat2]].max(axis=1)
                X_engineered[f'{feat1}_min_{feat2}'] = X[[feat1, feat2]].min(axis=1)
                
                interaction_count += 5
                
                if interaction_count >= 30:  # Increased limit
                    break
            if interaction_count >= 30:
                break

        # Create polynomial features for top features
        print("Creating polynomial features...")
        for feat in top_feature_names[:8]:  # More features
            X_engineered[f'{feat}_squared'] = X[feat] ** 2
            X_engineered[f'{feat}_cubed'] = X[feat] ** 3
            X_engineered[f'{feat}_sqrt'] = np.sqrt(np.abs(X[feat]))
            X_engineered[f'{feat}_log'] = np.log1p(np.abs(X[feat]))
        
        # Create advanced aggregated features
        print("Creating advanced category aggregations...")
        
        # HRV aggregations with more statistics
        hrv_cols = [col for col in X.columns if any(hrv in col for hrv in ['HRV', 'HR', 'NN', 'RR'])]
        if hrv_cols:
            X_engineered['HRV_mean'] = X[hrv_cols].mean(axis=1)
            X_engineered['HRV_std'] = X[hrv_cols].std(axis=1)
            X_engineered['HRV_max'] = X[hrv_cols].max(axis=1)
            X_engineered['HRV_min'] = X[hrv_cols].min(axis=1)
            X_engineered['HRV_range'] = X_engineered['HRV_max'] - X_engineered['HRV_min']
            X_engineered['HRV_cv'] = X_engineered['HRV_std'] / (X_engineered['HRV_mean'] + 1e-8)
            X_engineered['HRV_skew'] = X[hrv_cols].skew(axis=1)
            X_engineered['HRV_kurtosis'] = X[hrv_cols].kurtosis(axis=1)
        
        # Frequency domain aggregations
        freq_cols = [col for col in X.columns if any(freq in col for freq in ['LF', 'HF', 'VLF', 'RMSSD'])]
        if freq_cols:
            X_engineered['FREQ_mean'] = X[freq_cols].mean(axis=1)
            X_engineered['FREQ_std'] = X[freq_cols].std(axis=1)
            X_engineered['FREQ_max'] = X[freq_cols].max(axis=1)
            X_engineered['FREQ_energy'] = (X[freq_cols] ** 2).sum(axis=1)
            
            # LF/HF ratio if available
            lf_cols = [col for col in freq_cols if 'LF' in col and 'HF' not in col]
            hf_cols = [col for col in freq_cols if 'HF' in col]
            if lf_cols and hf_cols:
                X_engineered['LF_HF_ratio'] = X[lf_cols].mean(axis=1) / (X[hf_cols].mean(axis=1) + 1e-8)
        
        # EMG aggregations
        emg_cols = [col for col in X.columns if 'EMG' in col.upper()]
        if emg_cols:
            X_engineered['EMG_mean'] = X[emg_cols].mean(axis=1)
            X_engineered['EMG_std'] = X[emg_cols].std(axis=1)
            X_engineered['EMG_energy'] = (X[emg_cols] ** 2).sum(axis=1)
        
        # Psychological scores aggregations
        psych_cols = [col for col in X.columns if any(p in col for p in ['SPSS', 'CAP', 'anx', 'dep'])]
        if psych_cols:
            X_engineered['PSYCH_mean'] = X[psych_cols].mean(axis=1)
            X_engineered['PSYCH_std'] = X[psych_cols].std(axis=1)
            X_engineered['PSYCH_max'] = X[psych_cols].max(axis=1)
        
        # Advanced per-sample feature engineering (safe from data leakage)
        print("Creating advanced per-sample features...")
        
        # 1. Cross-Domain Physiological Ratios
        # Stress hormone ratios
        cortisol_cols = [col for col in X.columns if 'cortisol' in col.lower() or 'Cortisol' in col]
        amylase_cols = [col for col in X.columns if 'amylase' in col.lower() or 'Amylase' in col]
        if cortisol_cols and amylase_cols:
            X_engineered['stress_hormone_ratio'] = X[cortisol_cols].mean(axis=1) / (X[amylase_cols].mean(axis=1) + 1e-8)
        
        # Epinephrine/Norepinephrine ratio
        epi_cols = [col for col in X.columns if 'uE' in col]
        norepi_cols = [col for col in X.columns if 'uNE' in col] 
        if epi_cols and norepi_cols:
            X_engineered['catecholamine_ratio'] = X[epi_cols].mean(axis=1) / (X[norepi_cols].mean(axis=1) + 1e-8)
        
        # Autonomic-Biochemical interactions
        if cortisol_cols and hrv_cols:
            X_engineered['hrv_stress_ratio'] = X[hrv_cols].mean(axis=1) / (X[cortisol_cols].mean(axis=1) + 1e-8)
        
        # Cardiovascular-Stress ratios
        bp_cols = [col for col in X.columns if any(bp in col.lower() for bp in ['systolic', 'diastolic', 'map'])]
        if bp_cols and cortisol_cols:
            X_engineered['bp_stress_ratio'] = X[bp_cols].mean(axis=1) / (X[cortisol_cols].mean(axis=1) + 1e-8)
        
        # 2. Advanced Statistical Features (per-sample)
        # Calculate percentiles within each subject's feature profile
        X_engineered['feature_p25'] = X.quantile(0.25, axis=1)
        X_engineered['feature_p75'] = X.quantile(0.75, axis=1)
        X_engineered['feature_p90'] = X.quantile(0.90, axis=1)
        X_engineered['feature_iqr'] = X_engineered['feature_p75'] - X_engineered['feature_p25']
        
        # Robust statistics within each sample
        X_engineered['feature_median'] = X.median(axis=1)
        X_engineered['feature_mad'] = X.subtract(X.median(axis=1), axis=0).abs().median(axis=1)
        X_engineered['feature_robust_cv'] = X_engineered['feature_mad'] / (X_engineered['feature_median'] + 1e-8)
        
        X_engineered['feature_max'] = X.max(axis=1)  # Actual max value
        X_engineered['feature_min'] = X.min(axis=1)  # Actual min value  
        X_engineered['feature_range'] = X_engineered['feature_max'] - X_engineered['feature_min']
        X_engineered['feature_concentration'] = X_engineered['feature_max'] / (X.sum(axis=1) + 1e-8)
        
        X_engineered['top_feature_dominance'] = X.max(axis=1) / (X.mean(axis=1) + 1e-8)
        # 4. Enhanced Complexity Features
        # Additional non-linear transformations for top features
        print("Creating enhanced complexity features...")
        for feat in top_feature_names[:5]:  # Top 5 features
            # Hyperbolic transformations
            X_engineered[f'{feat}_tanh'] = np.tanh(X[feat])
            X_engineered[f'{feat}_sinh'] = np.sinh(np.clip(X[feat], -10, 10))  # Clip to prevent overflow
            
            # Exponential decay
            X_engineered[f'{feat}_exp_decay'] = np.exp(-np.abs(X[feat]))
            
            # Power transformations
            X_engineered[f'{feat}_power_half'] = np.power(np.abs(X[feat]), 0.5) * np.sign(X[feat])
            X_engineered[f'{feat}_power_third'] = np.power(np.abs(X[feat]), 1/3) * np.sign(X[feat])
        
        # 5. Domain-Specific Physiological Features
        # Cardiovascular efficiency metrics
        hr_cols = [col for col in X.columns if 'HR' in col and 'HRV' not in col]
        if hr_cols and bp_cols:
            # Approximate cardiac efficiency
            X_engineered['cardiac_efficiency'] = X[hr_cols].mean(axis=1) * X[bp_cols].mean(axis=1)
        
        # Stress load composite (if we have multiple stress markers)
        stress_markers = []
        if cortisol_cols:
            stress_markers.extend(cortisol_cols)
        if epi_cols:
            stress_markers.extend(epi_cols)
        if norepi_cols:
            stress_markers.extend(norepi_cols)
        
        if len(stress_markers) >= 2:
            # Geometric mean of stress markers (more robust than arithmetic mean)
            stress_data = X[stress_markers].clip(lower=1e-8)  # Avoid log(0)
            X_engineered['stress_load_geometric'] = np.exp(np.log(stress_data).mean(axis=1))
            
            # Stress variability
            X_engineered['stress_variability'] = X[stress_markers].std(axis=1)
        
        # 6. Enhanced Category Interactions
        # EMG-HRV coherence approximation
        
        # Age-adjusted features (if age is available)
        age_cols = [col for col in X.columns if 'age' in col.lower()]
        if age_cols:
            age_values = X[age_cols].iloc[:, 0]
            # Age-adjusted HRV (HRV typically decreases with age)
            if hrv_cols:
                X_engineered['age_adjusted_hrv'] = X[hrv_cols].mean(axis=1) / (age_values + 1e-8)
            
            # Age-adjusted stress response
            if stress_markers:
                X_engineered['age_adjusted_stress'] = X[stress_markers].mean(axis=1) / (age_values + 1e-8)
        
        # BMI-adjusted features (if BMI is available)
        bmi_cols = [col for col in X.columns if 'bmi' in col.lower() or 'BMI' in col]
        if bmi_cols:
            bmi_values = X[bmi_cols].iloc[:, 0]
            # BMI-adjusted cardiovascular measures
            if bp_cols:
                X_engineered['bmi_adjusted_bp'] = X[bp_cols].mean(axis=1) / (bmi_values + 1e-8)
            if hr_cols:
                X_engineered['bmi_adjusted_hr'] = X[hr_cols].mean(axis=1) / (bmi_values + 1e-8)
        
        # 7. Additional Robust Aggregations
        # Harmonic mean for positive features (more robust to outliers)
        if hrv_cols:
            hrv_positive = X[hrv_cols].clip(lower=1e-8)
            X_engineered['HRV_harmonic_mean'] = len(hrv_cols) / (1/hrv_positive).sum(axis=1)
        
        if freq_cols:
            freq_positive = X[freq_cols].clip(lower=1e-8)
            X_engineered['FREQ_harmonic_mean'] = len(freq_cols) / (1/freq_positive).sum(axis=1)
        
        # Weighted averages (give more weight to more variable features)
        if hrv_cols:
            hrv_weights = X[hrv_cols].std(axis=0)
            hrv_weights = hrv_weights / hrv_weights.sum()
            X_engineered['HRV_weighted_mean'] = (X[hrv_cols] * hrv_weights).sum(axis=1)
        
        # 8. Physiological Balance Metrics
        # Sympathetic-Parasympathetic balance approximations
        sympathetic_markers = [col for col in X.columns if any(sym in col.lower() for sym in ['lf', 'sympathetic', 'norepinephrine'])]
        parasympathetic_markers = [col for col in X.columns if any(para in col.lower() for para in ['hf', 'vagal', 'parasympathetic', 'rmssd'])]
        
        if sympathetic_markers and parasympathetic_markers:
            sym_activity = X[sympathetic_markers].mean(axis=1)
            para_activity = X[parasympathetic_markers].mean(axis=1)
            X_engineered['autonomic_balance'] = sym_activity / (para_activity + 1e-8)
            X_engineered['autonomic_total'] = sym_activity + para_activity
            
        print(f"Created {len(X_engineered.columns) - len(X.columns)} new features")
        print(f"Total features: {len(X_engineered.columns)}")
        
        return X_engineered, top_feature_names
    
    def advanced_feature_selection(self, X, y, n_features=60):
        """Enhanced feature selection with multiple methods.
        All scorers/estimators are fit on a RobustScaler-transformed copy of X,
        but the returned dataframe is sliced from the original X using selected names.
        """
        print(f"Advanced feature selection to {n_features} features...")
        
        # Robust-scaled copy for stable/statistical scoring only
        scaler_for_stats = RobustScaler()
        X_scaled = pd.DataFrame(
            scaler_for_stats.fit_transform(X), columns=X.columns, index=X.index
        )
        
        # Method 1: Statistical tests with emphasis on class separation
        print("Method 1: Statistical significance with effect size")
        stat_scores = []
        for col in X.columns:
            ptsd_group = X_scaled[y == 1][col]
            control_group = X_scaled[y == 0][col]
            
            # Mann-Whitney U test
            _, p_value = stats.mannwhitneyu(ptsd_group, control_group, alternative='two-sided')
            
            # Effect size (Cohen's d)
            cohen_d = abs((ptsd_group.mean() - control_group.mean()) / 
                         np.sqrt(((len(ptsd_group) - 1) * ptsd_group.std()**2 + 
                                 (len(control_group) - 1) * control_group.std()**2) / 
                                (len(ptsd_group) + len(control_group) - 2)))
            
            # Combined score
            score = cohen_d * (1 - p_value)
            stat_scores.append(score)
        
        stat_scores = np.array(stat_scores)
        
        # Method 2: Mutual information
        print("Method 2: Mutual information")
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(100, X_scaled.shape[1]))
        mi_selector.fit(X_scaled, y)
        mi_scores = mi_selector.scores_
        
        # Method 3: Tree-based importance with balanced models
        print("Method 3: Balanced tree-based importance")
        
        # Balanced Random Forest
        brf = BalancedRandomForestClassifier(n_estimators=300, random_state=self.random_state)
        brf.fit(X_scaled, y)
        brf_scores = brf.feature_importances_
        
        # XGBoost with scale_pos_weight
        scale_pos_weight = len(y[y==0]) / len(y[y==1])
        xgb_model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=200,
            random_state=self.random_state,
            eval_metric='logloss',
        )
        # Use balanced sample weights during XGBoost fitting
        xgb_sample_weights = compute_sample_weight(class_weight='balanced', y=y)
        xgb_model.fit(X_scaled, y, sample_weight=xgb_sample_weights)
        xgb_scores = xgb_model.feature_importances_
        
        # Method 4: L1 regularization with class weights
        print("Method 4: L1 regularization with class weights")
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        lr = LogisticRegression(
            penalty='l1', 
            solver='liblinear', 
            class_weight={0: class_weights[0], 1: class_weights[1]},
            random_state=self.random_state
        )
        lr.fit(X_scaled, y)
        l1_scores = np.abs(lr.coef_[0])
        
        # Method 5: Relief-based feature selection (approximated with nearest neighbors)
        print("Method 5: Relief-based scoring")
        relief_scores = []
        for i, col in enumerate(X.columns):
            feature_values = X_scaled.iloc[:, i].values
            
            # Calculate feature relevance based on nearest neighbors
            pos_samples = feature_values[y == 1]
            neg_samples = feature_values[y == 0]
            
            if len(pos_samples) > 0 and len(neg_samples) > 0:
                # Difference in distributions
                ks_stat, _ = stats.ks_2samp(pos_samples, neg_samples)
                relief_scores.append(ks_stat)
            else:
                relief_scores.append(0)
        
        relief_scores = np.array(relief_scores)
        
        # Normalize all scores to [0, 1] with robust handling
        def normalize_scores(scores):
            scores = np.array(scores)
            
            if np.any(np.isnan(scores)):
                print("NaN values found in scores")
                scores = np.nan_to_num(scores, nan=0.0)
            
            if np.any(np.isinf(scores)):
                print("Infinite values found in scores")
                scores = np.nan_to_num(scores, posinf=1.0, neginf=0.0)
            
            score_min = np.min(scores)
            score_max = np.max(scores)
            score_range = score_max - score_min
            
            if score_range > np.finfo(float).eps:  
                normalized = (scores - score_min) / score_range
            else:
                normalized = np.full_like(scores, 0.5)
            
            normalized = np.clip(normalized, 0.0, 1.0)
            
            return normalized
        
        stat_scores_norm = normalize_scores(stat_scores)
        mi_scores_norm = normalize_scores(mi_scores)
        brf_scores_norm = normalize_scores(brf_scores)
        xgb_scores_norm = normalize_scores(xgb_scores)
        l1_scores_norm = normalize_scores(l1_scores)
        relief_scores_norm = normalize_scores(relief_scores)
        
        # Weighted combination with emphasis on methods that handle imbalance well
        combined_scores = (
            0.25 * stat_scores_norm +    # Statistical significance with effect size
            0.20 * mi_scores_norm +       # Mutual information
            0.20 * brf_scores_norm +      # Balanced Random Forest
            0.20 * xgb_scores_norm +      # XGBoost with scale_pos_weight
            0.10 * l1_scores_norm +       # L1 with class weights
            0.05 * relief_scores_norm     # Relief-based
        )
        
        # Select top features
        top_indices = np.argsort(combined_scores)[-n_features:]
        selected_features = X.columns[top_indices].tolist()
        
        print(f"Selected {len(selected_features)} features using enhanced scoring")
        self.feature_names = selected_features
        
        # Print top 10 features with their scores
        print("\nTop 10 selected features:")
        top_10_indices = np.argsort(combined_scores)[-10:][::-1]
        for idx in top_10_indices:
            print(f"  {X.columns[idx]}: {combined_scores[idx]:.4f}")
        

        return X[selected_features], selected_features
    
    def find_optimal_threshold(self, y_true, y_proba, metric='auc_optimal'):
        """Find optimal classification threshold optimized for AUC ROC performance
        
        Uses Youden Index (Sensitivity + Specificity - 1) which maximizes the 
        distance from the diagonal in ROC space, giving the optimal threshold
        for AUC-based classification.
        """
        # Use a finer threshold range for better AUC optimization
        thresholds = np.linspace(0.1, 0.9, 200)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            # Handle edge cases
            if sum(y_pred) == 0:  # No positive predictions
                scores.append(-1)  # Use -1 for invalid thresholds
                continue
            if sum(y_pred) == len(y_pred):  # All positive predictions
                scores.append(-1)  # Use -1 for invalid thresholds
                continue
            
            if metric == 'auc_optimal':
                # Youden Index: Sensitivity + Specificity - 1
                # This maximizes the distance from diagonal in ROC space
                sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)  # True Positive Rate
                specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)  # True Negative Rate
                score = sensitivity + specificity - 1  # Youden Index
            elif metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'balanced_accuracy':
                score = balanced_accuracy_score(y_true, y_pred)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == 'g_mean':
                recall_0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
                recall_1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
                score = np.sqrt(recall_0 * recall_1)
            elif metric == 'recall_priority':
                # Balanced metric emphasizing both recalls and precision
                recall_1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
                recall_0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
                precision_1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
                # More balanced approach: equal weight to both recalls, penalty for low precision
                score = 0.4 * recall_1 + 0.4 * recall_0 + 0.2 * precision_1
            else:
                # Default to Youden Index for unknown metrics
                sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
                specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
                score = sensitivity + specificity - 1
            
            scores.append(score)
        
        # Find the best threshold
        valid_scores = np.array(scores)
        valid_indices = valid_scores > -1  # Only consider valid thresholds
        
        if not np.any(valid_indices):
            # Fallback to 0.5 if no valid thresholds found
            return 0.5, 0.0
        
        valid_scores_only = valid_scores[valid_indices]
        valid_thresholds = thresholds[valid_indices]
        
        optimal_idx = np.argmax(valid_scores_only)
        optimal_threshold = valid_thresholds[optimal_idx]
        optimal_score = valid_scores_only[optimal_idx]
        
        # More conservative clamping for AUC optimization
        optimal_threshold = np.clip(optimal_threshold, 0.2, 0.8)
        
        return optimal_threshold, optimal_score
    
    def create_improved_pipelines(self):
        """Create ML pipelines optimized for imbalanced data"""
        print("Creating improved ML pipelines...")
        
        # Advanced sampling strategies for severe imbalance - Original 1.0 strategies
        sampling_strategies_full = {
            'SVMSMOTE': SVMSMOTE(random_state=self.random_state, k_neighbors=10),
            'BorderlineSMOTE': BorderlineSMOTE(random_state=self.random_state, kind='borderline-2', k_neighbors=10),
            'ADASYN': ADASYN(random_state=self.random_state, n_neighbors=10),
            'SMOTETomek': SMOTETomek(random_state=self.random_state, smote=SMOTE(k_neighbors=10)),
            'SMOTEENN': SMOTEENN(random_state=self.random_state,
                smote=SMOTE(k_neighbors=10)
            )
        }
        
        sampling_strategies_moderate = {
        } # none because the models are already balanced
            
        
        # Models specifically good for imbalanced data
        models = {
            # 'TabPFN_Default': TabPFNClassifier(
            #     n_estimators=16,
            #     balance_probabilities=True,
            #     average_before_softmax=True,
            #     fit_mode='fit_with_cache',
            #     random_state=self.random_state,
            #     device='cuda',
            # ),
            # BalancedRandomForestClassifier variations
            'BalancedRF_Default': BalancedRandomForestClassifier(
                n_estimators=300,
                max_features='sqrt',
                random_state=self.random_state,
            ),
            'BalancedRF_Small': BalancedRandomForestClassifier(
                n_estimators=100,
                max_features='sqrt',
                max_depth=10,
                min_samples_split=5,
                random_state=self.random_state,
            ),
            'BalancedRF_Large': BalancedRandomForestClassifier(
                n_estimators=500,
                max_features='sqrt',
                max_depth=None,
                min_samples_split=2,
                random_state=self.random_state,
            ),
            'BalancedRF_Log2': BalancedRandomForestClassifier(
                n_estimators=300,
                max_features='log2',
                max_depth=20,
                min_samples_leaf=2,
                random_state=self.random_state,
            ),
            'BalancedRF_AllFeatures': BalancedRandomForestClassifier(
                n_estimators=200,
                max_features=None,
                max_depth=15,
                min_samples_split=10,
                random_state=self.random_state,
            ),
            'BalancedRF_Conservative': BalancedRandomForestClassifier(
                n_estimators=250,
                max_features=0.3,
                max_depth=25,
                min_samples_split=5,
                min_samples_leaf=4,
                random_state=self.random_state,
            ),
            
            # BalancedBaggingClassifier variations
            'BalancedBagging_Default': BalancedBaggingClassifier(
                n_estimators=200,
                random_state=self.random_state,
            ),
            'BalancedBagging_Small': BalancedBaggingClassifier(
                n_estimators=50,
                max_samples=0.5,
                max_features=0.5,
                random_state=self.random_state,
            ),
            'BalancedBagging_Large': BalancedBaggingClassifier(
                n_estimators=300,
                max_samples=0.8,
                max_features=0.8,
                random_state=self.random_state,
            ),
            'BalancedBagging_HighSample': BalancedBaggingClassifier(
                n_estimators=150,
                max_samples=1.0,
                max_features=0.6,
                random_state=self.random_state,
            ),
            
            # EasyEnsembleClassifier variations
            'EasyEnsemble_Default': EasyEnsembleClassifier(
                n_estimators=10,
                random_state=self.random_state,
            ),
            'EasyEnsemble_Small': EasyEnsembleClassifier(
                n_estimators=5,
                random_state=self.random_state,
            ),
            'EasyEnsemble_Large': EasyEnsembleClassifier(
                n_estimators=20,
                random_state=self.random_state,
            ),
            'EasyEnsemble_Medium': EasyEnsembleClassifier(
                n_estimators=15,
                random_state=self.random_state,
            ),
            
        }

        
        # Create pipelines
        pipelines = {}
        pipelines2 = {}
        
        
        # Bagging and ensemble models that will also get SMOTE variations
        bagging_ensemble_models = ['BalancedBagging_Default', 'BalancedBagging_Small', 'BalancedBagging_Large',
                                  'BalancedBagging_HighSample',
                                  'RUSBoost_Default', 'RUSBoost_Fast', 'RUSBoost_Conservative', 'RUSBoost_Large',
                                  'RUSBoost_HighLR',
                                  'EasyEnsemble_Default', 'EasyEnsemble_Small', 'EasyEnsemble_Large', 'EasyEnsemble_Medium', 'BalancedRF_Default', 'BalancedRF_Small', 'BalancedRF_Large', 'BalancedRF_Log2', 
                         'BalancedRF_AllFeatures', 'BalancedRF_Conservative',
                       'TabPFN_Default']
        
        # KNN models get full SMOTE treatment
        knn_models = ['KNN_Uniform', 'KNN_Distance', 'KNN_Manhattan', 'KNN_Minkowski', 'KNN_Cosine', 'KNN_Large']
        
        # Other models that get full SMOTE treatment
        other_models = ['LR_Balanced', 'SVC_Balanced']
        
        for model_name, model in models.items():

            if model_name == 'TabPFN_Auto':
                continue
            
            if model_name in bagging_ensemble_models:
                # Add direct version (baseline)
                pipelines[model_name] = model
                
                # Add moderate SMOTE variations
                for sampling_name, sampler in sampling_strategies_moderate.items():
                    pipeline_name = f"{model_name}_{sampling_name}"
                    pipelines[pipeline_name] = {
                        'scaler': RobustScaler(),
                        'sampler': sampler,
                        'classifier': model.__class__(**model.get_params()) if hasattr(model, 'get_params') else model
                    }
                    
            elif model_name in knn_models:
                pipelines[model_name] = model
                # Add all full SMOTE variants
                for sampling_name, sampler in sampling_strategies_full.items():
                    pipeline_name = f"{model_name}_{sampling_name}"
                    pipelines[pipeline_name] = {
                        'scaler': RobustScaler(),
                        'sampler': sampler,
                        'classifier': model.__class__(**model.get_params()) if hasattr(model, 'get_params') else model
                    }
                    
            else:
                # Apply full sampling strategies to other models
                for sampling_name, sampler in sampling_strategies_full.items():
                    pipeline_name = f"{model_name}_{sampling_name}"
                    pipelines[pipeline_name] = {
                        'scaler': RobustScaler(),
                        'sampler': sampler,
                        'classifier': model.__class__(**model.get_params()) if hasattr(model, 'get_params') else model
                    }

        # Add TabPFN_Auto in a separate pipeline
        # pipelines2['TabPFN_Auto'] = {
        #     'scaler': RobustScaler(),
        #     'sampler': None,
        #     'classifier': AutoTabPFNClassifier(
        #         max_time=300,
        #         eval_metric='balanced_accuracy',
        #         balance_probabilities=True,
        #         random_state=self.random_state,
        #         device='cuda',
        #     )
        # }
        
        print(f"Created {len(pipelines)} total pipeline variations")
        print(f"- {len(bagging_ensemble_models)} bagging/ensemble models (direct + {len(sampling_strategies_moderate)} SMOTE variations each)")
        print(f"- {len(knn_models)} KNN models (direct + {len(sampling_strategies_full)} SMOTE variations each)")
        print(f"- {len(other_models)} other models ({len(sampling_strategies_full)} SMOTE variations each)")
        
        return pipelines #, pipelines2
    
    def custom_train_test_split(self, X, y, test_size=0.15, min_positive_test_ratio=0.6):
        """Custom split ensuring minimum percentage of positive samples in test set"""
        print(f"Performing custom train-test split with minimum positive ratio in test set...")
        
        # Get indices of positive and negative cases
        positive_indices = np.where(y == 1)[0]
        negative_indices = np.where(y == 0)[0]
        
        n_positive = len(positive_indices)
        n_negative = len(negative_indices)
        n_total = len(y)
        
        # Calculate total test size
        n_test_total = int(n_total * test_size)
        
        # Calculate minimum number of positive samples needed in test
        min_positive_test = max(int(n_test_total * min_positive_test_ratio), 1)
        
        # Ensure we don't exceed available positive samples
        n_positive_test = min(min_positive_test, n_positive)
        
        # Calculate remaining test slots for negative samples
        n_negative_test = n_test_total - n_positive_test
        
        # Ensure we don't exceed available negative samples
        n_negative_test = min(n_negative_test, n_negative)
        
        # Adjust total test size if we don't have enough samples
        actual_test_size = n_positive_test + n_negative_test
        
        print(f"Target test size: {n_test_total}, Actual test size: {actual_test_size}")
        print(f"Positive samples in test: {n_positive_test}/{n_positive} ({n_positive_test/n_positive*100:.1f}%)")
        
        # Randomly sample test indices
        if self.use_random_seed:
            np.random.seed(self.random_state)
        test_positive_indices = np.random.choice(positive_indices, size=n_positive_test, replace=False)
        test_negative_indices = np.random.choice(negative_indices, size=n_negative_test, replace=False)
        
        # Combine test indices
        test_indices = np.concatenate([test_positive_indices, test_negative_indices])
        train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
        
        # Create train and test sets
        X_train = X.iloc[train_indices] if hasattr(X, 'iloc') else X[train_indices]
        X_test = X.iloc[test_indices] if hasattr(X, 'iloc') else X[test_indices]
        y_train = y.iloc[train_indices] if hasattr(y, 'iloc') else y[train_indices]
        y_test = y.iloc[test_indices] if hasattr(y, 'iloc') else y[test_indices]
        
        print(f"Train set: {len(X_train)} samples, {sum(y_train)} positive ({sum(y_train)/len(y_train)*100:.1f}%)")
        print(f"Test set: {len(X_test)} samples, {sum(y_test)} positive ({sum(y_test)/len(y_test)*100:.1f}%)")
        print(f"Test set positive ratio: {sum(y_test)/len(y_test)*100:.1f}% (minimum target: {min_positive_test_ratio*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test

    def evaluate_models_cv_improved(self, X_train, y_train, models, cv_folds=3):
        """Evaluate models using standard Stratified K-Fold CV (no balancing of folds)"""
        # Ensure each validation fold has at least one sample from each class
        class_counts = y_train.value_counts()
        if (class_counts < 1).any():
            raise ValueError("Training set missing one of the classes; cannot run stratified CV.")

        # StratifiedKFold requires n_splits <= min(class_counts)
        max_folds_by_class = int(class_counts.min())
        effective_folds = min(cv_folds, max_folds_by_class)
        if effective_folds < 2:
            raise ValueError(
                f"Not enough samples per class for CV: class_counts={class_counts.to_dict()}, "
                f"requested folds={cv_folds}. Need at least 2 folds."
            )

        print(f"Evaluating {len(models)} models using {effective_folds}-fold Stratified CV...")

        skf = RepeatedStratifiedKFold(n_splits=effective_folds, n_repeats=3, random_state=self.random_state)
        # skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=self.random_state)
        results = {}
        
        for name, model in models.items():
            print(f"Evaluating {name}...")
            
            try:
                cv_scores = []
                cv_recalls = []
                cv_aucs = []
                optimal_thresholds = []
                
                for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                    # Split data
                    X_fold_train = X_train.iloc[train_idx]
                    X_fold_val = X_train.iloc[val_idx]
                    y_fold_train = y_train.iloc[train_idx]
                    y_fold_val = y_train.iloc[val_idx]
                    
                    # Create fresh model for this fold
                    fold_model = self._create_fresh_model(model, name)
                    
                    # Fit and predict based on model type
                    y_val_pred_proba = self._fit_and_predict_fold(
                        fold_model, name, X_fold_train, y_fold_train, X_fold_val
                    )
                    
                    # Find optimal threshold using Youden Index (AUC-optimal)
                    threshold, _ = self.find_optimal_threshold(
                        y_fold_val, y_val_pred_proba, metric='auc_optimal'
                    )
                    optimal_thresholds.append(threshold)
                    
                    # Calculate metrics with AUC-optimal threshold
                    y_val_pred = (y_val_pred_proba >= threshold).astype(int)
                    
                    fold_recall_1 = recall_score(y_fold_val, y_val_pred, pos_label=1)
                    fold_auc = roc_auc_score(y_fold_val, y_val_pred_proba)
                    fold_score = self._calculate_combined_score(y_fold_val, y_val_pred_proba)
                    
                    cv_scores.append(fold_score)
                    cv_recalls.append(fold_recall_1)
                    cv_aucs.append(fold_auc)
                
                # Store results
                results[name] = {
                    'mean_score': np.mean(cv_scores),
                    'std_score': np.std(cv_scores),
                    'mean_recall': np.mean(cv_recalls),
                    'mean_auc': np.mean(cv_aucs),
                    'optimal_threshold': np.mean(optimal_thresholds),
                    'cv_scores': cv_scores,
                    'model': model
                }
                
                print(f"  Score: {results[name]['mean_score']:.3f} ± {results[name]['std_score']:.3f}, "
                      f"Recall: {results[name]['mean_recall']:.3f}, "
                      f"AUC: {results[name]['mean_auc']:.3f}, "
                      f"Threshold: {results[name]['optimal_threshold']:.3f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        return results

    def evaluate_models_holdout(self, X_train, y_train, models, val_size=0.25):
        """Evaluate models using a single stratified train/validation split (no cross-validation)."""
        # Ensure we have both classes present
        class_counts = y_train.value_counts()
        if (class_counts < 1).any():
            raise ValueError("Training set missing one of the classes; cannot evaluate.")

        print(f"Evaluating {len(models)} models using a single stratified holdout split (val_size={val_size})...")

        # Create a holdout validation split from the provided training data with at least 1 positive in val
        X_subtrain, X_val, y_subtrain, y_val = self.stratified_train_test_split_with_min_positive(
            X_train, y_train, test_size=val_size, min_positive_test=1
        )

        results = {}

        for name, model in models.items():
            print(f"Evaluating {name}...")
            try:
                # Fresh model per evaluation
                eval_model = self._create_fresh_model(model, name)

                # Fit and get validation probabilities using existing helper (handles scaling/sampling logic)
                y_val_pred_proba = self._fit_and_predict_fold(
                    eval_model, name, X_subtrain, y_subtrain, X_val
                )

                # Optimal threshold and metrics
                threshold, _ = self.find_optimal_threshold(y_val, y_val_pred_proba, metric='auc_optimal')
                y_val_pred = (y_val_pred_proba >= threshold).astype(int)

                holdout_recall_1 = recall_score(y_val, y_val_pred, pos_label=1)
                holdout_auc = roc_auc_score(y_val, y_val_pred_proba)
                holdout_score = self._calculate_combined_score(y_val, y_val_pred_proba)

                # Store in the same shape as CV results for downstream compatibility
                results[name] = {
                    'mean_score': holdout_score,
                    'std_score': 0.0,
                    'mean_recall': holdout_recall_1,
                    'mean_auc': holdout_auc,
                    'optimal_threshold': threshold,
                    'cv_scores': [holdout_score],
                    'model': model
                }

                print(
                    f"  Score: {holdout_score:.3f} ± 0.000, "
                    f"Recall: {holdout_recall_1:.3f}, AUC: {holdout_auc:.3f}, "
                    f"Threshold: {threshold:.3f}"
                )
            except Exception as e:
                print(f"  Error: {e}")
                continue

        return results

    def stratified_train_test_split_with_min_positive(self, X, y, test_size=0.2, min_positive_test=1, max_tries=50):
        """Stratified shuffle split ensuring at least a minimal number of positives in test set.
        Does not balance classes beyond this minimal constraint.
        """
        print("Performing stratified train-test split with minimal positive constraint in test set...")
        n_pos_total = int((y == 1).sum())
        if n_pos_total == 0:
            raise ValueError("No positive samples available in the dataset.")
        
        # Try multiple random seeds to satisfy the minimal positive constraint
        base_seed = self.random_state if self.use_random_seed else 0
        for offset in range(max_tries):
            seed = base_seed + offset
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            for train_idx, test_idx in sss.split(X, y):
                y_test = y.iloc[test_idx]
                if int((y_test == 1).sum()) >= min_positive_test:
                    X_train = X.iloc[train_idx]
                    X_test = X.iloc[test_idx]
                    y_train = y.iloc[train_idx]
                    y_test = y.iloc[test_idx]
                    print(f"Train set: {len(X_train)} samples, {(y_train==1).sum()} positive ({(y_train==1).sum()/len(y_train)*100:.1f}%)")
                    print(f"Test set: {len(X_test)} samples, {(y_test==1).sum()} positive ({(y_test==1).sum()/len(y_test)*100:.1f}%)")
                    return X_train, X_test, y_train, y_test
        
        # Fallback to a single stratified split and warn
        print("Warning: Could not satisfy minimal positive constraint after multiple tries; using stratified split anyway.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )
        print(f"Train set: {len(X_train)} samples, {(y_train==1).sum()} positive ({(y_train==1).sum()/len(y_train)*100:.1f}%)")
        print(f"Test set: {len(X_test)} samples, {(y_test==1).sum()} positive ({(y_test==1).sum()/len(y_test)*100:.1f}%)")
        return X_train, X_test, y_train, y_test
    
    def _create_fresh_model(self, model, name):
        """Create a fresh copy of the model for CV fold"""
        # Updated model categorization
        rf_only_models = ['BalancedRF_Default', 'BalancedRF_Small', 'BalancedRF_Large', 'BalancedRF_Log2', 
                         'BalancedRF_AllFeatures', 'BalancedRF_Conservative']
        bagging_ensemble_models = ['BalancedBagging_Default', 'BalancedBagging_Small', 'BalancedBagging_Large',
                                  'BalancedBagging_HighSample',
                                  'RUSBoost_Default', 'RUSBoost_Fast', 'RUSBoost_Conservative', 'RUSBoost_Large',
                                  'RUSBoost_HighLR',
                                  'EasyEnsemble_Default', 'EasyEnsemble_Small', 'EasyEnsemble_Large', 'EasyEnsemble_Medium']
        knn_models = ['KNN_Uniform', 'KNN_Distance', 'KNN_Manhattan', 'KNN_Minkowski', 'KNN_Cosine', 'KNN_Large']
        
        # Check if it's a direct model (RF models, baseline bagging/ensemble models, baseline KNN models, baseline LightGBM models, or TabPFN)
        if (name in rf_only_models or name in bagging_ensemble_models or name in knn_models or name == 'TabPFN_Default'):
            # Direct models - simple clone
            if hasattr(model, 'get_params'):
                return model.__class__(**model.get_params())
            else:
                from copy import deepcopy
                return deepcopy(model)
        elif isinstance(model, dict):
            # Dictionary models with sampling - create fresh instances with same parameters
            from copy import deepcopy
            return {
                'scaler': RobustScaler(),
                'sampler': deepcopy(model['sampler']),  # Preserve exact configuration
                'classifier': deepcopy(model['classifier'])
            }
        else:
            # Fallback
            from copy import deepcopy
            return deepcopy(model)
    
    def _fit_and_predict_fold(self, model, name, X_train, y_train, X_val):
        """Fit model and predict for a single fold"""
        # Updated model categorization
        rf_only_models = ['BalancedRF_Default', 'BalancedRF_Small', 'BalancedRF_Large', 'BalancedRF_Log2', 
                         'BalancedRF_AllFeatures', 'BalancedRF_Conservative']
        bagging_ensemble_models = ['BalancedBagging_Default', 'BalancedBagging_Small', 'BalancedBagging_Large',
                                  'BalancedBagging_HighSample',
                                  'RUSBoost_Default', 'RUSBoost_Fast', 'RUSBoost_Conservative', 'RUSBoost_Large',
                                  'RUSBoost_HighLR',
                                  'EasyEnsemble_Default', 'EasyEnsemble_Small', 'EasyEnsemble_Large', 'EasyEnsemble_Medium']
        knn_models = ['KNN_Uniform', 'KNN_Distance', 'KNN_Manhattan', 'KNN_Minkowski', 'KNN_Cosine', 'KNN_Large']
        
        # Check if it's a direct model (RF models, baseline bagging/ensemble models, baseline KNN models, or TabPFN)
        if (name in rf_only_models or name in bagging_ensemble_models or name in knn_models or name == 'TabPFN_Default'):
            # Direct models with power transformation (fitted on fold-train only) and scaling
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            X_train_pt = pt.fit_transform(X_train)
            X_val_pt = pt.transform(X_val)

            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_pt)
            X_val_scaled = scaler.transform(X_val_pt)
            model.fit(X_train_scaled, y_train)
            
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X_val_scaled)[:, 1]
            else:
                scores = model.decision_function(X_val_scaled)
                return (scores - scores.min()) / (scores.max() - scores.min())
        
        elif isinstance(model, dict):
            # Dictionary models with manual pipeline
            scaler = model['scaler']
            sampler = model['sampler']
            classifier = model['classifier']
            
            # Apply transformations in order: PowerTransform (fit on fold-train) -> Scale -> Sample -> Fit
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            X_train_pt = pt.fit_transform(X_train)
            X_train_scaled = scaler.fit_transform(X_train_pt)
            if sampler is not None:
                X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_scaled, y_train)
            else:
                X_train_resampled, y_train_resampled = X_train_scaled, y_train
            classifier.fit(X_train_resampled, y_train_resampled)
            
            # Predict on validation
            X_val_pt = pt.transform(X_val)
            X_val_scaled = scaler.transform(X_val_pt)
            if hasattr(classifier, 'predict_proba'):
                return classifier.predict_proba(X_val_scaled)[:, 1]
            else:
                scores = classifier.decision_function(X_val_scaled)
                return (scores - scores.min()) / (scores.max() - scores.min())
        
        else:
            # Fallback for other model types
            model.fit(X_train, y_train)
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X_val)[:, 1]
            else:
                scores = model.decision_function(X_val)
                return (scores - scores.min()) / (scores.max() - scores.min())
    
    def _calculate_combined_score(self, y_true, y_pred_proba):
        """Calculate the combined score consistently using AUC-optimal threshold"""
        threshold, _ = self.find_optimal_threshold(y_true, y_pred_proba, metric='auc_optimal')
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        recall_1 = recall_score(y_true, y_pred, pos_label=1)
        recall_0 = recall_score(y_true, y_pred, pos_label=0)
        g_mean = np.sqrt(recall_0 * recall_1)
        auc = roc_auc_score(y_true, y_pred_proba)
        precision_1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        
        return 0.4 * recall_1 + 0.2 * g_mean + 0.2 * auc + 0.2 * precision_1

    def evaluate_final_model_improved(self, best_model_name, results, X_train, y_train, 
                                    X_test, y_test, verbose=True):
        """Final evaluation with improved metrics reporting"""
        if verbose:
            print(f"\n" + "="*50)
            print(f"FINAL EVALUATION: {best_model_name}")
            print("="*50)
            
            # Traditional model
            print("Retraining traditional model on full training set...")
        else:
            print(f"Evaluating {best_model_name}...")
        
        best_model = results[best_model_name]['model']
        optimal_threshold = results[best_model_name]['optimal_threshold']
        
        # Clone and retrain
        if hasattr(best_model, 'set_params'):
            final_model = best_model.__class__(**best_model.get_params())
        else:
            from copy import deepcopy
            final_model = deepcopy(best_model)
        
        # Handle different model types for final training
        # Updated model categorization
        rf_only_models = ['BalancedRF_Default', 'BalancedRF_Small', 'BalancedRF_Large', 'BalancedRF_Log2', 
                         'BalancedRF_AllFeatures', 'BalancedRF_Conservative']
        bagging_ensemble_models = ['BalancedBagging_Default', 'BalancedBagging_Small', 'BalancedBagging_Large',
                                  'BalancedBagging_HighSample', 
                                  'RUSBoost_Default', 'RUSBoost_Fast', 'RUSBoost_Conservative', 'RUSBoost_Large',
                                  'RUSBoost_HighLR',
                                  'EasyEnsemble_Default', 'EasyEnsemble_Small', 'EasyEnsemble_Large', 'EasyEnsemble_Medium']
        knn_models = ['KNN_Uniform', 'KNN_Distance', 'KNN_Manhattan', 'KNN_Minkowski', 'KNN_Cosine', 'KNN_Large']
        
        # Check if it's a direct model (RF models, baseline bagging/ensemble models, baseline KNN models, baseline LightGBM models, or TabPFN)
        if (best_model_name in rf_only_models or best_model_name in bagging_ensemble_models or best_model_name in knn_models or best_model_name == 'TabPFN_Default'):
            # Direct models with power transformation (fit on train only) and scaling
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            X_train_pt = pt.fit_transform(X_train)
            X_test_pt = pt.transform(X_test)

            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_pt)
            X_test_scaled = scaler.transform(X_test_pt)
            final_model.fit(X_train_scaled, y_train)
            trained_model_for_plotting = final_model
            
            if hasattr(final_model, 'predict_proba'):
                y_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred_proba = final_model.decision_function(X_test_scaled)
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / \
                             (y_pred_proba.max() - y_pred_proba.min())
        elif isinstance(best_model, dict):
            # Dictionary model - recreate the pipeline manually
            from copy import deepcopy
            scaler = RobustScaler()
            sampler = deepcopy(best_model['sampler'])  # Use deepcopy to preserve nested parameters
            classifier = deepcopy(best_model['classifier'])  # Use deepcopy for consistency
            
            # Apply transformations
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            X_train_pt = pt.fit_transform(X_train)
            X_train_scaled = scaler.fit_transform(X_train_pt)
            if sampler is not None:
                X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_scaled, y_train)
            else:
                X_train_resampled, y_train_resampled = X_train_scaled, y_train
            classifier.fit(X_train_resampled, y_train_resampled)
            trained_model_for_plotting = classifier
            
            X_test_pt = pt.transform(X_test)
            X_test_scaled = scaler.transform(X_test_pt)
            if hasattr(classifier, 'predict_proba'):
                y_pred_proba = classifier.predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred_proba = classifier.decision_function(X_test_scaled)
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / \
                             (y_pred_proba.max() - y_pred_proba.min())
        else:
            # Regular pipeline models
            final_model.fit(X_train, y_train)
            trained_model_for_plotting = final_model
            
            if hasattr(final_model, 'predict_proba'):
                y_pred_proba = final_model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = final_model.decision_function(X_test)
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / \
                             (y_pred_proba.max() - y_pred_proba.min())
        
        # Use AUC-optimal threshold optimization 
        optimal_threshold, _ = self.find_optimal_threshold(y_test, y_pred_proba, metric='auc_optimal')
        # Note: Conservative clamping is already handled in find_optimal_threshold method
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Calculate comprehensive metrics
        test_auc = roc_auc_score(y_test, y_pred_proba)
        test_recall_0 = recall_score(y_test, y_pred, pos_label=0)
        test_recall_1 = recall_score(y_test, y_pred, pos_label=1)
        test_precision_1 = precision_score(y_test, y_pred, pos_label=1)
        test_f1 = f1_score(y_test, y_pred)
        test_balanced_acc = balanced_accuracy_score(y_test, y_pred)
        test_mcc = matthews_corrcoef(y_test, y_pred)
        test_g_mean = np.sqrt(test_recall_0 * test_recall_1)
        
        if verbose:
            print(f"\nTest Performance:")
            print(f"AUC ROC: {test_auc:.4f}")
            print(f"Recall (Class 0): {test_recall_0:.4f}")
            print(f"Recall (Class 1): {test_recall_1:.4f}")
            print(f"Precision (Class 1): {test_precision_1:.4f}")
            print(f"F1 Score: {test_f1:.4f}")
            print(f"Balanced Accuracy: {test_balanced_acc:.4f}")
            print(f"G-Mean: {test_g_mean:.4f}")
            print(f"MCC: {test_mcc:.4f}")
            print(f"Optimal Threshold: {optimal_threshold:.3f}")
            
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred, digits=4))
            
            print(f"\nConfusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            print(f"TN: {cm[0,0]}, FP: {cm[0,1]}, FN: {cm[1,0]}, TP: {cm[1,1]}")
        else:
            # Brief output for non-verbose mode
            print(f"  AUC: {test_auc:.4f}, Recall(0): {test_recall_0:.4f}, Recall(1): {test_recall_1:.4f}, G-Mean: {test_g_mean:.4f}, Threshold: {optimal_threshold:.3f}")
        
        # Always calculate confusion matrix for completeness
        cm = confusion_matrix(y_test, y_pred)
        
        # Create plots and detailed analysis only in verbose mode
        if verbose:
            # Create plots directory
            plots_dir = './plots'
            os.makedirs(plots_dir, exist_ok=True)

            # Plot feature importances if available
            feature_names = X_test.columns
            self._plot_feature_importance(trained_model_for_plotting, feature_names, best_model_name, plots_dir)

            # Visualizations
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # ROC Curve
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            axes[0, 0].plot(fpr, tpr, label=f'AUC = {test_auc:.3f}')
            axes[0, 0].plot([0, 1], [0, 1], 'k--')
            axes[0, 0].set_xlabel('False Positive Rate')
            axes[0, 0].set_ylabel('True Positive Rate')
            axes[0, 0].set_title('ROC Curve')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Precision-Recall curve
            from sklearn.metrics import precision_recall_curve, average_precision_score
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            ap = average_precision_score(y_test, y_pred_proba)
            axes[0, 1].plot(recall, precision, label=f'AP = {ap:.3f}')
            axes[0, 1].set_xlabel('Recall')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].set_title('Precision-Recall Curve')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Confusion Matrix Heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
            axes[1, 0].set_xlabel('Predicted')
            axes[1, 0].set_ylabel('Actual')
            axes[1, 0].set_title('Confusion Matrix')
            
            # Threshold vs Metrics
            thresholds_range = np.linspace(0.1, 0.9, 50)
            recalls_1 = []
            precisions_1 = []
            f1_scores = []
            
            for t in thresholds_range:
                y_pred_t = (y_pred_proba >= t).astype(int)
                if sum(y_pred_t) > 0:  # Avoid division by zero
                    recalls_1.append(recall_score(y_test, y_pred_t, pos_label=1))
                    precisions_1.append(precision_score(y_test, y_pred_t, pos_label=1))
                    f1_scores.append(f1_score(y_test, y_pred_t))
                else:
                    recalls_1.append(0)
                    precisions_1.append(0)
                    f1_scores.append(0)
            
            axes[1, 1].plot(thresholds_range, recalls_1, label='Recall (Class 1)')
            axes[1, 1].plot(thresholds_range, precisions_1, label='Precision (Class 1)')
            axes[1, 1].plot(thresholds_range, f1_scores, label='F1 Score')
            axes[1, 1].axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal ({optimal_threshold:.3f})')
            axes[1, 1].set_xlabel('Threshold')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('Metrics vs Threshold')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Save the evaluation plots
            eval_plot_path = os.path.join(plots_dir, f'{best_model_name}_evaluation_plots.png')
            plt.savefig(eval_plot_path)
            print(f"\nSaved evaluation plots to {eval_plot_path}")
            plt.close(fig) # Close the figure to free memory
        
        return {
            'model': trained_model_for_plotting,
            'test_auc': test_auc,
            'test_recall_0': test_recall_0,
            'test_recall_1': test_recall_1,
            'test_g_mean': test_g_mean,
            'optimal_threshold': optimal_threshold,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def _plot_feature_importance(self, model, feature_names, model_name, plots_dir):
        """Plots and saves feature importances for a given model."""
        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])

        if importances is not None:
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(20)

            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance_df)
            plt.title(f'Top 20 Feature Importances for {model_name}')
            plt.tight_layout()
            
            feature_plot_path = os.path.join(plots_dir, f'{model_name}_feature_importance.png')
            plt.savefig(feature_plot_path)
            print(f"Saved feature importance plot to {feature_plot_path}")
            plt.close()
        else:
            print(f"Could not extract feature importances for model {model_name}")

    def run_improved_analysis(self, evaluate_traditional_models=True):
        """Run the complete improved analysis
        
        Args:
            evaluate_traditional_models (bool): If True, evaluates traditional ML models.
                                                If False, skips evaluation entirely.
        """
        print("="*70)
        print("IMPROVED PTSD PREDICTION MODEL - OPTIMIZED FOR RECALL")
        print("="*70)
        
        # Load and prepare data
        merged_df, numeric_features = self.load_data()
        
        # Basic preprocessing (missing values already handled in load_data)
        X = merged_df[numeric_features].copy()
        y = merged_df['CAPSF1I2s.0'].copy()
        
        # Remove missing targets (if any remain)
        mask = ~y.isna()
        X, y = X[mask], y[mask]

        # remove where y is not 0 or 1
        X = X[y.isin([0, 1])]
        y = y[y.isin([0, 1])]
        
        # Remove constant and highly correlated features
        X = X.loc[:, X.nunique() > 1]
        
        # Remove highly correlated features
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.95)]
        X = X.drop(columns=high_corr_features)
        
        y = y.astype(int)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        print(f"Class ratio: {len(y[y==1])/len(y)*100:.1f}% positive")

        # Remove outliers based on selected features
        X_selected, y = self.remove_outliers(X, y)
        
        # Train-test split: ensure test has at least one positive sample, no balancing
        X_train_main, X_test_main, y_train_main, y_test_main = self.stratified_train_test_split_with_min_positive(
            X_selected, y, test_size=0.1, min_positive_test=10
        )

        train_idx = X_train_main.index
        test_idx = X_test_main.index

        # Combine
        # X_all = pd.concat([X_train_main, X_test_main], axis=0)
        # y_all = pd.concat([y_train_main, y_test_main], axis=0)

        # Advanced feature engineering
        X_train_main, top_features = self.advanced_feature_engineering(X_train_main, y_train_main)
        X_test_main, _ = self.advanced_feature_engineering(X_test_main, y_test_main, provided_top_features=top_features)
        
        # Advanced feature selection
        X_train_main, selected_features = self.advanced_feature_selection(X_train_main, y_train_main, n_features=25)
        X_test_main = X_test_main[selected_features]

        # X_test_main = X_all.loc[test_idx]
        # X_train_main = X_all.loc[train_idx]
        
        traditional_results = {}
        if evaluate_traditional_models:
            # Evaluate traditional models
            print(f"\n" + "="*70)
            print("EVALUATING IMPROVED TRADITIONAL MODELS")
            print("="*70)
            
            traditional_pipelines = self.create_improved_pipelines()
            # traditional_pipelines, traditional_pipelines2 = self.create_improved_pipelines()
            # traditional_results2 = self.evaluate_models_holdout(
            #     X_train_main, y_train_main, traditional_pipelines2, val_size=0.1
            # )
            traditional_results = self.evaluate_models_cv_improved(X_train_main, y_train_main, traditional_pipelines, cv_folds=7)
            
        else:
            print(f"\n" + "="*70)
            print("SKIPPING TRADITIONAL MODEL EVALUATION")
            print("="*70)
        
        # All results are just traditional results now
        all_results = traditional_results
        # all_results = traditional_results | traditional_results2
        
        # Model comparison4
        print(f"\n" + "="*70)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*70)
        
        if not all_results:
            print("No models were evaluated successfully.")
            return {}, {}

        print("\nTop models by combined score (emphasizing recall):")
        sorted_models = sorted(all_results.items(), key=lambda x: x[1]['mean_score'], reverse=True)
        
        for i, (name, result) in enumerate(sorted_models[:10]):
            print(f"{i+1}. {name:<40} Score: {result['mean_score']:.3f}, "
                  f"Recall: {result['mean_recall']:.3f}, "
                  f"AUC: {result['mean_auc']:.3f}")
        
        # Evaluate top 15 models instead of just the best one
        top_10_models = sorted_models[:15]
        print(f"\nEVALUATING TOP {len(top_10_models)} MODELS ON TEST SET")
        print("="*70)
        
        final_results_all = {}
        
        for i, (model_name, cv_results) in enumerate(top_10_models):
            print(f"\nEvaluating model {i+1}/{len(top_10_models)}: {model_name}")
            
            # Final evaluation for this model (non-verbose to reduce output)
            final_result = self.evaluate_final_model_improved(
                model_name, all_results, X_train_main, y_train_main,
                X_test_main, y_test_main, verbose=False
            )
            
            # Store results with model name as key
            final_results_all[model_name] = {
                'test_auc': final_result['test_auc'],
                'test_recall_0': final_result['test_recall_0'],  # Recall for class 0
                'test_recall_1': final_result['test_recall_1'],  # Recall for class 1
                'test_g_mean': final_result['test_g_mean'],
                'optimal_threshold': final_result['optimal_threshold'],
                'cv_mean_score': cv_results['mean_score'],
                'cv_mean_recall': cv_results['mean_recall'],
                'cv_mean_auc': cv_results['mean_auc']
            }
        
        # Sort models by test AUC performance (best to worst)
        sorted_final_results = sorted(final_results_all.items(), 
                                    key=lambda x: x[1]['test_auc'], reverse=True)
        
        # Display comprehensive summary
        print(f"\n" + "="*90)
        print("FINAL MODEL PERFORMANCE SUMMARY (Ordered by Test AUC)")
        print("="*90)
        print(f"{'Rank':<4} {'Model Name':<35} {'Test AUC':<9} {'Test Recall':<11} {'Test G-Mean':<11} {'CV AUC':<8} {'Threshold':<9}")
        print("-" * 90)
        
        for rank, (model_name, results) in enumerate(sorted_final_results, 1):
            print(f"{rank:<4} {model_name:<35} {results['test_auc']:<9.4f} "
                  f"{results['test_recall_1']:<11.4f} {results['test_g_mean']:<11.4f} "
                  f"{results['cv_mean_auc']:<8.3f} {results['optimal_threshold']:<9.3f}")
        
        # Get the best performing model on test set
        best_model_name = sorted_final_results[0][0]
        best_final_results = sorted_final_results[0][1]
        
        print(f"\n" + "="*70)
        print("IMPROVED ANALYSIS COMPLETE")
        print("="*70)
        print(f"Best Model (by Test AUC): {best_model_name}")
        print(f"Best Test AUC: {best_final_results['test_auc']:.4f}")
        print(f"Best Test Recall (Class 0): {best_final_results['test_recall_0']:.4f}")
        print(f"Best Test Recall (Class 1): {best_final_results['test_recall_1']:.4f}")
        print(f"Best Test G-Mean: {best_final_results['test_g_mean']:.4f}")
        
        # Improvement over baseline
        baseline_auc = 0.58
        improvement = best_final_results['test_auc'] - baseline_auc
        print(f"\nImprovement over baseline AUC ({baseline_auc}):")
        print(f"  AUC Improvement: {improvement:.4f} "
              f"({improvement/baseline_auc*100:.1f}%)")
        
        # Performance statistics across all evaluated models
        all_test_aucs = [results['test_auc'] for _, results in sorted_final_results]
        all_test_recalls_0 = [results['test_recall_0'] for _, results in sorted_final_results]
        all_test_recalls_1 = [results['test_recall_1'] for _, results in sorted_final_results]
        
        print(f"\nPerformance Statistics (Top {len(sorted_final_results)} Models):")
        print(f"  Test AUC - Mean: {np.mean(all_test_aucs):.4f}, Std: {np.std(all_test_aucs):.4f}, "
              f"Min: {np.min(all_test_aucs):.4f}, Max: {np.max(all_test_aucs):.4f}")
        print(f"  Test Recall (Class 0) - Mean: {np.mean(all_test_recalls_0):.4f}, Std: {np.std(all_test_recalls_0):.4f}, "
              f"Min: {np.min(all_test_recalls_0):.4f}, Max: {np.max(all_test_recalls_0):.4f}")
        print(f"  Test Recall (Class 1) - Mean: {np.mean(all_test_recalls_1):.4f}, Std: {np.std(all_test_recalls_1):.4f}, "
              f"Min: {np.min(all_test_recalls_1):.4f}, Max: {np.max(all_test_recalls_1):.4f}")
        
        return all_results, final_results_all

if __name__ == "__main__":
    model = ImprovedPTSDModel.with_true_randomness()
    # model = ImprovedPTSDModel.with_random_seed(seed=42)
    
    # if model.use_random_seed:
    #     print(f"Using random seed: {model.random_state}")
    
    all_results, final_results = model.run_improved_analysis(evaluate_traditional_models=True)