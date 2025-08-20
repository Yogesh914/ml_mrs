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
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import (StratifiedKFold, StratifiedShuffleSplit, cross_val_score, train_test_split, RepeatedStratifiedKFold,
                                   GridSearchCV, RandomizedSearchCV)
from sklearn.feature_selection import (SelectKBest, f_classif, RFE, SelectFromModel,
                                     mutual_info_classif)
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.ensemble import (BalancedRandomForestClassifier, BalancedBaggingClassifier,
                              RUSBoostClassifier, EasyEnsembleClassifier)
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import warnings
import os
warnings.filterwarnings('ignore')

class PTSDData:
    """Enhanced PTSD Prediction Model with focus on improving recall for positive class"""
    
    def __init__(self, use_random_seed=True, random_state=42):
        self.use_random_seed = use_random_seed
        self.random_state = random_state if use_random_seed else None
        self.feature_names = None
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
    
    def load_data(self, file_path='pre_deployment_data.pkl'):
        """Load and prepare the data with sophisticated missing value handling"""
        print("Loading data...")
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        self.merged = data_dict['merged_df']
        self.HRV_features = data_dict['HRV_features']
        self.EMG_features = data_dict['EMG_features']
        self.CUT_features = data_dict['CUT_feature']
        self.SPSS_features = data_dict['SPSS_feature']
        self.categorical_features = data_dict['categorical_features']
        self.PTSD_features = data_dict['PTSD_features']
        
        
        # Combine all numerical features
        self.all_numeric_features = (self.HRV_features + self.CUT_features + 
                                   self.SPSS_features + self.EMG_features)
        
        print(f"Total numerical features: {len(self.all_numeric_features)}")
        
        # Apply sophisticated missing value handling
        self.merged = self._handle_missing_values(self.merged, self.all_numeric_features)
        
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

        # Columns to use for outlier detection only
        columns_for_outliers = [
            'HRV_SI', 'habituation', 'base', 'pNPY', 'HRV_MFDFA_alpha1_Delta',
            'HRV_CMSEn', 'HRV_GI', 'p114_t1', 'HRV_Cd', 'sCotinine',
            'HRV_MFDFA_alpha1_Asymmetry', 'p95', 'SAI_idx', 'uNE', 'sCortisol'
        ]

        cols_in_X = [c for c in columns_for_outliers if c in X.columns]
        if len(cols_in_X) == 0:
            print("No specified columns present for outlier detection; skipping removal.")
            return X, y

        # Use IQR method with less aggressive thresholds on the specified columns only
        Q1 = X[cols_in_X].quantile(0.15)
        Q3 = X[cols_in_X].quantile(0.85)
        IQR = Q3 - Q1
        outlier_mask = ~((X[cols_in_X] < (Q1 - 1.5 * IQR)) | (X[cols_in_X] > (Q3 + 1.5 * IQR))).any(axis=1)

        X_clean = X[outlier_mask]
        y_clean = y[outlier_mask]

        print(f"Outlier columns used: {len(cols_in_X)} of {len(columns_for_outliers)}")
        print(f"Removed {len(X) - len(X_clean)} outliers")
        print(f"Cleaned dataset shape: {X_clean.shape}")
        print(f"Class distribution after cleaning: {y_clean.value_counts().to_dict()}")

        return X_clean, y_clean
    
    def advanced_feature_engineering(self, X, y):
        """Create advanced engineered features with focus on discriminative power"""
        print("Advanced feature engineering...")
        
        X_engineered = X.copy()
        
        # Create interaction features between top features
        feature_stats = {}
        for col in X.columns:
            ptsd_group = X[y == 1][col]
            control_group = X[y == 0][col]
            
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
        
        # Get top features based on discriminative power
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

        X_engineered['HR_5min.0_max_HRV_pNN20'] = X[['HR_5min.0', 'HRV_pNN20']].max(axis=1)
        X_engineered['HRV_pNN20_exp_decay'] = np.exp(-np.abs(X['HRV_pNN20']))
        X_engineered['HRV_MeanNN_exp_decay'] = np.exp(-np.abs(X['HRV_MeanNN']))
        X_engineered['HR_5min.0_x_HRV_pNN50'] = X['HR_5min.0'] * X['HRV_pNN50']

        
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
        
        return X_engineered
    
    def advanced_feature_selection(self, X, y, n_features=60):
        """Enhanced feature selection with multiple methods"""
        print(f"Advanced feature selection to {n_features} features...")
        
        # Method 1: Statistical tests with emphasis on class separation
        print("Method 1: Statistical significance with effect size")
        stat_scores = []
        for col in X.columns:
            ptsd_group = X[y == 1][col]
            control_group = X[y == 0][col]
            
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
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(100, X.shape[1]))
        mi_selector.fit(X, y)
        mi_scores = mi_selector.scores_
        
        # Method 3: Tree-based importance with balanced models
        print("Method 3: Balanced tree-based importance")
        
        # Balanced Random Forest
        brf = BalancedRandomForestClassifier(n_estimators=300, random_state=self.random_state)
        brf.fit(X, y)
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
        xgb_model.fit(X, y, sample_weight=xgb_sample_weights)
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
        lr.fit(X, y)
        l1_scores = np.abs(lr.coef_[0])
        
        # Method 5: Relief-based feature selection (approximated with nearest neighbors)
        print("Method 5: Relief-based scoring")
        relief_scores = []
        for i, col in enumerate(X.columns):
            feature_values = X.iloc[:, i].values
            
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
        
        return X[selected_features]
    

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
    

    def split_data(self, save_path=None):
        
        # Load and prepare data
        merged_df, numeric_features = self.load_data()
        
        # Basic preprocessing (missing values already handled in load_data)
        X = merged_df[numeric_features].copy()
        y = merged_df['CAPSF1I2s.0'].copy()
        
        # Remove missing targets (if any remain)
        mask = ~y.isna()
        X, y = X[mask], y[mask]
        
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

        X_train_main = self.apply_power_transformation(X_train_main)
        X_test_main = self.apply_power_transformation(X_test_main)

        train_idx = X_train_main.index
        test_idx = X_test_main.index

        # Combine
        X_all = pd.concat([X_train_main, X_test_main], axis=0)
        y_all = pd.concat([y_train_main, y_test_main], axis=0)

        # Advanced feature engineering
        X_all = self.advanced_feature_engineering(X_all, y_all)
        
        # Advanced feature selection
        X_all = self.advanced_feature_selection(X_all, y_all, n_features=25)

        X_test_main = X_all.loc[test_idx]
        X_train_main = X_all.loc[train_idx]

        y_test_main = y_all.loc[test_idx]
        y_train_main = y_all.loc[train_idx]

        if save_path is not None:
            self.save_splits(X_train_main, X_test_main, y_train_main, y_test_main, save_path)

        return X_test_main, X_train_main, y_test_main, y_train_main

    def save_splits(self, X_train, X_test, y_train, y_test, file_path='pre_deployment_splits.pkl'):
        print(f"Saving splits to {file_path} ...")
        output = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': list(X_train.columns),
            'train_index': X_train.index.tolist(),
            'test_index': X_test.index.tolist(),
        }
        with open(file_path, 'wb') as f:
            pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saved splits.")

if __name__ == "__main__":
    data = PTSDData.with_true_randomness()
    # model = ImprovedPTSDModel.with_random_seed(seed=42)
    
    X_test_main, X_train_main, y_test_main, y_train_main = data.split_data(save_path='pre_deployment_splits.pkl')