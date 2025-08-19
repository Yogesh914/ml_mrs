# Numerical Features Analysis - Pre-deployment Data

## Overview
- **Total numerical features**: 140
- **Total samples**: 2,569 subjects
- **Total missing values**: 1,811 across all numerical columns
- **Value ranges**: Means from -3.964 to 45,639.643; Standard deviations from 0.001 to 685,429.854

## Feature Categories

### 1. Heart Rate Variability (HRV) Features (95 features)
HRV features measure various aspects of heart rate variability and autonomic nervous system function:

#### Time Domain Measures
- **HRV_MeanNN**: Mean of normal-to-normal intervals
- **HRV_SDNN**: Standard deviation of normal-to-normal intervals  
- **HRV_SDANN1**: Standard deviation of averages of NN intervals in all 5-min segments
- **HRV_SDNNI1**: Mean of standard deviations of NN intervals in all 5-min segments
- **HRV_RMSSD**: Root mean square of successive RR interval differences
- **HRV_SDSD**: Standard deviation of successive RR interval differences
- **HRV_pNN50**: Percentage of consecutive RR intervals differing by more than 50ms
- **HRV_pNN20**: Percentage of consecutive RR intervals differing by more than 20ms

#### Frequency Domain Measures
- **HRV_VLF**: Very low frequency power (0.0033-0.04 Hz)
- **HRV_LF**: Low frequency power (0.04-0.15 Hz) - sympathetic activity indicator
- **HRV_HF**: High frequency power (0.15-0.4 Hz) - parasympathetic activity indicator
- **HRV_VHF**: Very high frequency power
- **HRV_TP**: Total power of all frequency bands
- **HRV_LFHF**: LF/HF ratio - sympathetic/parasympathetic balance
- **HRV_LFn**: Normalized low frequency power
- **HRV_HFn**: Normalized high frequency power

#### Geometric Measures
- **HRV_HTI**: HRV triangular index
- **HRV_TINN**: Triangular interpolation of NN interval histogram

#### Poincaré Plot Measures
- **HRV_SD1**: Standard deviation perpendicular to line of identity (short-term variability)
- **HRV_SD2**: Standard deviation along line of identity (long-term variability)
- **HRV_SD1SD2**: Ratio of SD1 to SD2

#### Nonlinear Measures
- **HRV_DFA_alpha1**: Detrended fluctuation analysis short-term scaling exponent
- **HRV_DFA_alpha2**: Detrended fluctuation analysis long-term scaling exponent
- **HRV_ApEn**: Approximate entropy
- **HRV_SampEn**: Sample entropy
- **HRV_ShanEn**: Shannon entropy
- **HRV_FuzzyEn**: Fuzzy entropy
- **HRV_MSEn**: Multiscale entropy
- **HRV_CMSEn**: Composite multiscale entropy
- **HRV_RCMSEn**: Refined composite multiscale entropy

#### Fractal Measures
- **HRV_CD**: Correlation dimension
- **HRV_HFD**: Higuchi fractal dimension
- **HRV_KFD**: Katz fractal dimension
- **HRV_LZC**: Lempel-Ziv complexity

#### Cardiovascular Indices
- **HRV_CSI**: Cardiac sympathetic index
- **HRV_CVI**: Cardiac vagal index
- **HRV_CSI_Modified**: Modified cardiac sympathetic index
- **HRV_PIP**: Poincaré plot parameters
- **HRV_IALS**: Index of autonomic modulation
- **HRV_PSS**: Poincaré stress score
- **HRV_PAS**: Poincaré autonomic score

### 2. Electromyography (EMG) Features (16 features)
EMG features measure muscle activity and startle response:

- **base**: Baseline EMG activity
- **pospic**: EMG response to positive pictures
- **negpic**: EMG response to negative pictures  
- **posant**: EMG response to positive anticipation
- **negant**: EMG response to negative anticipation
- **habituation**: EMG habituation response
- **p80, p85, p90, p95, p100, p105**: EMG responses at different stimulus intensities (dB)
- **p114_t1**: EMG response at 114dB trial 1
- **isi30, isi60, isi120**: Inter-stimulus intervals (30ms, 60ms, 120ms)

### 3. Clinical/Physiological (CUT) Features (13 features)
Clinical and physiological biomarkers:

- **age**: Subject age (18.23-47.75 years range)
- **pCRP**: Plasma C-reactive protein (inflammation marker)
- **pNPY**: Plasma neuropeptide Y (stress hormone)
- **uE**: Urinary epinephrine (stress hormone)
- **uNE**: Urinary norepinephrine (stress hormone)
- **sCortisol**: Salivary cortisol (stress hormone)
- **sAlpha_Amylase**: Salivary alpha-amylase (stress marker)
- **sCotinine**: Salivary cotinine (nicotine exposure)
- **systolic_use**: Systolic blood pressure
- **diastolic_use**: Diastolic blood pressure
- **BMI**: Body mass index
- **waist**: Waist circumference
- **MAP**: Mean arterial pressure

### 4. SPSS-derived Features (16 features)
Additional HRV measures calculated using SPSS:

- **HR_5min.0**: 5-minute heart rate
- **HRnooutliers.0**: Heart rate with outliers removed
- **VLF.0, LF.0, HF.0**: Frequency domain measures (SPSS version)
- **lnVLF.0, lnHF.0, lnLFHF.0**: Natural log-transformed frequency measures
- **RMSSD.0, SDNN.0**: Time domain measures (SPSS version)
- **lnRMSSD.0, lnSDNN.0**: Natural log-transformed time domain measures
- **Sympathetic.0**: Sympathetic activity measure
- **Vagal.0**: Vagal (parasympathetic) activity measure
- **Sympathetic_vagalbalance.0**: Sympathetic-vagal balance
- **LFHF.0**: LF/HF ratio (SPSS version)

## Additional HRV-related Features (from main list)
- **VLF, LF, HF**: Frequency domain power measures
- **LFHF**: LF/HF ratio
- **TOTAL**: Total power
- **HR**: Heart rate
- **SAI_idx, PAI_idx, BAI_idx**: Autonomic indices

## Data Quality Notes
- Dataset contains 2,569 subjects with 224 total columns
- Numerical features show wide variability in scales and distributions
- Some features have very large standard deviations indicating potential outliers
- Missing values are present across numerical columns
- Features span multiple physiological systems: cardiovascular, autonomic nervous system, stress response, and muscle activity

## Feature Usage Context
This appears to be a comprehensive physiological dataset likely used for:
- Stress and PTSD research (given EMG startle response measures)
- Cardiovascular health assessment
- Autonomic nervous system function analysis
- Biomarker-based health prediction modeling