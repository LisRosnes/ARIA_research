"""
ARIA Prediction Pipeline using Elastic Net Logistic Regression
Handles small sample sizes and incremental data accumulation
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                            roc_curve, precision_recall_curve,
                            confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Minimum events threshold for modeling
MIN_EVENTS_FOR_MODELING = 10
MIN_EVENTS_FOR_CV = 15

# Feature prefixes to include (FreeSurfer outputs)
FEATURE_PREFIXES = ['aseg_', 'lh_', 'rh_']

# Features to exclude (identifiers, dates, non-numeric)
EXCLUDE_FEATURES = ['subject_id', 'date', 'session_date', 'timepoints', 
                   'full_name', 'SubjectID']

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_and_merge_data(freesurfer_path, aria_path):
    """
    Load FreeSurfer and ARIA data and merge them
    Uses earliest (baseline) scan per subject to predict future ARIA outcomes
    
    Parameters:
    -----------
    freesurfer_path : str
        Path to FreeSurfer CSV file
    aria_path : str
        Path to ARIA outcomes CSV file
    
    Returns:
    --------
    merged_df : pd.DataFrame
        Merged dataset with outcomes (one row per subject)
    """
    # Load data
    fs_df = pd.read_csv(freesurfer_path)
    aria_df = pd.read_csv(aria_path)
    
    print(f"FreeSurfer data: {len(fs_df)} total scans from {fs_df['subject_id'].nunique()} subjects")
    
    # Clean subject IDs (handle potential formatting differences)
    fs_df['subject_id'] = fs_df['subject_id'].astype(str).str.strip()
    aria_df['SubjectID'] = aria_df['SubjectID'].astype(str).str.strip()
    
    # Convert date column to datetime for sorting
    date_col = 'date (YYYY_MM_DD)' if 'date (YYYY_MM_DD)' in fs_df.columns else 'date'
    if date_col in fs_df.columns:
        fs_df[date_col] = pd.to_datetime(fs_df[date_col].astype(str).str.replace('_', '-'), 
                                         errors='coerce')
        
        # Sort by subject and date, then take earliest scan per subject
        fs_df = fs_df.sort_values(['subject_id', date_col])
        fs_df_baseline = fs_df.groupby('subject_id').first().reset_index()
        
        print(f"After selecting earliest scan per subject: {len(fs_df_baseline)} subjects")
        print(f"Dropped {len(fs_df) - len(fs_df_baseline)} duplicate timepoint scans")
    else:
        # No date column - just take first occurrence per subject
        fs_df_baseline = fs_df.groupby('subject_id').first().reset_index()
        print(f"No date column found - using first occurrence per subject: {len(fs_df_baseline)} subjects")
    
    # Merge on subject ID
    merged_df = fs_df_baseline.merge(aria_df, 
                                     left_on='subject_id', 
                                     right_on='SubjectID', 
                                     how='inner')
    
    print(f"\nAfter merging with ARIA outcomes:")
    print(f"  Total subjects: {len(merged_df)}")
    print(f"  ARIA-E cases: {merged_df['ARIA-E'].sum()} subjects")
    print(f"  ARIA-H cases: {merged_df['ARIA-H'].sum()} subjects")
    
    return merged_df

def prepare_features(df):
    """
    Extract and clean features for modeling
    
    Parameters:
    -----------
    df : pd.DataFrame
        Merged dataframe
    
    Returns:
    --------
    X : pd.DataFrame
        Feature matrix
    feature_names : list
        List of feature names
    """
    # Select features based on prefixes
    feature_cols = [col for col in df.columns 
                   if any(col.startswith(prefix) for prefix in FEATURE_PREFIXES)
                   and col not in EXCLUDE_FEATURES]
    
    X = df[feature_cols].copy()
    
    # Handle missing values
    missing_pct = X.isnull().sum() / len(X) * 100
    high_missing = missing_pct[missing_pct > 50].index.tolist()
    
    if high_missing:
        print(f"\nRemoving {len(high_missing)} features with >50% missing data")
        X = X.drop(columns=high_missing)
    
    # Impute remaining missing with median
    for col in X.columns:
        if X[col].isnull().any():
            X[col].fillna(X[col].median(), inplace=True)
    
    # Remove zero-variance features
    zero_var = X.columns[X.std() == 0].tolist()
    if zero_var:
        print(f"Removing {len(zero_var)} zero-variance features")
        X = X.drop(columns=zero_var)
    
    print(f"\nFinal feature count: {X.shape[1]}")
    
    return X, X.columns.tolist()

# ============================================================================
# DESCRIPTIVE ANALYSIS (for small sample sizes)
# ============================================================================

def descriptive_analysis(X, y, outcome_name):
    """
    Perform descriptive statistics and univariate associations
    Useful when sample size is too small for modeling
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Outcome variable
    outcome_name : str
        Name of outcome (e.g., 'ARIA-E')
    """
    print(f"\n{'='*80}")
    print(f"DESCRIPTIVE ANALYSIS: {outcome_name}")
    print(f"{'='*80}")
    
    n_cases = y.sum()
    n_controls = len(y) - n_cases
    
    print(f"\nSample composition:")
    print(f"  Cases: {n_cases}")
    print(f"  Controls: {n_controls}")
    print(f"  Total: {len(y)}")
    print(f"  Prevalence: {n_cases/len(y)*100:.2f}%")
    
    # Univariate associations
    print(f"\n{'='*80}")
    print("TOP 20 UNIVARIATE ASSOCIATIONS (Mann-Whitney U Test)")
    print(f"{'='*80}")
    
    # Check if we have any cases
    if n_cases == 0:
        print("\n⚠️  Cannot perform univariate associations - no cases in the dataset")
        print("   All subjects are controls (negative for outcome)")
        return None
    
    results = []
    for col in X.columns:
        cases = X.loc[y == 1, col]
        controls = X.loc[y == 0, col]
        
        # Mann-Whitney U test (non-parametric)
        statistic, pval = stats.mannwhitneyu(cases, controls, alternative='two-sided')
        
        # Effect size (rank-biserial correlation)
        effect_size = 1 - (2*statistic) / (len(cases) * len(controls))
        
        results.append({
            'feature': col,
            'case_median': cases.median(),
            'control_median': controls.median(),
            'p_value': pval,
            'effect_size': effect_size
        })
    
    results_df = pd.DataFrame(results).sort_values('p_value')
    
    print(results_df.head(20).to_string(index=False))
    
    # Save full results
    results_df.to_csv(f'univariate_associations_{outcome_name}.csv', index=False)
    print(f"\nFull results saved to: univariate_associations_{outcome_name}.csv")
    
    return results_df

# ============================================================================
# ELASTIC NET MODEL WITH CROSS-VALIDATION
# ============================================================================

def check_modeling_feasibility(y, outcome_name):
    """
    Check if we have enough events for modeling
    
    Returns:
    --------
    can_model : bool
        Whether modeling is feasible
    can_cv : bool
        Whether cross-validation is feasible
    """
    n_events = y.sum()
    
    print(f"\n{'='*80}")
    print(f"MODELING FEASIBILITY CHECK: {outcome_name}")
    print(f"{'='*80}")
    print(f"Number of events: {n_events}")
    
    can_model = n_events >= MIN_EVENTS_FOR_MODELING
    can_cv = n_events >= MIN_EVENTS_FOR_CV
    
    if not can_model:
        print(f"\n⚠️  WARNING: Insufficient events for modeling!")
        print(f"   Need at least {MIN_EVENTS_FOR_MODELING} events")
        print(f"   Currently have: {n_events}")
        print(f"\n   RECOMMENDATION: Wait for more data or perform descriptive analysis only")
    elif not can_cv:
        print(f"\n⚠️  WARNING: Limited events for cross-validation!")
        print(f"   Have {n_events} events (minimum {MIN_EVENTS_FOR_CV} recommended for CV)")
        print(f"\n   RECOMMENDATION: Use simple train/test split or reduced CV folds")
    else:
        print(f"\n✓ Sufficient events for full modeling with cross-validation")
    
    return can_model, can_cv

def train_elastic_net(X, y, outcome_name, use_cv=True):
    """
    Train elastic net logistic regression with hyperparameter tuning
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix (unstandardized)
    y : pd.Series
        Binary outcome
    outcome_name : str
        Name of outcome
    use_cv : bool
        Whether to use cross-validation
    
    Returns:
    --------
    results : dict
        Dictionary containing model, scaler, and performance metrics
    """
    print(f"\n{'='*80}")
    print(f"ELASTIC NET MODELING: {outcome_name}")
    print(f"{'='*80}")
    
    # Calculate class weights
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, class_weights))
    
    print(f"\nClass weights: {class_weight_dict}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # Hyperparameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]  # Elastic net mixing parameter
    }
    
    # Base model
    base_model = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        class_weight=class_weight_dict,
        max_iter=10000,
        random_state=42
    )
    
    if use_cv and len(y[y==1]) >= MIN_EVENTS_FOR_CV:
        # Cross-validated hyperparameter search
        n_splits = min(5, len(y[y==1]))  # Adapt folds to sample size
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        print(f"\nPerforming {n_splits}-fold cross-validated grid search...")
        print(f"Testing {len(param_grid['C']) * len(param_grid['l1_ratio'])} parameter combinations")
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_scaled, y)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV AUROC: {grid_search.best_score_:.4f}")
        
        model = grid_search.best_estimator_
        cv_scores = grid_search.cv_results_
        
    else:
        # Simple model without CV (use default mid-range parameters)
        print("\nTraining single model without cross-validation...")
        model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            C=1.0,
            l1_ratio=0.5,
            class_weight=class_weight_dict,
            max_iter=10000,
            random_state=42
        )
        model.fit(X_scaled, y)
        cv_scores = None
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = model.predict(X_scaled)
    
    # Calculate metrics
    auroc = roc_auc_score(y, y_pred_proba)
    auprc = average_precision_score(y, y_pred_proba)
    
    print(f"\n{'='*60}")
    print("MODEL PERFORMANCE (Training Data)")
    print(f"{'='*60}")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred, 
                                target_names=['No ARIA', 'ARIA'],
                                zero_division=0))
    
    # Feature importance (non-zero coefficients)
    coef_df = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    
    non_zero_features = coef_df[coef_df['coefficient'] != 0]
    
    print(f"\n{'='*60}")
    print(f"FEATURE IMPORTANCE ({len(non_zero_features)} non-zero features)")
    print(f"{'='*60}")
    print(non_zero_features.head(20).to_string(index=False))
    
    # Save feature importance
    non_zero_features.to_csv(f'feature_importance_{outcome_name}.csv', index=False)
    print(f"\nFull feature importance saved to: feature_importance_{outcome_name}.csv")
    
    # Package results
    results = {
        'model': model,
        'scaler': scaler,
        'auroc': auroc,
        'auprc': auprc,
        'y_true': y,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
        'features': X.columns.tolist(),
        'feature_importance': non_zero_features,
        'cv_scores': cv_scores,
        'class_weights': class_weight_dict
    }
    
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_roc_curve(results, outcome_name, save_path=None):
    """
    Plot ROC curve
    
    Parameters:
    -----------
    results : dict
        Results dictionary from train_elastic_net
    outcome_name : str
        Name of outcome
    save_path : str, optional
        Path to save figure
    """
    fpr, tpr, thresholds = roc_curve(results['y_true'], results['y_pred_proba'])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUROC = {results['auroc']:.3f}")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve: {outcome_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    plt.close()

def plot_precision_recall_curve(results, outcome_name, save_path=None):
    """
    Plot Precision-Recall curve
    
    Parameters:
    -----------
    results : dict
        Results dictionary from train_elastic_net
    outcome_name : str
        Name of outcome
    save_path : str, optional
        Path to save figure
    """
    precision, recall, thresholds = precision_recall_curve(
        results['y_true'], results['y_pred_proba']
    )
    
    baseline = results['y_true'].sum() / len(results['y_true'])
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f"AUPRC = {results['auprc']:.3f}")
    plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                label=f'Baseline (prevalence = {baseline:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve: {outcome_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to: {save_path}")
    plt.close()

def plot_feature_importance(results, outcome_name, top_n=20, save_path=None):
    """
    Plot top feature importances
    
    Parameters:
    -----------
    results : dict
        Results dictionary from train_elastic_net
    outcome_name : str
        Name of outcome
    top_n : int
        Number of top features to plot
    save_path : str, optional
        Path to save figure
    """
    feat_imp = results['feature_importance'].head(top_n).copy()
    
    # Create horizontal bar plot
    plt.figure(figsize=(10, max(6, top_n * 0.3)))
    colors = ['red' if x < 0 else 'blue' for x in feat_imp['coefficient']]
    
    plt.barh(range(len(feat_imp)), feat_imp['coefficient'], color=colors, alpha=0.7)
    plt.yticks(range(len(feat_imp)), feat_imp['feature'], fontsize=10)
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.title(f'Top {top_n} Features: {outcome_name}', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")
    plt.close()

def plot_confusion_matrix(results, outcome_name, save_path=None):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    results : dict
        Results dictionary from train_elastic_net
    outcome_name : str
        Name of outcome
    save_path : str, optional
        Path to save figure
    """
    cm = confusion_matrix(results['y_true'], results['y_pred'])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No ARIA', 'ARIA'],
                yticklabels=['No ARIA', 'ARIA'],
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(f'Confusion Matrix: {outcome_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    plt.close()

def create_all_plots(results, outcome_name, output_dir='multivariate_plots'):
    """
    Create all visualization plots for a model
    
    Parameters:
    -----------
    results : dict
        Results dictionary from train_elastic_net
    outcome_name : str
        Name of outcome
    output_dir : str
        Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"CREATING VISUALIZATIONS: {outcome_name}")
    print(f"{'='*80}")
    
    plot_roc_curve(results, outcome_name, 
                   save_path=f'{output_dir}/roc_curve_{outcome_name}.png')
    plot_precision_recall_curve(results, outcome_name,
                                save_path=f'{output_dir}/pr_curve_{outcome_name}.png')
    plot_feature_importance(results, outcome_name,
                           save_path=f'{output_dir}/feature_importance_{outcome_name}.png')
    plot_confusion_matrix(results, outcome_name,
                         save_path=f'{output_dir}/confusion_matrix_{outcome_name}.png')
    
    print(f"\nAll plots saved to: {output_dir}/")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_analysis_for_outcome(df, outcome_col, feature_cols):
    """
    Run complete analysis pipeline for a single outcome
    
    Parameters:
    -----------
    df : pd.DataFrame
        Merged dataframe with features and outcomes
    outcome_col : str
        Column name for outcome (e.g., 'ARIA-E' or 'ARIA-H')
    feature_cols : list
        List of feature column names
    
    Returns:
    --------
    results : dict or None
        Model results if modeling was performed, None otherwise
    """
    print(f"\n{'#'*80}")
    print(f"# ANALYSIS FOR: {outcome_col}")
    print(f"{'#'*80}\n")
    
    # Prepare outcome
    y = df[outcome_col].copy()
    X = df[feature_cols].copy()
    
    # Check feasibility
    can_model, can_cv = check_modeling_feasibility(y, outcome_col)
    
    if not can_model:
        # Only descriptive analysis
        print(f"\n⚠️  Performing descriptive analysis only for {outcome_col}")
        descriptive_analysis(X, y, outcome_col)
        return None
    
    else:
        # Full modeling pipeline
        results = train_elastic_net(X, y, outcome_col, use_cv=can_cv)
        create_all_plots(results, outcome_col)
        
        # Also save descriptive stats
        descriptive_analysis(X, y, outcome_col)
        
        return results

def main():
    """
    Main execution pipeline
    """
    print("="*80)
    print("ARIA PREDICTION: MULTIVARIATE ANALYSIS PIPELINE")
    print("="*80)
    print("\nThis pipeline performs elastic net logistic regression to predict")
    print("ARIA-E and ARIA-H outcomes from FreeSurfer brain MRI features.\n")
    
    # ========================================================================
    # STEP 1: Load and merge data
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING AND MERGING")
    print("="*80)
    
    FREESURFER_PATH = 'merged_stats_full_tp.csv'
    ARIA_PATH = 'aria_mapping_output.csv'
    
    df = load_and_merge_data(FREESURFER_PATH, ARIA_PATH)
    
    # ========================================================================
    # STEP 2: Prepare features
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: FEATURE PREPARATION")
    print("="*80)
    
    X, feature_names = prepare_features(df)
    
    # ========================================================================
    # STEP 3: Analyze ARIA-E
    # ========================================================================
    results_aria_e = run_analysis_for_outcome(
        df=df,
        outcome_col='ARIA-E',
        feature_cols=feature_names
    )
    
    # ========================================================================
    # STEP 4: Analyze ARIA-H
    # ========================================================================
    results_aria_h = run_analysis_for_outcome(
        df=df,
        outcome_col='ARIA-H',
        feature_cols=feature_names
    )
    
    # ========================================================================
    # STEP 5: Summary
    # ========================================================================
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    print("\nGenerated outputs:")
    print("  - Feature importance CSVs for each outcome")
    print("  - Univariate association CSVs for each outcome")
    print("  - Visualization plots in multivariate_plots/")
    
    if results_aria_e:
        print(f"\nARIA-E Model Performance:")
        print(f"  AUROC: {results_aria_e['auroc']:.4f}")
        print(f"  AUPRC: {results_aria_e['auprc']:.4f}")
        print(f"  Non-zero features: {len(results_aria_e['feature_importance'])}")
    
    if results_aria_h:
        print(f"\nARIA-H Model Performance:")
        print(f"  AUROC: {results_aria_h['auroc']:.4f}")
        print(f"  AUPRC: {results_aria_h['auprc']:.4f}")
        print(f"  Non-zero features: {len(results_aria_h['feature_importance'])}")
    
    print("\n" + "="*80)
    print("Thank you for using the ARIA Prediction Pipeline!")
    print("="*80 + "\n")
    
    return {
        'aria_e': results_aria_e,
        'aria_h': results_aria_h
    }

# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    results = main()