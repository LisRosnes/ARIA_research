import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu

def summarize_features(csv_path: str):
    # Read CSV
    df = pd.read_csv(csv_path)

    # Clean up: treat pure whitespace as blank
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    total_rows = len(df)
    print(f"Total rows: {total_rows}\n")

    summary_rows = []

    for col in df.columns:
        # Consider a cell "blank" if it is NaN or an empty string
        non_blank_mask = df[col].notna() & (df[col] != "")
        non_blank_count = non_blank_mask.sum()
        blank_count = total_rows - non_blank_count

        summary_rows.append({
            "feature": col,
            "non_blank": non_blank_count,
            "blank": blank_count
        })

    # Convert summary to DataFrame for nice display (and easy export if you want)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    return df


def standardize_binary(series):
    """Convert yes/no/present/absent/none to binary 1/0, return as numeric."""
    def convert(val):
        if pd.isna(val) or val == "":
            return np.nan
        v = str(val).lower().strip()
        if v in ["yes", "present", "positive", "1"]:
            return 1
        elif v in ["no", "none", "absent", "negative", "0"]:
            return 0
        else:
            return np.nan
    return series.apply(convert)


def standardize_ordinal_grade(series):
    """Convert mild/moderate/severe/normal/none to numeric 0-3."""
    def convert(val):
        if pd.isna(val) or val == "":
            return np.nan
        v = str(val).lower().strip()
        if v in ["normal", "none", "no"]:
            return 0
        elif v in ["mild", "1"]:
            return 1
        elif v in ["moderate", "2"]:
            return 2
        elif v in ["severe", "3"]:
            return 3
        else:
            return np.nan
    return series.apply(convert)


def standardize_apoe(series):
    """Convert APOE to carrier (1) vs non-carrier (0)."""
    def convert(val):
        if pd.isna(val) or val == "":
            return np.nan
        v = str(val).lower().strip().replace(" ", "").replace("-", "")
        if "noncarrier" in v or v == "negative":
            return 0
        elif "heterozygous" in v or "homozygous" in v or "positive" in v or "carrier" in v:
            return 1
        else:
            return np.nan
    return series.apply(convert)


def standardize_lecanemab_dose(series):
    """Extract numeric dose from text like '1st', '2nd', 'prior to first dose' -> 0."""
    def convert(val):
        if pd.isna(val) or val == "":
            return np.nan
        v = str(val).lower().strip()
        if "prior" in v:
            return 0
        # Extract number from ordinals: 1st, 2nd, 3rd, etc.
        import re
        m = re.search(r'(\d+)', v)
        if m:
            return int(m.group(1))
        return np.nan
    return series.apply(convert)


def perform_chi2_or_fisher(contingency_table):
    """
    Perform Chi-square test if expected frequencies > 5, otherwise Fisher's exact.
    Returns test name and p-value.
    """
    # Check if any expected frequency is < 5
    try:
        chi2, p_chi, dof, expected = chi2_contingency(contingency_table)
        if (expected < 5).any():
            # Use Fisher's exact for 2x2 tables
            if contingency_table.shape == (2, 2):
                odds_ratio, p_fisher = fisher_exact(contingency_table)
                return "Fisher's exact", p_fisher
            else:
                # For larger tables with small expected frequencies, still use chi2 but note limitation
                return "Chi-square (warning: low expected freq)", p_chi
        else:
            return "Chi-square", p_chi
    except Exception as e:
        return "Error", np.nan


def analyze_univariate(df):
    """
    Perform univariate tests for features vs ARIA-E and ARIA-H.
    """
    print("\n" + "="*80)
    print("UNIVARIATE STATISTICAL ANALYSIS")
    print("="*80 + "\n")
    
    # Standardize outcomes
    df['ARIA_E_binary'] = standardize_binary(df['ARIA-E'])
    df['ARIA_H_binary'] = standardize_binary(df['ARIA-H'])
    
    # Features to analyze
    features_config = {
        'Apoe': ('categorical', standardize_apoe),
        'Lecanemab dose': ('ordinal', standardize_lecanemab_dose),
        'acute hemmorage': ('binary', standardize_binary),
        'Acute/subacute cortical infarct': ('binary', standardize_binary),
        'Chronic hemorrhage': ('binary', standardize_binary),
        'Chronic ischemic changes': ('binary', standardize_binary),
        'MTA-right': ('ordinal', standardize_ordinal_grade),
        'MTA-left': ('ordinal', standardize_ordinal_grade),
        'ERiCA-right': ('ordinal', standardize_ordinal_grade),
        'ERiCA-left': ('ordinal', standardize_ordinal_grade),
    }
    
    results = []
    
    for outcome in ['ARIA_E_binary', 'ARIA_H_binary']:
        outcome_label = 'ARIA-E' if outcome == 'ARIA_E_binary' else 'ARIA-H'
        print(f"\n{'='*80}")
        print(f"OUTCOME: {outcome_label}")
        print('='*80)
        
        # Filter to rows with valid outcome
        df_outcome = df[df[outcome].notna()].copy()
        n_outcome = len(df_outcome)
        n_positive = (df_outcome[outcome] == 1).sum()
        n_negative = (df_outcome[outcome] == 0).sum()
        
        print(f"\nSample size: {n_outcome} ({n_positive} positive, {n_negative} negative)")
        print("\n" + "-"*80 + "\n")
        
        for feature_name, (feature_type, standardize_func) in features_config.items():
            print(f"Feature: {feature_name} ({feature_type})")
            
            # Standardize feature
            df_outcome[f'{feature_name}_std'] = standardize_func(df_outcome[feature_name])
            
            # Filter to valid feature values
            valid_mask = df_outcome[f'{feature_name}_std'].notna()
            df_valid = df_outcome[valid_mask]
            n_valid = len(df_valid)
            
            if n_valid < 5:
                print(f"  ⚠ Insufficient data (n={n_valid})")
                print()
                results.append({
                    'outcome': outcome_label,
                    'feature': feature_name,
                    'n': n_valid,
                    'test': 'N/A',
                    'p_value': np.nan,
                    'note': 'Insufficient data'
                })
                continue
            
            # Create contingency table or perform appropriate test
            if feature_type in ['binary', 'categorical']:
                # Chi-square or Fisher's exact
                contingency = pd.crosstab(df_valid[f'{feature_name}_std'], df_valid[outcome])
                print(f"  Contingency table:")
                print(f"  {contingency}")
                
                test_name, p_value = perform_chi2_or_fisher(contingency.values)
                print(f"  Test: {test_name}")
                print(f"  p-value: {p_value:.4f}" if not np.isnan(p_value) else "  p-value: N/A")
                
                results.append({
                    'outcome': outcome_label,
                    'feature': feature_name,
                    'n': n_valid,
                    'test': test_name,
                    'p_value': p_value,
                    'note': ''
                })
                
            elif feature_type == 'ordinal':
                # Mann-Whitney U test (non-parametric test for ordinal data)
                group_0 = df_valid[df_valid[outcome] == 0][f'{feature_name}_std']
                group_1 = df_valid[df_valid[outcome] == 1][f'{feature_name}_std']
                
                print(f"  {outcome_label}=No: n={len(group_0)}, median={group_0.median():.1f}, mean={group_0.mean():.2f}")
                print(f"  {outcome_label}=Yes: n={len(group_1)}, median={group_1.median():.1f}, mean={group_1.mean():.2f}")
                
                if len(group_0) >= 3 and len(group_1) >= 3:
                    stat, p_value = mannwhitneyu(group_0, group_1, alternative='two-sided')
                    print(f"  Test: Mann-Whitney U")
                    print(f"  p-value: {p_value:.4f}")
                    
                    results.append({
                        'outcome': outcome_label,
                        'feature': feature_name,
                        'n': n_valid,
                        'test': 'Mann-Whitney U',
                        'p_value': p_value,
                        'note': ''
                    })
                else:
                    print(f"  ⚠ Insufficient data in one or both groups")
                    results.append({
                        'outcome': outcome_label,
                        'feature': feature_name,
                        'n': n_valid,
                        'test': 'N/A',
                        'p_value': np.nan,
                        'note': 'Insufficient group size'
                    })
            
            print()
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80 + "\n")
    
    results_df = pd.DataFrame(results)
    results_df['significant'] = results_df['p_value'] < 0.05
    
    # Sort by p-value
    results_df = results_df.sort_values('p_value')
    
    print(results_df.to_string(index=False))
    
    # Save to CSV
    results_df.to_csv('univariate_results.csv', index=False)
    print("\n✓ Results saved to 'univariate_results.csv'")
    
    return results_df

def create_visualizations(df, results_df, output_dir='univariate_plots'):
    """
    Create visualizations for statistically significant results.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    
    # Get significant results (p < 0.05)
    sig_results = results_df[results_df['significant'] == True].copy()
    
    if len(sig_results) == 0:
        print("\n⚠ No significant results to visualize")
        return
    
    print("\n" + "="*80)
    print(f"CREATING VISUALIZATIONS FOR {len(sig_results)} SIGNIFICANT RESULTS")
    print("="*80 + "\n")
    
    # Re-standardize features for plotting
    df['ARIA_E_binary'] = standardize_binary(df['ARIA-E'])
    df['ARIA_H_binary'] = standardize_binary(df['ARIA-H'])
    
    features_config = {
        'Apoe': ('categorical', standardize_apoe),
        'Lecanemab dose': ('ordinal', standardize_lecanemab_dose),
        'acute hemmorage': ('binary', standardize_binary),
        'Acute/subacute cortical infarct': ('binary', standardize_binary),
        'Chronic hemorrhage': ('binary', standardize_binary),
        'Chronic ischemic changes': ('binary', standardize_binary),
        'MTA-right': ('ordinal', standardize_ordinal_grade),
        'MTA-left': ('ordinal', standardize_ordinal_grade),
        'ERiCA-right': ('ordinal', standardize_ordinal_grade),
        'ERiCA-left': ('ordinal', standardize_ordinal_grade),
    }
    
    for idx, row in sig_results.iterrows():
        outcome_label = row['outcome']
        feature_name = row['feature']
        p_value = row['p_value']
        
        outcome_col = 'ARIA_E_binary' if outcome_label == 'ARIA-E' else 'ARIA_H_binary'
        
        print(f"Creating plot: {feature_name} vs {outcome_label} (p={p_value:.4f})")
        
        # Get feature type and standardization function
        feature_type, standardize_func = features_config.get(feature_name, (None, None))
        if feature_type is None:
            continue
        
        # Standardize feature
        df[f'{feature_name}_std'] = standardize_func(df[feature_name])
        
        # Filter to valid data
        plot_df = df[[outcome_col, f'{feature_name}_std']].dropna()
        plot_df['Outcome'] = plot_df[outcome_col].map({0: 'No', 1: 'Yes'})
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # LEFT PANEL: Main visualization
        if feature_type in ['ordinal']:
            # Box plot for ordinal data
            sns.boxplot(data=plot_df, x='Outcome', y=f'{feature_name}_std', 
                       palette=['lightblue', 'salmon'], ax=axes[0])
            sns.stripplot(data=plot_df, x='Outcome', y=f'{feature_name}_std',
                         color='black', alpha=0.3, size=3, ax=axes[0])
            axes[0].set_ylabel(f'{feature_name} (grade)', fontsize=12)
            
            # Add means
            means = plot_df.groupby('Outcome')[f'{feature_name}_std'].mean()
            for i, (outcome, mean_val) in enumerate(means.items()):
                axes[0].plot([i-0.2, i+0.2], [mean_val, mean_val], 
                           'r-', linewidth=2, label='Mean' if i == 0 else '')
            
        elif feature_type in ['binary', 'categorical']:
            # Stacked bar chart for categorical data
            contingency = pd.crosstab(plot_df[f'{feature_name}_std'], 
                                     plot_df['Outcome'], normalize='index') * 100
            contingency.plot(kind='bar', stacked=False, 
                           color=['lightblue', 'salmon'], ax=axes[0])
            axes[0].set_ylabel('Percentage (%)', fontsize=12)
            axes[0].set_xlabel(f'{feature_name}', fontsize=12)
            axes[0].legend(title=outcome_label, labels=['No', 'Yes'])
            axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
        
        axes[0].set_xlabel(f'{outcome_label}', fontsize=12)
        axes[0].set_title(f'{feature_name} vs {outcome_label}\n(p = {p_value:.4f})', 
                         fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # RIGHT PANEL: Count distribution
        count_data = plot_df.groupby(['Outcome', f'{feature_name}_std']).size().reset_index(name='count')
        
        if feature_type in ['ordinal']:
            # Grouped bar chart
            pivot_counts = count_data.pivot(index=f'{feature_name}_std', 
                                           columns='Outcome', values='count').fillna(0)
            pivot_counts.plot(kind='bar', color=['lightblue', 'salmon'], ax=axes[1])
            axes[1].set_ylabel('Count', fontsize=12)
            axes[1].set_xlabel(f'{feature_name} (grade)', fontsize=12)
            axes[1].legend(title=outcome_label, labels=['No', 'Yes'])
        else:
            # Simple bar chart
            sns.countplot(data=plot_df, x=f'{feature_name}_std', hue='Outcome',
                         palette=['lightblue', 'salmon'], ax=axes[1])
            axes[1].set_ylabel('Count', fontsize=12)
            axes[1].set_xlabel(f'{feature_name}', fontsize=12)
            axes[1].legend(title=outcome_label, labels=['No', 'Yes'])
        
        axes[1].set_title('Sample Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        safe_filename = f"{feature_name.replace('/', '_').replace(' ', '_')}_vs_{outcome_label}.png"
        filepath = os.path.join(output_dir, safe_filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {filepath}")
    
    print(f"\n✓ All visualizations saved to '{output_dir}/' directory")


if __name__ == "__main__":
    # Change this to your actual file name
    csv_file = "output_features_2.csv"
    
    print("="*80)
    print("FEATURE SUMMARY")
    print("="*80)
    df = summarize_features(csv_file)
    
    # Run univariate analysis
    results = analyze_univariate(df)
    
    # Create visualizations for significant results
    create_visualizations(df, results)