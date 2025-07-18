"""
Analysis functions for steel tariffs employment study
Author: [Your Name]
Date: July 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def load_and_clean_data(data_path):
    """Load and clean BLS QCEW data for analysis"""
    all_data = []
    
    # Industry mapping
    industry_mapping = {
        'Iron_steel': {'name': 'Iron and Steel', 'treatment_group': 'treatment'},
        'MotorVehicle': {'name': 'Motor Vehicle', 'treatment_group': 'treatment'},
        'Textile_Mill': {'name': 'Textile Mills', 'treatment_group': 'control'}
    }
    
    # Load and process files
    for filename, industry_info in industry_mapping.items():
        # This would load actual files in a real implementation
        pass
    
    return all_data

def create_treatment_variables(df):
    """Create treatment and time period variables for analysis"""
    df['treated'] = (df['treatment_group'] == 'treatment').astype(int)
    df['biden_era'] = (df['treatment_period'] == 'biden_era').astype(int)
    df['treated_biden'] = df['treated'] * df['biden_era']
    
    return df

def run_did_regression(df, outcome_var):
    """Run difference-in-differences regression with robust standard errors"""
    reg_data = df.dropna(subset=[outcome_var])
    
    # Create design matrix
    X = reg_data[['treated', 'biden_era', 'treated_biden']].copy()
    X = sm.add_constant(X)
    y = reg_data[outcome_var]
    
    # Run regression with robust standard errors
    model = sm.OLS(y, X).fit(cov_type='HC3')
    
    return model

def create_employment_trends_plot(df, save_path=None):
    """Create professional employment trends visualization"""
    plt.figure(figsize=(12, 8))
    
    # Group data by industry and time
    quarterly_trends = df.groupby(['industry_name', 'quarter_date']).agg({
        'avg_monthly_employment': 'mean'
    }).reset_index()
    
    # Plot trends for each industry
    for industry in df['industry_name'].unique():
        industry_data = quarterly_trends[quarterly_trends['industry_name'] == industry]
        plt.plot(industry_data['quarter_date'], industry_data['avg_monthly_employment'], 
                marker='o', linewidth=2, label=industry, markersize=4)
    
    # Add policy markers
    plt.axvline(pd.to_datetime('2018-03-23'), color='red', linestyle='--', 
                alpha=0.7, label='Steel Tariffs Implemented')
    plt.axvline(pd.to_datetime('2021-01-20'), color='orange', linestyle='--', 
                alpha=0.7, label='Biden Administration')
    
    plt.title('Employment Trends by Industry: Steel Tariff Effects', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Quarter', fontsize=12)
    plt.ylabel('Average Monthly Employment per Establishment', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def calculate_economic_effects(model):
    """Calculate economic interpretation of regression results"""
    results = {}
    
    if 'treated' in model.params.index:
        treated_coef = model.params['treated']
        results['treatment_effect_pct'] = (np.exp(treated_coef) - 1) * 100
        results['treatment_pvalue'] = model.pvalues['treated']
        results['treatment_significant'] = model.pvalues['treated'] < 0.05
    
    if 'treated_biden' in model.params.index:
        biden_coef = model.params['treated_biden']
        results['biden_interaction_pct'] = (np.exp(biden_coef) - 1) * 100
        results['biden_interaction_pvalue'] = model.pvalues['treated_biden']
        results['biden_interaction_significant'] = model.pvalues['treated_biden'] < 0.05
    
    return results

def create_summary_statistics(df):
    """Create comprehensive summary statistics table"""
    summary = df.groupby(['industry_name', 'treatment_period']).agg({
        'avg_monthly_employment': ['count', 'mean', 'std', 'median'],
        'total_qtrly_wages': ['mean', 'std'],
        'avg_wkly_wage': ['mean', 'std']
    }).round(2)
    
    return summary
