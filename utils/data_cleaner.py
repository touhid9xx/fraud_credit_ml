"""
Handles missing values, duplicates, outlier detection, and data quality assessment
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Any, Tuple

warnings.filterwarnings('ignore')


class DataCleaner:
    
    
    def __init__(self, df: pd.DataFrame, table_name: str = 'unknown'):
       
        self.df = df.copy()
        self.table_name = table_name
        self.cleaning_report = {}
        self.original_shape = df.shape
        
    def __repr__(self) -> str:
        
        quality_score = self.get_quality_score()
        score = quality_score['score'] if quality_score else 'N/A'
        grade = quality_score['grade'] if quality_score else 'N/A'
        return f"DataCleaner(table='{self.table_name}', shape={self.df.shape}, quality_score={score} ({grade}))"
    
    # ============================================================
    # MISSING VALUES HANDLING
    # ============================================================
    
    def detect_missing_values(self) -> pd.DataFrame:
       
        missing_counts = self.df.isnull().sum()
        missing_percentages = (missing_counts / len(self.df)) * 100
        
        missing_report = pd.DataFrame({
            'Column': missing_counts.index,
            'Missing_Count': missing_counts.values,
            'Missing_Percentage': missing_percentages.values
        })
        
        missing_report = missing_report[missing_report['Missing_Count'] > 0]
        missing_report = missing_report.sort_values('Missing_Percentage', ascending=False)
        
        self.cleaning_report['missing_values'] = missing_report
        
        return missing_report
    
    def handle_missing_values(
        self,
        strategy: str = 'auto',
        fill_value: Any = None,
        columns: Optional[List[str]] = None,
        categorical_fill: str = 'mode',
        custom_unknown: str = 'Unknown'
    ) -> pd.DataFrame:
       
        if columns is None:
            columns = self.df.columns.tolist()
        elif not isinstance(columns, list):
            columns = [columns]
        
        cleaned_df = self.df.copy()
        missing_before = cleaned_df[columns].isnull().sum().sum()
        
        # Mapping for categorical fill options
        categorical_fill_map = {
            'unknown': 'Unknown',
            'n/a': 'N/A',
            'missing': 'Missing',
            'none': 'None',
            'empty': '',
            'null': 'Null',
            'na': 'NA'
        }
        
        for col in columns:
            if col not in cleaned_df.columns:
                print(f"⚠ Column '{col}' not found in DataFrame")
                continue
                
            if not cleaned_df[col].isnull().any():
                continue
            
            is_numeric = pd.api.types.is_numeric_dtype(cleaned_df[col])
            col_strategy = strategy
            
            # Auto-strategy based on column type and missing percentage
            if strategy == 'auto':
                missing_pct = cleaned_df[col].isnull().sum() / len(cleaned_df)
                
                # If more than 50% missing, consider dropping
                if missing_pct > 0.5:
                    col_strategy = 'drop'
                    print(f"   ℹ {col}: {missing_pct:.1%} missing → dropping column")
                elif is_numeric:
                    col_strategy = 'median'
                else:
                    col_strategy = 'mode'
            
            # Handle based on strategy
            if col_strategy == 'drop':
                cleaned_df = cleaned_df.dropna(subset=[col])
                print(f"   ✓ {col}: Dropped rows with missing values")
            
            elif col_strategy == 'mean' and is_numeric:
                mean_val = cleaned_df[col].mean()
                cleaned_df[col].fillna(mean_val, inplace=True)
                print(f"   ✓ {col}: Filled with mean ({mean_val:.2f})")
            
            elif col_strategy == 'median' and is_numeric:
                median_val = cleaned_df[col].median()
                cleaned_df[col].fillna(median_val, inplace=True)
                print(f"   ✓ {col}: Filled with median ({median_val:.2f})")
            
            elif col_strategy == 'mode':
                mode_val = cleaned_df[col].mode()
                
                if not mode_val.empty:
                    # Use mode value
                    cleaned_df[col].fillna(mode_val[0], inplace=True)
                    print(f"   ✓ {col}: Filled with mode ({mode_val[0]})")
                else:
                    # No mode found - use categorical fill strategy
                    if categorical_fill in categorical_fill_map:
                        fill_val = categorical_fill_map[categorical_fill]
                    elif categorical_fill == 'custom':
                        fill_val = custom_unknown
                    else:
                        # Default to 'Unknown'
                        fill_val = 'Unknown'
                    
                    cleaned_df[col].fillna(fill_val, inplace=True)
                    print(f"   ✓ {col}: No mode found → filled with '{fill_val}'")
            
            elif col_strategy == 'constant':
                cleaned_df[col].fillna(fill_value, inplace=True)
                print(f"   ✓ {col}: Filled with constant ({fill_value})")
            
            elif col_strategy == 'ffill':
                cleaned_df[col].fillna(method='ffill', inplace=True)
                print(f"   ✓ {col}: Forward filled")
            
            elif col_strategy == 'bfill':
                cleaned_df[col].fillna(method='bfill', inplace=True)
                print(f"   ✓ {col}: Backward filled")
            
            else:
                # Unknown strategy - use mode with categorical fallback
                print(f"⚠ Unknown strategy '{col_strategy}' for column {col}. Using mode.")
                mode_val = cleaned_df[col].mode()
                if not mode_val.empty:
                    cleaned_df[col].fillna(mode_val[0], inplace=True)
                else:
                    fill_val = categorical_fill_map.get(categorical_fill, 'Unknown')
                    cleaned_df[col].fillna(fill_val, inplace=True)
        
        missing_after = cleaned_df[columns].isnull().sum().sum()
        missing_fixed = missing_before - missing_after
        
        self.cleaning_report['missing_values_handled'] = {
            'before': missing_before,
            'after': missing_after,
            'fixed': missing_fixed,
            'strategy': strategy,
            'columns_processed': columns,
            'categorical_fill': categorical_fill
        }
        
        self.df = cleaned_df
        print(f"\n✓ Handled {missing_fixed} missing values in '{self.table_name}'")
        
        return cleaned_df
    
    # ============================================================
    # DUPLICATES HANDLING
    # ============================================================
    
    def detect_duplicates(self, subset: Optional[List[str]] = None) -> int:
        
        if subset is None:
            duplicates = self.df.duplicated()
        else:
            duplicates = self.df.duplicated(subset=subset)
        
        duplicate_count = duplicates.sum()
        
        self.cleaning_report['duplicates'] = {
            'count': duplicate_count,
            'percentage': (duplicate_count / len(self.df)) * 100 if len(self.df) > 0 else 0,
            'subset': subset
        }
        
        print(f"ℹ Found {duplicate_count} duplicate rows in '{self.table_name}'")
        
        return duplicate_count
    
    def handle_duplicates(
        self,
        subset: Optional[List[str]] = None,
        keep: str = 'first'
    ) -> pd.DataFrame:
        
        before_count = len(self.df)
        
        if subset is None:
            self.df = self.df.drop_duplicates(keep=keep)
        else:
            self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        
        after_count = len(self.df)
        removed = before_count - after_count
        
        self.cleaning_report['duplicates_handled'] = {
            'before': before_count,
            'after': after_count,
            'removed': removed,
            'keep': keep,
            'subset': subset
        }
        
        print(f"✓ Removed {removed} duplicate rows from '{self.table_name}'")
        
        return self.df
    
    # ============================================================
    # OUTLIER DETECTION AND HANDLING
    # ============================================================
    
    def detect_outliers_iqr(
        self,
        columns: Optional[List[str]] = None,
        multiplier: float = 1.5
    ) -> Dict[str, Dict]:
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        elif not isinstance(columns, list):
            columns = [columns]
        
        outlier_report = {}
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                outlier_report[col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(self.df)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'q1': Q1,
                    'q3': Q3,
                    'iqr': IQR
                }
                print(f"ℹ {col}: Found {outlier_count} outliers (IQR method)")
        
        self.cleaning_report['outliers_iqr'] = outlier_report
        
        return outlier_report
    
    def detect_outliers_zscore(
        self,
        columns: Optional[List[str]] = None,
        threshold: float = 3
    ) -> Dict[str, Dict]:
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        elif not isinstance(columns, list):
            columns = [columns]
        
        outlier_report = {}
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            # Drop NaN values for Z-score calculation
            col_data = self.df[col].dropna()
            if len(col_data) < 2:
                continue
                
            z_scores = np.abs(stats.zscore(col_data))
            outliers = z_scores > threshold
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                outlier_report[col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(self.df)) * 100,
                    'threshold': threshold,
                    'max_zscore': z_scores.max() if len(z_scores) > 0 else 0
                }
                print(f"ℹ {col}: Found {outlier_count} outliers (Z-score method)")
        
        self.cleaning_report['outliers_zscore'] = outlier_report
        
        return outlier_report
    
    def handle_outliers(
        self,
        method: str = 'iqr',
        columns: Optional[List[str]] = None,
        strategy: str = 'cap',
        multiplier: float = 1.5,
        threshold: float = 3
    ) -> pd.DataFrame:
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        elif not isinstance(columns, list):
            columns = [columns]
        
        cleaned_df = self.df.copy()
        total_outliers_handled = 0
        
        for col in columns:
            if col not in cleaned_df.columns:
                continue
                
            # Detect outliers
            if method == 'iqr':
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                outlier_mask = (cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)
                
            else:  # zscore
                col_data = cleaned_df[col].dropna()
                if len(col_data) < 2:
                    continue
                z_scores = np.abs(stats.zscore(col_data))
                outlier_indices = col_data.index[z_scores > threshold]
                outlier_mask = cleaned_df.index.isin(outlier_indices)
            
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                total_outliers_handled += outlier_count
                
                if strategy == 'cap':
                    if method == 'iqr':
                        cleaned_df.loc[cleaned_df[col] < lower_bound, col] = lower_bound
                        cleaned_df.loc[cleaned_df[col] > upper_bound, col] = upper_bound
                    else:
                        # Cap to mean ± threshold * std
                        mean_val = cleaned_df[col].mean()
                        std_val = cleaned_df[col].std()
                        lower_cap = mean_val - threshold * std_val
                        upper_cap = mean_val + threshold * std_val
                        cleaned_df.loc[cleaned_df[col] < lower_cap, col] = lower_cap
                        cleaned_df.loc[cleaned_df[col] > upper_cap, col] = upper_cap
                    print(f"   ✓ {col}: Capped {outlier_count} outliers")
                
                elif strategy == 'remove':
                    cleaned_df = cleaned_df[~outlier_mask]
                    print(f"   ✓ {col}: Removed {outlier_count} outliers")
                
                elif strategy == 'transform' and (cleaned_df[col] > 0).all():
                    cleaned_df[col] = np.log1p(cleaned_df[col])
                    print(f"   ✓ {col}: Log-transformed {outlier_count} outliers")
                
                elif strategy == 'median':
                    median_val = cleaned_df[col].median()
                    cleaned_df.loc[outlier_mask, col] = median_val
                    print(f"   ✓ {col}: Replaced {outlier_count} outliers with median")
                
                else:
                    print(f"⚠ Unknown strategy '{strategy}' for column {col}. Using cap.")
                    if method == 'iqr':
                        cleaned_df.loc[cleaned_df[col] < lower_bound, col] = lower_bound
                        cleaned_df.loc[cleaned_df[col] > upper_bound, col] = upper_bound
        
        self.df = cleaned_df
        self.cleaning_report['outliers_handled'] = {
            'method': method,
            'strategy': strategy,
            'columns_processed': columns,
            'total_outliers_handled': total_outliers_handled
        }
        
        print(f"\n✓ Handled {total_outliers_handled} outliers in '{self.table_name}'")
        
        return cleaned_df
    
    # ============================================================
    # DATA TYPE VALIDATION
    # ============================================================
    
    def validate_dtypes(
        self,
        expected_types: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        
        if expected_types:
            conversions = 0
            for col, dtype in expected_types.items():
                if col in self.df.columns:
                    try:
                        self.df[col] = self.df[col].astype(dtype)
                        print(f"✓ Converted {col} to {dtype}")
                        conversions += 1
                    except Exception as e:
                        print(f"⚠ Could not convert {col}: {e}")
            self.cleaning_report['type_conversions'] = {
                'attempted': len(expected_types),
                'successful': conversions
            }
        
        # Report current types
        type_report = pd.DataFrame({
            'Column': self.df.columns,
            'Current_Type': self.df.dtypes.values,
            'Unique_Values': [self.df[col].nunique() for col in self.df.columns],
            'Null_Count': [self.df[col].isnull().sum() for col in self.df.columns],
            'Null_Percentage': [(self.df[col].isnull().sum() / len(self.df)) * 100 for col in self.df.columns]
        })
        
        self.cleaning_report['data_types'] = type_report
        return type_report
    
    # ============================================================
    # DATA QUALITY SCORE
    # ============================================================
    
    def get_quality_score(self) -> Dict[str, Union[float, str, Dict]]:
        
        score = 100
        penalties = {}
        
        # Check missing values
        missing_pct = self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100 if len(self.df) > 0 else 0
        if missing_pct > 0:
            score -= missing_pct * 2  # 2% penalty per missing percentage
            penalties['missing_values'] = round(missing_pct, 2)
        
        # Check duplicate rows
        dup_pct = self.df.duplicated().sum() / len(self.df) * 100 if len(self.df) > 0 else 0
        if dup_pct > 0:
            score -= dup_pct * 3  # 3% penalty per duplicate percentage
            penalties['duplicates'] = round(dup_pct, 2)
        
        # Check for constant columns (no variance)
        constant_cols = [col for col in self.df.columns if self.df[col].nunique() <= 1]
        if constant_cols:
            score -= len(constant_cols) * 2
            penalties['constant_columns'] = constant_cols
        
        # Check for high cardinality text columns
        text_cols = self.df.select_dtypes(include=['object']).columns
        high_cardinality = []
        for col in text_cols:
            if len(self.df) > 0 and self.df[col].nunique() / len(self.df) > 0.5:
                high_cardinality.append(col)
        if high_cardinality:
            score -= len(high_cardinality) * 1
            penalties['high_cardinality'] = high_cardinality
        
        # Check for all-null columns
        all_null = [col for col in self.df.columns if self.df[col].isnull().all()]
        if all_null:
            score -= len(all_null) * 5
            penalties['all_null_columns'] = all_null
        
        # Cap score at 0-100
        score = max(0, min(100, score))
        
        quality_report = {
            'score': round(score, 1),
            'grade': self._get_grade(score),
            'penalties': penalties
        }
        
        self.cleaning_report['quality_score'] = quality_report
        return quality_report
    
    def _get_grade(self, score: float) -> str:

        if score >= 90:
            return 'A (Excellent)'
        elif score >= 80:
            return 'B (Good)'
        elif score >= 70:
            return 'C (Fair)'
        elif score >= 60:
            return 'D (Poor)'
        else:
            return 'F (Needs Work)'
    
    # ============================================================
    # VISUALIZATION METHODS
    # ============================================================
    
    def visualize_missing(self, figsize: Tuple[int, int] = (12, 6)):
        
        missing_data = self.df.isnull()
        
        plt.figure(figsize=figsize)
        sns.heatmap(missing_data, cbar=True, yticklabels=False, cmap='viridis')
        plt.title(f'Missing Values Heatmap - {self.table_name}', fontsize=14)
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.tight_layout()
        plt.show()
    
    def visualize_outliers(
        self,
        columns: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (15, 8)
    ):
       
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        elif not isinstance(columns, list):
            columns = [columns]
        
        if not columns:
            print("No numeric columns to visualize")
            return
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Handle single subplot case
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if i < len(axes):
                self.df.boxplot(column=col, ax=axes[i])
                axes[i].set_title(f'{col}')
                axes[i].set_ylabel('')
        
        # Hide extra subplots
        for i in range(len(columns), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Outlier Visualization - {self.table_name}', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def visualize_distributions(
        self,
        columns: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (15, 10)
    ):
    
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        elif not isinstance(columns, list):
            columns = [columns]
        
        if not columns:
            print("No numeric columns to visualize")
            return
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Handle single subplot case
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if i < len(axes):
                self.df[col].hist(bins=30, ax=axes[i], color='skyblue', edgecolor='black')
                axes[i].set_title(f'{col}')
                axes[i].set_xlabel('')
                axes[i].set_ylabel('Frequency')
        
        # Hide extra subplots
        for i in range(len(columns), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Distribution Visualization - {self.table_name}', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    # ============================================================
    # COMPLETE CLEANING PIPELINE
    # ============================================================
    
    def clean_pipeline(
        self,
        handle_missing: bool = True,
        remove_duplicates: bool = True,
        handle_outliers: bool = True,
        missing_strategy: str = 'auto',
        categorical_fill: str = 'unknown',
        custom_unknown: str = 'Unknown',
        outlier_method: str = 'iqr',
        outlier_strategy: str = 'cap',
        verbose: bool = True
    ) -> 'DataCleaner':
        if verbose:
            print(f"\n{'='*60}")
            print(f"🧹 STARTING CLEANING PIPELINE: '{self.table_name}'")
            print(f"{'='*60}")
            print(f"📊 Original shape: {self.original_shape[0]:,} rows × {self.original_shape[1]} cols")
            print(f"{'='*60}\n")
        
        # Step 1: Handle missing values
        if handle_missing:
            if verbose:
                print("📌 Step 1: Handling Missing Values")
                print(f"   Categorical fill: {categorical_fill}")
            missing_before = self.df.isnull().sum().sum()
            self.handle_missing_values(
                strategy=missing_strategy,
                categorical_fill=categorical_fill,
                custom_unknown=custom_unknown
            )
            missing_after = self.df.isnull().sum().sum()
            if verbose:
                print(f"   ✅ Missing values: {missing_before:,} → {missing_after:,} (fixed {missing_before - missing_after:,})\n")
        
        # Step 2: Remove duplicates
        if remove_duplicates:
            if verbose:
                print("📌 Step 2: Removing Duplicates")
            dup_before = len(self.df)
            self.handle_duplicates()
            dup_after = len(self.df)
            if verbose:
                print(f"   ✅ Duplicates removed: {dup_before - dup_after:,} rows\n")
        
        # Step 3: Handle outliers
        if handle_outliers:
            if verbose:
                print("📌 Step 3: Handling Outliers")
            self.handle_outliers(method=outlier_method, strategy=outlier_strategy)
            if verbose:
                print(f"   ✅ Outliers handled\n")
        
        # Summary
        if verbose:
            print(f"{'='*60}")
            print(f"✅ CLEANING COMPLETED: '{self.table_name}'")
            print(f"{'='*60}")
            print(f"📊 Final shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} cols")
            
            quality = self.get_quality_score()
            print(f"📊 Quality Score: {quality['score']} ({quality['grade']})")
            
            if quality['penalties']:
                print(f"\n⚠ Remaining issues:")
                for key, value in quality['penalties'].items():
                    if key in ['constant_columns', 'high_cardinality', 'all_null_columns']:
                        print(f"   • {key.replace('_', ' ').title()}: {', '.join(value)}")
                    else:
                        print(f"   • {key.replace('_', ' ').title()}: {value}%")
            
            print(f"{'='*60}\n")
        
        return self  # For method chaining
    
    # ============================================================
    # REPORTING METHODS
    # ============================================================
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        return self.cleaning_report
    
    def get_cleaning_summary(self) -> pd.DataFrame:
        summary = {
            'Table': [self.table_name],
            'Original_Rows': [self.original_shape[0]],
            'Original_Cols': [self.original_shape[1]],
            'Final_Rows': [self.df.shape[0]],
            'Final_Cols': [self.df.shape[1]],
            'Missing_Fixed': [self.cleaning_report.get('missing_values_handled', {}).get('fixed', 0)],
            'Duplicates_Removed': [self.cleaning_report.get('duplicates_handled', {}).get('removed', 0)],
            'Outliers_Handled': [self.cleaning_report.get('outliers_handled', {}).get('total_outliers_handled', 0)]
        }
        
        # Add quality score if available
        quality = self.get_quality_score()
        summary['Quality_Score'] = [quality['score']]
        summary['Grade'] = [quality['grade']]
        
        return pd.DataFrame(summary)
    
    def print_report(self):
        print(f"\n{'='*60}")
        print(f"📋 CLEANING REPORT: '{self.table_name}'")
        print(f"{'='*60}")
        
        # Shape information
        print(f"\n📐 Shape:")
        print(f"   Original: {self.original_shape[0]:,} rows × {self.original_shape[1]} cols")
        print(f"   Final:    {self.df.shape[0]:,} rows × {self.df.shape[1]} cols")
        
        # Missing values
        missing_info = self.cleaning_report.get('missing_values_handled', {})
        if missing_info:
            print(f"\n📊 Missing Values:")
            print(f"   Before: {missing_info.get('before', 0):,}")
            print(f"   After:  {missing_info.get('after', 0):,}")
            print(f"   Fixed:  {missing_info.get('fixed', 0):,}")
            print(f"   Categorical Fill: {missing_info.get('categorical_fill', 'N/A')}")
        
        # Duplicates
        dup_info = self.cleaning_report.get('duplicates_handled', {})
        if dup_info:
            print(f"\n📊 Duplicates:")
            print(f"   Removed: {dup_info.get('removed', 0):,}")
        
        # Outliers
        outlier_info = self.cleaning_report.get('outliers_handled', {})
        if outlier_info:
            print(f"\n📊 Outliers:")
            print(f"   Handled: {outlier_info.get('total_outliers_handled', 0):,}")
        
        # Quality Score
        quality = self.get_quality_score()
        print(f"\n📊 Quality Score:")
        print(f"   Score: {quality['score']} ({quality['grade']})")
        if quality['penalties']:
            print(f"   Penalties: {quality['penalties']}")
        
        print(f"\n{'='*60}")