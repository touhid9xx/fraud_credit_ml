"""
Universal CSV Data Loader with Intelligent Date Detection
Automatically loads all CSV files from a folder and parses date columns
"""

import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime
import warnings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UniversalDataLoader:
    """
    Universal CSV loader that auto-detects and parses date columns
    
    Features:
    - Loads all CSV files from a folder
    - Auto-detects date columns using multiple strategies
    - Parses dates with format detection
    - Provides detailed logging
    - Handles errors gracefully
    
    Usage:
        loader = UniversalDataLoader(data_path='../data/raw/')
        data = loader.load_all(parse_dates=True)
        print(loader.get_summary())
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        encoding: str = 'utf-8',
        sample_rows: int = 10,
        date_keywords: Optional[List[str]] = None,
        verbose: bool = True
    ):
        """
        Initialize the Universal Data Loader
        
        Args:
            data_path: Path to folder containing CSV files
            encoding: File encoding (default: 'utf-8')
            sample_rows: Number of rows to sample for date detection
            date_keywords: Custom keywords to detect date columns
            verbose: Print detailed information
        """
        self.data_path = Path(data_path)
        self.encoding = encoding
        self.sample_rows = sample_rows
        self.verbose = verbose
        
        # Default date keywords
        self.date_keywords = date_keywords or [
            'date', 'datetime', 'time', 'birth', 'birthday',
            'issued', 'maturity', 'opening', 'closing', 
            'created', 'updated', 'start', 'end', 'timestamp',
            'transaction', 'due', 'payment', 'period'
        ]
        
        # Store loaded data
        self.data = {}
        self.date_columns_detected = {}
        self.date_columns_parsed = {}
        self.loading_errors = {}
        self.loading_summary = {}
        
    def _get_csv_files(self) -> List[Path]:
        """Get all CSV files from the data path"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        
        csv_files = list(self.data_path.glob('*.csv'))
        csv_files.extend(list(self.data_path.glob('*.CSV')))
        csv_files = list(set(csv_files))  # Remove duplicates
        
        if not csv_files:
            logger.warning(f"No CSV files found in {self.data_path}")
        
        return sorted(csv_files)
    
    def _detect_date_columns(
        self,
        filepath: Path,
        df_sample: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """
        Detect potential date columns using multiple strategies
        
        Returns:
            Dictionary with 'confident' and 'possible' date columns
        """
        confident_dates = []
        possible_dates = []
        
        for col in df_sample.columns:
            col_lower = col.lower()
            
            # Check if column name contains date keywords
            has_keyword = any(keyword in col_lower for keyword in self.date_keywords)
            
            if not has_keyword:
                continue
                
            # Check if values look like dates
            sample_values = df_sample[col].dropna()
            if len(sample_values) == 0:
                possible_dates.append(col)
                continue
            
            # Try to convert to datetime
            try:
                # Use pandas to_datetime with coercion
                test_conversion = pd.to_datetime(sample_values, errors='coerce')
                success_rate = test_conversion.notna().sum() / len(sample_values)
                
                if success_rate > 0.8:  # 80% success rate
                    confident_dates.append(col)
                    if self.verbose:
                        print(f"   ✓ Confident date: {col} (success rate: {success_rate:.1%})")
                elif success_rate > 0.3:
                    possible_dates.append(col)
                    if self.verbose:
                        print(f"   ? Possible date: {col} (success rate: {success_rate:.1%})")
                else:
                    if self.verbose:
                        print(f"   ✗ Not a date: {col} (success rate: {success_rate:.1%})")
                        
            except Exception as e:
                possible_dates.append(col)
                if self.verbose:
                    print(f"   ⚠ Could not test: {col} ({str(e)})")
        
        return {
            'confident': confident_dates,
            'possible': possible_dates
        }
    
    def _parse_column_as_date(
        self,
        df: pd.DataFrame,
        col: str
    ) -> pd.Series:
        """
        Parse a single column as date with multiple format attempts
        """
        # Try multiple date formats
        formats = [
            None,  # Auto-detect
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
            '%b %d, %Y',
            '%d-%b-%Y',
            '%Y/%m/%d'
        ]
        
        for fmt in formats:
            try:
                # Attempt conversion with specific format
                if fmt is None:
                    result = pd.to_datetime(df[col], errors='coerce')
                else:
                    result = pd.to_datetime(df[col], format=fmt, errors='coerce')
                
                # Check if conversion was successful
                if result.notna().sum() > 0:
                    success_rate = result.notna().sum() / len(df[col])
                    if success_rate > 0.5:
                        if self.verbose:
                            print(f"   ✓ Parsed {col} with format: {fmt or 'auto'}")
                        return result
                    else:
                        # Try next format
                        continue
            except Exception:
                continue
        
        # If all attempts fail, return original column
        if self.verbose:
            print(f"   ⚠ Failed to parse {col} as date")
        return df[col]
    
    def _load_single_file(
        self,
        filepath: Path,
        parse_dates: bool = True,
        infer_objects: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Load a single CSV file with intelligent date parsing
        """
        try:
            # First, try reading with basic settings
            try:
                df = pd.read_csv(filepath, encoding=self.encoding, low_memory=False)
            except UnicodeDecodeError:
                # Try with different encoding
                df = pd.read_csv(filepath, encoding='latin-1', low_memory=False)
            except Exception as e:
                raise Exception(f"Failed to read file: {str(e)}")
            
            if df.empty:
                logger.warning(f"Empty file: {filepath.name}")
                return None
            
            # If parse_dates is True, detect and parse date columns
            if parse_dates:
                # Sample for detection
                df_sample = df.head(self.sample_rows)
                date_detection = self._detect_date_columns(filepath, df_sample)
                
                # Track detected columns
                table_name = filepath.stem
                self.date_columns_detected[table_name] = date_detection
                
                # Parse confident date columns
                parsed_cols = []
                for col in date_detection['confident']:
                    df[col] = self._parse_column_as_date(df, col)
                    parsed_cols.append(col)
                
                # Try parsing possible date columns too
                for col in date_detection['possible']:
                    if col not in parsed_cols:
                        df[col] = self._parse_column_as_date(df, col)
                        parsed_cols.append(col)
                
                self.date_columns_parsed[table_name] = parsed_cols
            
            # Infer object types if requested
            if infer_objects:
                df = df.infer_objects()
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {filepath.name}: {str(e)}")
            self.loading_errors[filepath.stem] = str(e)
            return None
    
    def load_all(
        self,
        parse_dates: bool = True,
        infer_objects: bool = True,
        verbose: bool = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files from the data folder
        
        Args:
            parse_dates: Auto-detect and parse date columns
            infer_objects: Infer object data types
            verbose: Override verbosity setting
        
        Returns:
            Dictionary with table names as keys and DataFrames as values
        """
        if verbose is not None:
            self.verbose = verbose
        
        csv_files = self._get_csv_files()
        
        if not csv_files:
            logger.warning("No CSV files found to load")
            return self.data
        
        print(f"\n{'='*60}")
        print(f"📂 Loading {len(csv_files)} CSV files from: {self.data_path}")
        print(f"{'='*60}\n")
        
        for filepath in csv_files:
            table_name = filepath.stem
            
            try:
                df = self._load_single_file(
                    filepath,
                    parse_dates=parse_dates,
                    infer_objects=infer_objects
                )
                
                if df is not None:
                    self.data[table_name] = df
                    
                    # Build summary
                    parsed_info = ""
                    if parse_dates and table_name in self.date_columns_parsed:
                        parsed_info = f" (parsed: {', '.join(self.date_columns_parsed[table_name])})"
                    
                    print(f"✓ Loaded {table_name}: {df.shape[0]:,} rows, {df.shape[1]} columns{parsed_info}")
                    
                    # Show column types for debugging
                    if self.verbose and len(df.columns) <= 20:
                        print(f"  Columns: {', '.join(df.columns.tolist())}")
                
            except Exception as e:
                print(f"✗ Error loading {table_name}: {str(e)}")
                self.loading_errors[table_name] = str(e)
                self.data[table_name] = None
        
        print(f"\n{'='*60}")
        print(f"✅ Loaded {sum(1 for v in self.data.values() if v is not None)} tables successfully")
        print(f"⚠ Errors: {len(self.loading_errors)} files")
        print(f"{'='*60}\n")
        
        return self.data
    
    def get_table(self, table_name: str) -> Optional[pd.DataFrame]:
        """Get a specific table by name"""
        return self.data.get(table_name)
    
    def get_all_tables(self) -> Dict[str, pd.DataFrame]:
        """Get all loaded tables"""
        return self.data
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the loading process"""
        summary = {
            'total_files_loaded': sum(1 for v in self.data.values() if v is not None),
            'files_with_errors': len(self.loading_errors),
            'date_columns_detected': self.date_columns_detected,
            'date_columns_parsed': self.date_columns_parsed,
            'loading_errors': self.loading_errors,
            'table_shapes': {
                name: df.shape if df is not None else None
                for name, df in self.data.items()
            }
        }
        return summary
    
    def print_summary(self):
        """Print a formatted summary"""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("📊 LOADING SUMMARY")
        print("="*70)
        print(f"✅ Tables loaded: {summary['total_files_loaded']}")
        print(f"❌ Errors: {summary['files_with_errors']}")
        
        print("\n📁 Table Details:")
        for name, shape in summary['table_shapes'].items():
            if shape is not None:
                parsed_cols = self.date_columns_parsed.get(name, [])
                if parsed_cols:
                    print(f"  • {name}: {shape[0]:,} rows × {shape[1]} cols (parsed: {', '.join(parsed_cols)})")
                else:
                    print(f"  • {name}: {shape[0]:,} rows × {shape[1]} cols")
            else:
                print(f"  ✗ {name}: Failed to load")
        
        if summary['loading_errors']:
            print("\n⚠ Errors:")
            for name, error in summary['loading_errors'].items():
                print(f"  • {name}: {error}")
        
        print("="*70)


# Additional utility functions
def validate_data_types(df: pd.DataFrame, table_name: str = "") -> Dict[str, str]:
    """
    Validate and report data types for a DataFrame
    
    Returns:
        Dictionary of column names and their types
    """
    type_summary = {}
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        unique_count = df[col].nunique()
        
        type_summary[col] = {
            'dtype': str(dtype),
            'nulls': null_count,
            'null_pct': f"{null_count/len(df)*100:.1f}%",
            'unique': unique_count
        }
        
        # Detect potential issues
        if dtype == 'object' and unique_count > 1000:
            print(f"⚠ {table_name}.{col}: High cardinality text column ({unique_count} unique)")
        elif dtype == 'object' and len(df[col].str.strip()) == 0:
            print(f"⚠ {table_name}.{col}: Empty strings detected")
        elif dtype == 'float64' and null_count > 0:
            print(f"ℹ {table_name}.{col}: Float with nulls")
    
    return type_summary


# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = UniversalDataLoader(
        data_path='../data/raw/',
        verbose=True,
        sample_rows=5
    )
    
    # Load all files
    data = loader.load_all(parse_dates=True)
    
    # Print summary
    loader.print_summary()
    
    # Validate data types
    for table_name, df in data.items():
        if df is not None:
            print(f"\n📋 Validating {table_name}:")
            validate_data_types(df, table_name)