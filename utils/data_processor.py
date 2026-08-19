"""
Data Processing Utilities
=========================
Functions for loading, cleaning, profiling, and analyzing datasets.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from io import StringIO


class DataProcessor:
    """
    Comprehensive data processing class for loading, cleaning, 
    profiling, and analyzing datasets.
    """
    
    def __init__(self, file_path: str = None):
        self.file_path = file_path
        self.df = None
        self.df_cleaned = None
        self.profile = {}
        self.cleaning_report = {}
        
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load data from CSV or Excel file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Loaded DataFrame
        """
        file_path = file_path or self.file_path
        
        if file_path is None:
            raise ValueError("No file path provided")
        
        # Determine file type and load
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.csv':
            self.df = pd.read_csv(file_path)
        elif ext in ['.xlsx', '.xls']:
            self.df = pd.read_excel(file_path)
        elif ext == '.json':
            self.df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        self.file_path = file_path
        return self.df
    
    def load_from_uploaded_file(self, uploaded_file) -> pd.DataFrame:
        """
        Load data from Streamlit uploaded file.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            Loaded DataFrame
        """
        file_name = uploaded_file.name
        ext = os.path.splitext(file_name)[1].lower()
        
        if ext == '.csv':
            self.df = pd.read_csv(uploaded_file)
        elif ext in ['.xlsx', '.xls']:
            self.df = pd.read_excel(uploaded_file)
        elif ext == '.json':
            self.df = pd.read_json(uploaded_file)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        self.file_path = file_name
        return self.df
    
    def get_basic_info(self) -> Dict[str, Any]:
        """Get basic dataset information."""
        if self.df is None:
            raise ValueError("No data loaded")
        
        return {
            "shape": self.df.shape,
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "column_names": list(self.df.columns),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "memory_usage": self.df.memory_usage(deep=True).sum() / 1024**2,  # MB
        }
    
    def get_missing_values(self) -> pd.DataFrame:
        """Analyze missing values in the dataset."""
        if self.df is None:
            raise ValueError("No data loaded")
        
        missing = pd.DataFrame({
            'column': self.df.columns,
            'missing_count': self.df.isnull().sum().values,
            'missing_percentage': (self.df.isnull().sum().values / len(self.df) * 100).round(2),
            'dtype': self.df.dtypes.astype(str).values
        })
        
        return missing.sort_values('missing_percentage', ascending=False)
    
    def get_duplicates(self) -> Dict[str, Any]:
        """Analyze duplicate rows."""
        if self.df is None:
            raise ValueError("No data loaded")
        
        dup_count = self.df.duplicated().sum()
        
        return {
            "duplicate_count": dup_count,
            "duplicate_percentage": round(dup_count / len(self.df) * 100, 2),
            "unique_rows": len(self.df) - dup_count,
        }
    
    def detect_outliers_iqr(self, column: str) -> Dict[str, Any]:
        """
        Detect outliers using IQR method for a numerical column.
        
        Args:
            column: Column name to analyze
            
        Returns:
            Dictionary with outlier information
        """
        if self.df is None:
            raise ValueError("No data loaded")
        
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        if not np.issubdtype(self.df[column].dtype, np.number):
            return {"column": column, "outliers": 0, "message": "Not a numerical column"}
        
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        
        return {
            "column": column,
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": len(outliers),
            "outlier_percentage": round(len(outliers) / len(self.df) * 100, 2),
        }
    
    def get_all_outliers(self) -> List[Dict[str, Any]]:
        """Detect outliers for all numerical columns."""
        if self.df is None:
            raise ValueError("No data loaded")
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers = []
        
        for col in numerical_cols:
            outlier_info = self.detect_outliers_iqr(col)
            if outlier_info["outlier_count"] > 0:
                outliers.append(outlier_info)
        
        return outliers
    
    def clean_data(self, 
                   fill_numerical: str = "median",
                   fill_categorical: str = "mode",
                   remove_duplicates: bool = True,
                   cap_outliers: bool = True) -> pd.DataFrame:
        """
        Clean the dataset.
        
        Args:
            fill_numerical: Strategy for filling numerical missing values ("mean", "median", "zero")
            fill_categorical: Strategy for filling categorical missing values ("mode", "unknown")
            remove_duplicates: Whether to remove duplicate rows
            cap_outliers: Whether to cap outliers using IQR method
            
        Returns:
            Cleaned DataFrame
        """
        if self.df is None:
            raise ValueError("No data loaded")
        
        self.df_cleaned = self.df.copy()
        self.cleaning_report = {
            "original_shape": self.df.shape,
            "actions": []
        }
        
        # Separate columns by type
        numerical_cols = self.df_cleaned.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df_cleaned.select_dtypes(exclude=[np.number]).columns
        
        # Fill numerical missing values
        for col in numerical_cols:
            missing_count = self.df_cleaned[col].isnull().sum()
            if missing_count > 0:
                if fill_numerical == "median":
                    fill_value = self.df_cleaned[col].median()
                elif fill_numerical == "mean":
                    fill_value = self.df_cleaned[col].mean()
                else:
                    fill_value = 0
                
                self.df_cleaned[col].fillna(fill_value, inplace=True)
                self.cleaning_report["actions"].append(
                    f"Filled {missing_count} missing values in '{col}' with {fill_numerical} ({fill_value:.2f})"
                )
        
        # Fill categorical missing values
        for col in categorical_cols:
            missing_count = self.df_cleaned[col].isnull().sum()
            if missing_count > 0:
                if fill_categorical == "mode":
                    mode_val = self.df_cleaned[col].mode()
                    fill_value = mode_val[0] if len(mode_val) > 0 else "Unknown"
                else:
                    fill_value = "Unknown"
                
                self.df_cleaned[col].fillna(fill_value, inplace=True)
                self.cleaning_report["actions"].append(
                    f"Filled {missing_count} missing values in '{col}' with {fill_categorical} ('{fill_value}')"
                )
        
        # Remove duplicates
        if remove_duplicates:
            dup_count = self.df_cleaned.duplicated().sum()
            if dup_count > 0:
                self.df_cleaned.drop_duplicates(inplace=True)
                self.cleaning_report["actions"].append(
                    f"Removed {dup_count} duplicate rows"
                )
        
        # Cap outliers
        if cap_outliers:
            for col in numerical_cols:
                Q1 = self.df_cleaned[col].quantile(0.25)
                Q3 = self.df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_below = (self.df_cleaned[col] < lower_bound).sum()
                outliers_above = (self.df_cleaned[col] > upper_bound).sum()
                
                if outliers_below > 0 or outliers_above > 0:
                    self.df_cleaned[col] = self.df_cleaned[col].clip(lower_bound, upper_bound)
                    self.cleaning_report["actions"].append(
                        f"Capped {outliers_below + outliers_above} outliers in '{col}' "
                        f"(bounds: [{lower_bound:.2f}, {upper_bound:.2f}])"
                    )
        
        self.cleaning_report["final_shape"] = self.df_cleaned.shape
        
        return self.df_cleaned
    
    def get_numerical_stats(self) -> pd.DataFrame:
        """Get descriptive statistics for numerical columns."""
        df = self.df_cleaned if self.df_cleaned is not None else self.df
        
        if df is None:
            raise ValueError("No data loaded")
        
        return df.describe().round(2)
    
    def get_categorical_stats(self) -> Dict[str, pd.Series]:
        """Get value counts for categorical columns."""
        df = self.df_cleaned if self.df_cleaned is not None else self.df
        
        if df is None:
            raise ValueError("No data loaded")
        
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        stats = {}
        
        for col in categorical_cols:
            stats[col] = df[col].value_counts().head(10)
        
        return stats
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix for numerical columns."""
        df = self.df_cleaned if self.df_cleaned is not None else self.df
        
        if df is None:
            raise ValueError("No data loaded")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            return pd.DataFrame()
        
        return df[numerical_cols].corr().round(3)
    
    def get_strong_correlations(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Find strong correlations (|r| > threshold)."""
        corr_matrix = self.get_correlation_matrix()
        
        if corr_matrix.empty:
            return []
        
        strong_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) > threshold:
                    strong_corr.append({
                        "variable_1": col1,
                        "variable_2": col2,
                        "correlation": corr_value,
                        "strength": "Strong Positive" if corr_value > 0 else "Strong Negative"
                    })
        
        return sorted(strong_corr, key=lambda x: abs(x["correlation"]), reverse=True)
    
    def create_data_dictionary(self) -> pd.DataFrame:
        """Create a comprehensive data dictionary."""
        df = self.df_cleaned if self.df_cleaned is not None else self.df
        
        if df is None:
            raise ValueError("No data loaded")
        
        dictionary = []
        
        for col in df.columns:
            col_data = df[col]
            
            entry = {
                "column_name": col,
                "data_type": str(col_data.dtype),
                "non_null_count": col_data.count(),
                "null_count": col_data.isnull().sum(),
                "unique_values": col_data.nunique(),
                "sample_values": str(col_data.dropna().head(3).tolist())[:50],
            }
            
            if np.issubdtype(col_data.dtype, np.number):
                entry["min"] = col_data.min()
                entry["max"] = col_data.max()
                entry["mean"] = col_data.mean()
            else:
                entry["min"] = "-"
                entry["max"] = "-"
                entry["mean"] = "-"
            
            dictionary.append(entry)
        
        return pd.DataFrame(dictionary)
    
    def generate_profile_text(self) -> str:
        """
        Generate a comprehensive text profile of the dataset.
        This is used as input for AI agents.
        """
        df = self.df_cleaned if self.df_cleaned is not None else self.df
        
        if df is None:
            raise ValueError("No data loaded")
        
        profile_parts = []
        
        # Basic Info
        basic_info = self.get_basic_info()
        profile_parts.append("=" * 60)
        profile_parts.append("DATASET PROFILE")
        profile_parts.append("=" * 60)
        profile_parts.append(f"\n📁 File: {self.file_path}")
        profile_parts.append(f"📊 Shape: {basic_info['rows']} rows × {basic_info['columns']} columns")
        profile_parts.append(f"💾 Memory: {basic_info['memory_usage']:.2f} MB")
        
        # Column Types
        profile_parts.append("\n" + "-" * 40)
        profile_parts.append("COLUMN TYPES")
        profile_parts.append("-" * 40)
        for col, dtype in basic_info['dtypes'].items():
            profile_parts.append(f"  • {col}: {dtype}")
        
        # Missing Values
        missing = self.get_missing_values()
        missing_cols = missing[missing['missing_count'] > 0]
        profile_parts.append("\n" + "-" * 40)
        profile_parts.append("MISSING VALUES")
        profile_parts.append("-" * 40)
        if len(missing_cols) > 0:
            for _, row in missing_cols.iterrows():
                profile_parts.append(
                    f"  • {row['column']}: {row['missing_count']} ({row['missing_percentage']}%)"
                )
        else:
            profile_parts.append("  ✓ No missing values")
        
        # Duplicates
        dups = self.get_duplicates()
        profile_parts.append("\n" + "-" * 40)
        profile_parts.append("DUPLICATES")
        profile_parts.append("-" * 40)
        profile_parts.append(f"  • Duplicate rows: {dups['duplicate_count']} ({dups['duplicate_percentage']}%)")
        
        # Numerical Statistics
        num_stats = self.get_numerical_stats()
        if not num_stats.empty:
            profile_parts.append("\n" + "-" * 40)
            profile_parts.append("NUMERICAL STATISTICS")
            profile_parts.append("-" * 40)
            profile_parts.append(num_stats.to_string())
        
        # Categorical Statistics
        cat_stats = self.get_categorical_stats()
        if cat_stats:
            profile_parts.append("\n" + "-" * 40)
            profile_parts.append("CATEGORICAL DISTRIBUTIONS (Top 10)")
            profile_parts.append("-" * 40)
            for col, counts in cat_stats.items():
                profile_parts.append(f"\n  {col}:")
                for val, count in counts.items():
                    profile_parts.append(f"    • {val}: {count}")
        
        # Strong Correlations
        strong_corr = self.get_strong_correlations()
        if strong_corr:
            profile_parts.append("\n" + "-" * 40)
            profile_parts.append("STRONG CORRELATIONS (|r| > 0.5)")
            profile_parts.append("-" * 40)
            for corr in strong_corr:
                profile_parts.append(
                    f"  • {corr['variable_1']} ↔ {corr['variable_2']}: "
                    f"{corr['correlation']:.3f} ({corr['strength']})"
                )
        
        # Outliers
        outliers = self.get_all_outliers()
        if outliers:
            profile_parts.append("\n" + "-" * 40)
            profile_parts.append("OUTLIERS (IQR Method)")
            profile_parts.append("-" * 40)
            for out in outliers:
                profile_parts.append(
                    f"  • {out['column']}: {out['outlier_count']} outliers "
                    f"({out['outlier_percentage']}%)"
                )
        
        # Cleaning Report
        if self.cleaning_report and self.cleaning_report.get("actions"):
            profile_parts.append("\n" + "-" * 40)
            profile_parts.append("CLEANING ACTIONS TAKEN")
            profile_parts.append("-" * 40)
            for action in self.cleaning_report["actions"]:
                profile_parts.append(f"  ✓ {action}")
        
        return "\n".join(profile_parts)
    
    def get_dataframe(self) -> pd.DataFrame:
        """Return the current (cleaned or original) DataFrame."""
        return self.df_cleaned if self.df_cleaned is not None else self.df
    
    def save_cleaned_data(self, output_path: str) -> str:
        """Save cleaned data to file."""
        df = self.df_cleaned if self.df_cleaned is not None else self.df
        
        if df is None:
            raise ValueError("No data to save")
        
        ext = os.path.splitext(output_path)[1].lower()
        
        if ext == '.csv':
            df.to_csv(output_path, index=False)
        elif ext in ['.xlsx', '.xls']:
            df.to_excel(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)
        
        return output_path


def quick_profile(file_path: str) -> str:
    """
    Quick utility function to generate a profile from a file path.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Profile text string
    """
    processor = DataProcessor(file_path)
    processor.load_data()
    processor.clean_data()
    return processor.generate_profile_text()
