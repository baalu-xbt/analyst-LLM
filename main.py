"""
Multi-Agent Data Analyst - Main Module
======================================
Orchestrates the multi-agent system for data analysis.
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from utils.data_processor import DataProcessor
from utils.visualizer import DataVisualizer, auto_visualize
from agents.crew_agents import DataAnalysisCrew


def ensure_directories():
    """Create necessary output directories."""
    dirs = [
        "outputs",
        "outputs/charts",
        "outputs/reports",
        "outputs/data"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def run_analysis(file_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Run the complete multi-agent data analysis pipeline.
    
    Args:
        file_path: Path to the dataset file
        verbose: Whether to print progress
        
    Returns:
        Dictionary with all results
    """
    ensure_directories()
    
    results = {
        "status": "started",
        "file_path": file_path,
        "timestamp": datetime.now().isoformat(),
    }
    
    try:
        # Step 1: Load and process data
        if verbose:
            print("\n" + "=" * 60)
            print("🚀 MULTI-AGENT DATA ANALYST")
            print("=" * 60)
            print(f"\n📁 Loading dataset: {file_path}")
        
        processor = DataProcessor(file_path)
        df = processor.load_data()
        
        if verbose:
            print(f"   ✓ Loaded {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Step 2: Clean data
        if verbose:
            print("\n🧹 Cleaning data...")
        
        df_cleaned = processor.clean_data()
        
        if verbose:
            print(f"   ✓ Cleaned data: {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns")
            for action in processor.cleaning_report.get("actions", []):
                print(f"      - {action}")
        
        # Step 3: Generate data profile
        if verbose:
            print("\n📊 Generating data profile...")
        
        data_profile = processor.generate_profile_text()
        results["data_profile"] = data_profile
        
        if verbose:
            print("   ✓ Profile generated")
        
        # Step 4: Generate visualizations
        if verbose:
            print("\n📈 Creating visualizations...")
        
        visualizer = auto_visualize(df_cleaned)
        chart_files = visualizer.save_charts(format="html")
        results["charts"] = chart_files
        
        if verbose:
            print(f"   ✓ Created {len(chart_files)} charts")
            for f in chart_files:
                print(f"      - {f}")
        
        # Step 5: Save cleaned data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cleaned_path = f"outputs/data/cleaned_{timestamp}.csv"
        processor.save_cleaned_data(cleaned_path)
        results["cleaned_data_path"] = cleaned_path
        
        if verbose:
            print(f"\n💾 Saved cleaned data: {cleaned_path}")
        
        # Step 6: Run AI agents (if API key is available)
        api_key = os.getenv("GROQ_API_KEY")
        
        if api_key and api_key.startswith("gsk_"):
            if verbose:
                print("\n🤖 Running AI Analysis Agents...")
            
            crew = DataAnalysisCrew(file_path)
            agent_results = crew.run_full_analysis(data_profile)
            
            results["agent_analysis"] = agent_results.get("analysis", "")
            results["agent_report"] = agent_results.get("report", "")
            results["visualization_code"] = agent_results.get("visualization", "")
            
            # Save report
            report_path = f"outputs/reports/report_{timestamp}.md"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(agent_results.get("report", ""))
            
            results["report_path"] = report_path
            
            if verbose:
                print(f"\n📝 Saved report: {report_path}")
        else:
            if verbose:
                print("\n⚠️ Groq API key not found. Skipping AI agent analysis.")
                print("   Set GROQ_API_KEY in .env file to enable AI features.")
            
            results["agent_analysis"] = "API key not configured"
            results["agent_report"] = "API key not configured"
        
        results["status"] = "completed"
        
        if verbose:
            print("\n" + "=" * 60)
            print("✅ ANALYSIS COMPLETE!")
            print("=" * 60)
        
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        
        if verbose:
            print(f"\n❌ Error: {e}")
    
    return results


def quick_profile(file_path: str) -> str:
    """
    Generate a quick profile of the dataset without AI agents.
    
    Args:
        file_path: Path to the dataset
        
    Returns:
        Profile text
    """
    processor = DataProcessor(file_path)
    processor.load_data()
    processor.clean_data()
    return processor.generate_profile_text()


def main():
    """Command line interface."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <dataset_path>")
        print("\nExample:")
        print("  python main.py data/sales.csv")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    results = run_analysis(file_path, verbose=True)
    
    if results["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
