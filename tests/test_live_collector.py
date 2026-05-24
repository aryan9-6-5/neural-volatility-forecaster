import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.collector import collect_snapshot

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    print("Running quick live option chain collection test...")
    # Trigger a real fetch for ticker 'SPY' but write to a test raw directory
    test_dir = "data/raw_test"
    df = collect_snapshot(ticker_symbol="SPY", output_dir=test_dir)
    
    if not df.empty:
        print("\nSUCCESS!")
        print(f"Collected a total of {len(df)} contracts.")
        print(f"Columns: {list(df.columns)}")
        print(f"Sample data:\n{df.head(2)}")
        
        # Clean up test file to keep repository clean
        import glob
        test_files = glob.glob(os.path.join(test_dir, "*.parquet"))
        for f in test_files:
            try:
                os.remove(f)
            except Exception:
                pass
        try:
            os.rmdir(test_dir)
        except Exception:
            pass
            
        sys.exit(0)
    else:
        print("\nFAILURE: Could not collect options data. Check network connection or yfinance status.")
        sys.exit(1)
