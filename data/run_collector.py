import argparse
import time
import sys
from data.collector import collect_snapshot, start_scheduler

def main():
    parser = argparse.ArgumentParser(description="Options Chain Data Collector Runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--now", action="store_true", help="Trigger a one-off options collection snapshot immediately")
    group.add_argument("--scheduler", action="store_true", help="Run the automated daily scheduler in the foreground")
    
    args = parser.parse_args()
    
    if args.now:
        print("Triggering immediate snapshot collection...")
        df = collect_snapshot()
        if not df.empty:
            print(f"Collection complete! Fetched {len(df)} option contracts.")
            sys.exit(0)
        else:
            print("Collection failed or returned no records.")
            sys.exit(1)
            
    elif args.scheduler:
        scheduler = start_scheduler()
        print("Scheduler is running. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            print("\nShutting down scheduler...")
            scheduler.shutdown()
            print("Scheduler stopped.")

if __name__ == "__main__":
    main()
