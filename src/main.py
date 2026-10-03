import sys
from src.data import generate_all_datasets

def main():
    if len(sys.argv) < 2:
        print("Usage: make [data|baselines|main|ablations|theory|figures|report|app|test|all]")
        sys.exit(1)
        
    command = sys.argv[1]
    
    if command == "data":
        generate_all_datasets()
    elif command == "baselines":
        print("Running baselines...")
        # To be implemented in M2
    elif command == "main":
        print("Running main...")
        # To be implemented in M3/M5
    elif command == "ablations":
        print("Running ablations...")
        # To be implemented in M5
    elif command == "theory":
        print("Running theory...")
        # To be implemented in M6
    elif command == "figures":
        print("Running figures...")
        # To be implemented in M8
    elif command == "report":
        print("Running report...")
        # To be implemented in M9
    else:
        print(f"Unknown command {command}")
        
if __name__ == "__main__":
    main()
