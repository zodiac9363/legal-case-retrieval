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
        from src.pipeline import cross_validate, load_config
        import os
        config = load_config("config.yaml")
        path = os.path.join(config['data']['base_path'], config['data']['default_regime'])
        cross_validate(path, config, method='bm25')
        cross_validate(path, config, method='mmr')
    elif command == "main":
        print("Running main proposed method...")
        from src.pipeline import cross_validate, load_config
        import os
        config = load_config("config.yaml")
        path = os.path.join(config['data']['base_path'], config['data']['default_regime'])
        cross_validate(path, config, method='proposed')
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
        from src.report import generate_report
        generate_report()
    elif command == "app":
        import subprocess
        subprocess.run(["python", "src/app.py"])
    else:
        print(f"Unknown command {command}")
        
if __name__ == "__main__":
    main()
