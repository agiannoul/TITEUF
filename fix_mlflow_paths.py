import os
import yaml

ROOT_MLFLOW_DIR = "./mlrunsp"      # your mlflow directory
LOCAL_BASE_DIR = os.getcwd()     # automatically use current directory

def fix_meta_yaml(meta_path):
    with open(meta_path, "r") as f:
        data = yaml.safe_load(f)

    exp_id = data.get("experiment_id")
    run_id = data.get("run_id")

    if exp_id is None:
        print(f"Skipped (no experiment_id): {meta_path}")
        return

    # Fix experiment-level artifact_location
    if "artifact_location" in data and run_id is None:
        data["artifact_location"] = os.path.join(
            LOCAL_BASE_DIR, os.path.basename(ROOT_MLFLOW_DIR), exp_id
        )

    # Fix run-level artifact_uri
    if "artifact_uri" in data and run_id is not None:
        data["artifact_uri"] = os.path.join(
            LOCAL_BASE_DIR,
            os.path.basename(ROOT_MLFLOW_DIR),
            exp_id,
            run_id,
            "artifacts",
        )

    with open(meta_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)

    print(f"Updated: {meta_path}")


def main():
    for root, dirs, files in os.walk(ROOT_MLFLOW_DIR):
        if "meta.yaml" in files:
            fix_meta_yaml(os.path.join(root, "meta.yaml"))


if __name__ == "__main__":
    main()
