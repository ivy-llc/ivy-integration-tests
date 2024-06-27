import argparse
from collections import defaultdict
import datetime
from pymongo import MongoClient


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write the test results for the remote MongoDB to the dashboard")
    parser.add_argument("--db-key", type=str, help="Key for the MongoDB database")

    args = parser.parse_args()

    uri = f"mongodb+srv://{args.db_key}@ivy-integration-tests.pkt1la7.mongodb.net/?retryWrites=true&w=majority&appName=ivy-integration-tests"
    client = MongoClient(uri)
    db = client.ivy_integration_tests
    collection = db["test_results"]

    records = collection.find()

    missing_button = f"[![missing](https://img.shields.io/badge/missing-gray)]()"
    test_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'numpy': missing_button, 'jax': missing_button, 'tensorflow': missing_button, 'torch': missing_button, 'trace_graph': missing_button})))

    color_codes = {
        "passed": "green",
        "failed": "red",
        "skipped": "yellow",
        "missing": "gray",
    }

    for record in records:
        target = record["target"]
        mode = record["mode"]
        backend_compile = record["backend_compile"]
        function = record["function"]
        
        split_fn = function.split(".")
        integration = split_fn[0]
        submodule = split_fn[1] if len(split_fn) > 2 else ""
        
        outcome = record['outcome']
        workflow_link = record['workflow_link']
        color = color_codes.get(outcome, 'yellow')
        button = f"[![{outcome}](https://img.shields.io/badge/{outcome}-{color})]({workflow_link})"
        if workflow_link not in [None, "null"]:
            test_results[integration][submodule][function][target if mode == "transpile" else "trace_graph"] = button

    # sort the paths & functions
    sorted_paths = sorted(test_results.keys())
    sorted_test_results = {path: dict(sorted(test_results[path].items())) for path in sorted_paths}

    now = datetime.datetime.now()
    current_date = now.date()

    readme_content = "# Ivy Integration Tests Dashboard\n\n"
    readme_content += f"### Last updated: {current_date}\n\n"

    for integration, submodule_functions in sorted_test_results.items():
        readme_content += f"<div style='margin-top: 35px; margin-bottom: 20px; margin-left: 25px;'>\n"
        readme_content += f"<details>\n<summary style='margin-right: 10px;'><span style='font-size: 1.5em; font-weight: bold'>{integration}</span></summary>\n\n"

        for submodule, functions in submodule_functions.items():
            readme_content += f"<div style='margin-top: 7px; margin-botton: 1px; margin-left: 25px;'>\n"
            readme_content += f"<details>\n<summary><span style=''>{submodule}</span></summary>\n\n"
            readme_content += "| Function | numpy | jax | tensorflow | torch | trace_graph |\n"
            readme_content += "|----------|-------|-----|------------|-------|-------------|\n"

            for function, results in functions.items():
                readme_content += f"| {function} | {results['numpy']} | {results['jax']} | {results['tensorflow']} | {results['torch']} | {results['trace_graph']} |\n"

            readme_content += "</details>\n\n"
            readme_content += "</div>\n\n"
        readme_content += "</details>\n\n"
        readme_content += "</div>\n\n"

    with open("DASHBOARD.md", "w") as f:
        f.write(readme_content)

    with open("DASHBOARD.md", "r") as f:
        lines = f.readlines()
        for line in lines:
            print(line)
