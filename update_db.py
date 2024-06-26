"""
Reads in a test_report.json which has been created by pytest and adds/replaces information
for each test that has been run in the ivy integrations testing remote db.
"""

import argparse
import json
from pymongo import MongoClient


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add all the tests within test_report.json to the remote MongoDB."
    )
    parser.add_argument(
        "--workflow-link",
        type=str,
        help="Link to the GitHub actions workflow corresponding to this test.",
    )
    parser.add_argument("--db-key", type=str, help="Key for the MongoDB database")

    args = parser.parse_args()
    json_report_file = "test_report.json"

    # load in the test report
    with open(json_report_file, "r") as file:
        data = json.load(file)
    tests_data = data.get("tests", None)

    # connect to the database
    uri = f"mongodb+srv://{args.integration_test_db_key}@ivy-integration-tests.pkt1la7.mongodb.net/?retryWrites=true&w=majority&appName=ivy-integration-tests"
    client = MongoClient(uri)
    db = client.ivytestdashboard
    collection = db["test_results"]

    # upload the information for each individual test ran to the database
    for test in tests_data:
        test_path, test_function_args = test["nodeid"].split("::")
        split_path = test_path.split("/", 1)
        integration = split_path[0]
        test_file = split_path[1]
        
        _, test_args = test_function_args.split("[")
        test_args = test_args.replace("]", "")
        test_args = test_args.split("-")

        target, mode, backend_compile = test_args
        integration_fn = test["call"]["stdout"].replace("\n", "")

        document = {
            "target": target,
            "mode": mode,
            "backend_compile": backend_compile,
            "function": integration_fn,
            "workflow_link": args.workflow_link,
            "outcome": test["outcome"],
        }
        filter_criteria = {
            "target": target,
            "mode": mode,
            "backend_compile": backend_compile,
            "function": integration_fn,
        }

        result = collection.replace_one(filter_criteria, document, upsert=True)
