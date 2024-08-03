"""
Reads in a test_report.json which has been created by pytest and
creates a csv file of the results with relevant info.
"""

import argparse
import json
from pymongo import MongoClient
import csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add all the tests within test_report.json to the remote MongoDB."
    )
    parser.add_argument(
        "--workflow-link",
        type=str,
        help="Link to the GitHub actions workflow corresponding to this test.",
    )

    args = parser.parse_args()
    json_report_file = "test_report.json"

    # load in the test report
    with open(json_report_file, "r") as file:
        data = json.load(file)
    tests_data = data.get("tests", None)

    all_documents = []

    # create a .csv file
    for test in tests_data:
        test_path, test_function_args = test["nodeid"].split("::")
        split_path = test_path.split("/", 1)
        integration = split_path[0]
        test_file = split_path[1]

        _, test_args = test_function_args.split("[")
        test_args = test_args.replace("]", "")
        test_args = test_args.split("-")

        target, mode, backend_compile = test_args
        integration_fn = test["call"]["stdout"].split("\n")[0]

        document = {
            "target": target,
            "mode": mode,
            "backend_compile": backend_compile,
            "function": integration_fn,
            "workflow_link": args.workflow_link,
            "outcome": test["outcome"],
        }
        all_documents.append(document)

        # write to a txt file
        with open("test_results.txt", "w") as file:
            for document in all_documents:
                line = ""
                for v in document.values():
                    line += v + ","
                file.write(line + "\n")
