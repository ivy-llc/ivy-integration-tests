import argparse
from collections import defaultdict
import datetime
import os


if __name__ == "__main__":
    passed = 0
    failed = 0
    test_outcomes = {}
    failing = []

    for subdir, _, files in os.walk("artifacts"):
        for file_name in files:
            file_path = os.path.join(subdir, file_name)

            with open(file_path, "r") as file:
                for line in file.readlines():
                    split_line = line.split(",")[:-1]
                    if len(split_line) != 6:
                        continue
                    record = {
                        "target": split_line[0],
                        "mode": split_line[1],
                        "backend_compile": split_line[2],
                        "function": split_line[3],
                        "workflow_link": split_line[4],
                        "outcome": split_line[5],
                    }

                    target = record["target"]
                    mode = record["mode"]
                    backend_compile = record["backend_compile"]
                    function = record["function"]
                    outcome = record['outcome']
                    workflow_link = record['workflow_link']

                    if outcome == "passed":
                        passed += 1
                    else:
                        failed += 1
                        failing.append(function)

                    if function not in test_outcomes:
                        test_outcomes[function] = {
                            "jax": False,
                            "numpy": False,
                            "tensorflow": False,
                        }

                    test_outcomes[function][target if mode in ["transpile", "s2s"] else "trace"] = outcome == "passed"

    fns_passing_all_targets = 0
    fns_passing_jax = 0
    fns_passing_numpy = 0
    fns_passing_tensorflow = 0
    total_fns = len(test_outcomes)
    for fn, outcomes in test_outcomes.items():
        if all(outcomes.values()): fns_passing_all_targets += 1
        if outcomes["jax"]: fns_passing_jax += 1
        if outcomes["numpy"]: fns_passing_numpy += 1
        if outcomes["tensorflow"]: fns_passing_tensorflow += 1
    percent_fns_passing_all_targets = round(100 * fns_passing_all_targets / total_fns, 2)
    percent_fns_passing_jax = round(100 * fns_passing_jax / total_fns, 2)
    percent_fns_passing_numpy = round(100 * fns_passing_numpy / total_fns, 2)
    percent_fns_passing_tensorflow = round(100 * fns_passing_tensorflow / total_fns, 2)

    if passed + failed > 0:
        percent_passing = round(100 * passed / (passed + failed), 1)
    else:
        percent_passing = 0

    failing = set(failing)
    print("Failing:")
    for failing_fn in failing:
        print(function)

    print(f"\n\nTotal Tests Passing: {passed}")
    print(f"Total Tests Failing: {failed}")
    print(f"Percent Tests Passing: {percent_passing}%")
    print(f"Successfully Transpiling to all targets: {percent_fns_passing_all_targets}%")
    print(f"Successfully Transpiling to TensorFlow: {percent_fns_passing_tensorflow}%")
