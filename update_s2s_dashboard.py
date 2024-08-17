import argparse
from collections import defaultdict
import datetime
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--backend-compile", default=False, action="store_true", help="")

    args = parser.parse_args()

    missing_button = f"[![missing](https://img.shields.io/badge/missing-gray)]()"
    test_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'numpy': missing_button, 'jax': missing_button, 'tensorflow': missing_button})))

    color_codes = {
        "passed": "green",
        "failed": "red",
        "skipped": "yellow",
        "missing": "gray",
    }

    passed = 0
    failed = 0
    test_outcomes = {}

    for subdir, _, files in os.walk("artifacts"):
        for file_name in files:
            if "_results" in file_name:
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

                        backend_compile = "T" in backend_compile.upper()  # convert to bool
                        if backend_compile != args.backend_compile:
                            # do not consider this test, as it uses the wrong setting of backend_compile
                            continue

                        if outcome == "passed":
                            passed += 1
                        else:
                            failed += 1

                        if function not in test_outcomes:
                            test_outcomes[function] = {
                                "jax": False,
                                "numpy": False,
                                "tensorflow": False,
                            }

                        test_outcomes[function][target if mode in ["transpile", "s2s"] else "trace"] = outcome == "passed"

                        split_fn = function.split(".")
                        integration = split_fn[0]
                        if len(split_fn) > 2:
                            submodule = split_fn[1] if split_fn[1] != "geometry" else "geometry." + split_fn[2]
                        else:
                            submodule = ""
                        if integration in ["transformers"]:
                            # remove the subsection name here, as it is not actually the submodule in these cases
                            function = integration + "." + split_fn[-1]

                        color = color_codes.get(outcome, 'yellow')
                        button = f"[![{outcome}](https://img.shields.io/badge/{outcome}-{color})]({workflow_link})"
                        if workflow_link not in [None, "null"]:
                            test_results[integration][submodule][function][target if mode in ["transpile", "s2s"] else "trace_graph"] = button

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

    # sort the paths & functions
    sorted_paths = sorted(test_results.keys())
    sorted_test_results = {path: dict(sorted(test_results[path].items())) for path in sorted_paths}

    now = datetime.datetime.now()
    current_date = now.date()

    readme_content = "# Ivy Integration Tests Dashboard\n\n"
    readme_content += f"### Last updated: {current_date}\n\n"
    readme_content += f"- Total Tests Passing: {passed}\n"
    readme_content += f"- Total Tests Failing: {failed}\n"
    readme_content += f"- Percent Tests Passing: {percent_passing}%\n"
    readme_content += f"- Successfully Transpiling to all targets: {percent_fns_passing_all_targets}%\n"
    readme_content += f"- Successfully Transpiling to TensorFlow: {percent_fns_passing_tensorflow}%\n"

    for integration, submodule_functions in sorted_test_results.items():
        readme_content += f"<div style='margin-top: 35px; margin-bottom: 20px; margin-left: 25px;'>\n"
        readme_content += f"<details>\n<summary style='margin-right: 10px;'><span style='font-size: 1.5em; font-weight: bold'>{integration}</span></summary>\n\n"

        for submodule, functions in submodule_functions.items():
            readme_content += f"<div style='margin-top: 7px; margin-botton: 1px; margin-left: 25px;'>\n"
            readme_content += f"<details>\n<summary><span style=''>{submodule}</span></summary>\n\n"
            readme_content += "| Function | numpy | jax | tensorflow |\n"
            readme_content += "|----------|-------|-----|------------|\n"

            for function, results in functions.items():
                readme_content += f"| {function} | {results['numpy']} | {results['jax']} | {results['tensorflow']} |\n"

            readme_content += "</details>\n\n"
            readme_content += "</div>\n\n"
        readme_content += "</details>\n\n"
        readme_content += "</div>\n\n"

    dashboard_name = "NATIVE_COMPILATION_DASHBOARD.md" if args.backend_compile else "DASHBOARD.md"

    with open(dashboard_name, "w") as f:
        f.write(readme_content)

    with open(dashboard_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            print(line)
    print("passed:", passed)
    print("failed:", failed)
    print(f"{percent_passing}% tests passing")
