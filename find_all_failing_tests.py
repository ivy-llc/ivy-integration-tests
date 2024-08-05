import os


if __name__ == "__main__":
    failing_tests = []

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

                    backend_compile = "T" in backend_compile.upper()  # convert to bool

                    if outcome == "failed":
                        failing_tests.append(f"Obj: {function} | Target: {target} | Native Compiled: {backend_compile}\n")

    failing_tests = sorted(failing_tests)

    with open("FAILING_TESTS.txt") as out_file:
        for line in failing_tests:
            out_file.write(line)
