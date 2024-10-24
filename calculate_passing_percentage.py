import os
import sys

def main():
    passed = 0
    failed = 0
    skipped = 0
    failing_tests = []

    for subdir, _, files in os.walk("artifacts"):
        for file_name in files:
            if "_results" in file_name:
                file_path = os.path.join(subdir, file_name)

                with open(file_path, "r") as file:
                    for line in file:
                        split_line = line.strip().split(",")[:-1]
                        if len(split_line) != 6:
                            continue
                        outcome = split_line[5]
                        if outcome == 'passed':
                            passed += 1
                        elif outcome == 'failed':
                            failing_tests.append(split_line[3])
                            failed += 1
                        elif outcome == 'skipped':
                            skipped += 1

    total_excluding_skipped = passed + failed
    if total_excluding_skipped == 0:
        passing_percentage = 0
    else:
        passing_percentage = 100 * passed / total_excluding_skipped

    print("Failing tests:")
    for test_name in failing_tests:
        print("-", test_name)
    print(f"\nTotal tests (excluding skips): {total_excluding_skipped}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Passing percentage: {passing_percentage:.2f}%")

    if passing_percentage >= 89.0:
        print("Passing percentage is >= 89%. Success!")
        sys.exit(0)
    else:
        print("Passing percentage is less than 89%. Failing the workflow.")
        sys.exit(1)

if __name__ == "__main__":
    main()
