#!/bin/bash -e

integration=$1
file=$2
backend_compile=$3
workflow_link=$4
db_key=$5

set +e
if "$backend_compile" -eq "T"; then
    pytest $integration/$file.py --backend-compile -p no:warnings --tb=short --json-report --json-report-file=test_report.json
    pytest_exit_code=$?
else
    pytest $integration/$file.py -p no:warnings --tb=short --json-report --json-report-file=test_report.json
    pytest_exit_code=$?
fi
set -e

if [ $pytest_exit_code -eq 0 ] || [ $pytest_exit_code -eq 1 ]; then
    python update_db.py --workflow-link $workflow_link --db-key $db_key
    exit 0
else
    exit 1
fi
