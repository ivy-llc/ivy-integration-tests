#!/bin/bash -e

integration=$1
file=$2
backend_compile=$3
target=$4
workflow_link=$5
api_key=$6

export IVY_KEY=$api_key

pip3 install -e ivy/
cd ivy-integration-tests
pip3 install -r requirements.txt
pip3 install color-operations
pip3 install onnx

if [ "$integration" = "transformers" ]; then
    pip3 install tf_keras
    pip3 install datasets
    pip3 install transformers
fi

# runs the tests on the latest ivy commit
set +e
if [ "$backend_compile" = "T" ]; then
    touch test_logs.txt
    DEBUG=0 COLUMNS=200 pytest $integration/$file.py --backend-compile --source-to-source --target=$target -p no:warnings --tb=long --json-report --json-report-file=test_report.json
    pytest_exit_code=$?
else
    DEBUG=0 COLUMNS=200 pytest $integration/$file.py -p no:warnings --source-to-source --target=$target --tb=long --json-report --json-report-file=test_report.json > test_logs.txt
    pytest_exit_code=$?
fi
set -e

if [ $pytest_exit_code -eq 0 ] || [ $pytest_exit_code -eq 1 ]; then
    python report_file_to_txt.py --workflow-link $workflow_link
    exit 0
else
    exit 1
fi
