#!/bin/bash -e

integration=$1
file=$2
backend_compile=$3
workflow_link=$4
api_key=$5

export IVY_KEY=$api_key
export VERSION=linux-nightly  # set the branch to pull the binaries from

pip3 install -e ivy/
cd ivy-integration-tests
pip3 install -r requirements.txt

if [ "$integration" = "transformers" ]; then
    pip3 install datasets
    pip3 install transformers
fi

# get the nightly binaries
python << 'EOF'
import ivy
ivy.utils.cleanup_and_fetch_binaries()
EOF

# runs the tests on the latest ivy commit, and the linux binaries that are built nightly
set +e
if [ "$backend_compile" = "T" ]; then
    pytest $integration/$file.py --backend-compile --source-to-source -p no:warnings --tb=short --json-report --json-report-file=test_report.json
    pytest_exit_code=$?
else
    pytest $integration/$file.py -p no:warnings --source-to-source --tb=short --json-report --json-report-file=test_report.json
    pytest_exit_code=$?
fi
set -e

if [ $pytest_exit_code -eq 0 ] || [ $pytest_exit_code -eq 1 ]; then
    python report_file_to_txt.py --workflow-link $workflow_link
    exit 0
else
    exit 1
fi
