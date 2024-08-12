#!/bin/bash -e

binaries=$1

export VERSION=$binaries  # set the branch to pull the binaries from

pip3 install -e ivy/
cd ivy-integration-tests
pip3 install -r requirements.txt

if [ "$integration" = "transformers" ]; then
    pip3 install tensorflow==2.15.1
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
COLUMNS=200 pytest kornia/test_contrib.py --source-to-source -p no:warnings --tb=line
pytest_exit_code=$?
set -e

if [ $pytest_exit_code -eq 0 ] || [ $pytest_exit_code -eq 1 ]; then
    exit 0
else
    exit 1
fi
