import os

# Directory where artifacts are downloaded
artifacts_dir = './artifacts'
output_file = 'aggregated-results/all-test-results.txt'

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w') as outfile:
    for subdir, _, files in os.walk(artifacts_dir):
        for file in files:
            if file.startswith('test-results-'):
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r') as infile:
                    outfile.write(infile.read())
                    outfile.write('\n')
