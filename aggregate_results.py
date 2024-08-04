import os

# Directory where artifacts are downloaded
artifacts_dir = 'artifacts'
output_file = 'aggregated-results/all-test-results.txt'

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

outfile = open(output_file, 'x')

for subdir, _, files in os.walk(artifacts_dir):
    for file in files:
        print('file', file)
        # if file.startswith('test-results-'):
        # print('file', file)
        file_path = os.path.join(subdir, file)
        print('subdir', subdir)
        print('file_path', file_path)
        infile = open(file_path, "r")
        print('file contents', infile.read())
        for line in infile:
            print('writing line', line)
            outfile.write(line)
outfile.close()

with open(output_file, 'r') as outfile:
    print('lines of out file:')
    for line in outfile:
        print('line of out file:', line)
    print('done')
