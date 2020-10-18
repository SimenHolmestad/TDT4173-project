"""Reduce dataset to a more manageable format for testing. The script
will create a new file containing the first 100 000 lines of the
original file.

Run script with:

python3 reduce_dataset.py <file-path-to-original-file> <output-filename>

"""

import sys

if len(sys.argv) < 3:
    print("You have to specify a filename for the original file and a filename for the output file")
    print("You should run script as:")
    print("python3 reduce_dataset.py <file-path-to-file> <output-filename>")
    sys.exit()

filepath = sys.argv[1]
output_filepath = sys.argv[2]

number_of_lines = 100000
lines = []

print("reading lines...")
cnt = 0
with open(filepath) as fp:
    line = 1
    while line and cnt < number_of_lines:
        cnt += 1
        line = fp.readline()
        lines.append(line)

print("Writing {} lines to {}".format(str(cnt), output_filepath))
f = open("output.json", "w")
f.writelines(lines)
