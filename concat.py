import sys
import os
import json
import re

HELP_MSG = \
"Usage: concat dir out\n" + \
"  dir: directory of json files to concat\n" + \
"  out: output file"

if __name__ == "__main__" :
    argc = len(sys.argv)

    if argc != 3 :
        print(HELP_MSG)
        sys.exit(1)

    directory = os.path.abspath(sys.argv[1])
    output = os.path.abspath(sys.argv[2])

    pattern = re.compile("^img[0-9]+.json$")

    labels = []

    for filename in os.listdir(directory) :

        if pattern.match(filename) :

            input_path = os.path.join(directory, filename)

            with open(input_path) as f :
                label = json.load(f)
            labels.append(label)

    with open(output, "w") as f :
        json.dump(labels, f)
        