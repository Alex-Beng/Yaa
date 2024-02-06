# open file named bb_times
# each line looks like "capture_time: xx" xx in milliseconds
# read the file and compute the average capture time

import os
import sys

def main():
    # open file
    try:
        file = open("./bb_times", "r")
    except:
        print("Error: cannot open file")
        sys.exit(1)

    # read file
    lines = file.readlines()
    file.close()

    # compute average
    total = 0
    count = 0
    for line in lines:
        if line.startswith("capture time: "):
            total += int(line.split()[-1])
            count += 1

    # print average
    if count > 0:
        print("Average capture time: " + str(total / count) + " ms")
    else:
        print("No capture time recorded")

def main_yap():
    # open file
    try:
        file = open("./bb_times_yap", "r", encoding="utf-8")
    except:
        print("Error: cannot open file")
        sys.exit(1)

    # read file
    lines = file.readlines()
    file.close()

    # compute average
    total = 0
    count = 0
    min_time = 9999
    max_time = -1
    for line in lines:
        # if line.startswith("capture time: "):
        if "capture time" in line:
            curr_cap_time_s = line.split()[-1][:-2]
            curr_cap_time = float(curr_cap_time_s)
            total += curr_cap_time
            count += 1
            min_time = min(min_time, curr_cap_time)
            max_time = max(max_time, curr_cap_time)

    # print average
    if count > 0:
        print("Average capture time: " + str(total / count) + " ms")
    else:
        print("No capture time recorded")


if __name__ == "__main__":
    main()
    main_yap()
