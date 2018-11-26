import os
import subprocess
import sys

sys.path.insert(0, "..")


if __name__ == '__main__':

    print("Parent Process: ", os.getpid())

    run_files = sys.argv[1]
    files = [line.rstrip('\n') for line in open(run_files)]

    threads = int(sys.argv[2])

    def run(file):
        result = subprocess.run(file.split(" "), stdout=subprocess.PIPE)
        print(result.stdout.decode('utf-8').rstrip("\n\r"))

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=threads) as executor:

        for f in files:
            a = executor.submit(run, f)

