import os
import re
import subprocess
import sys


if __name__ == '__main__':

    print("Parent Process: ", os.getpid())

    folder = sys.argv[1]
    threads = int(sys.argv[2])
    files = [f for f in os.listdir(folder) if re.match(r'.*.run', f)]

    def run(file):
        path_to_file = os.path.join(folder, file)
        result = subprocess.run(['python', 'execute.py', path_to_file, folder], stdout=subprocess.PIPE)
        print(result.stdout.decode('utf-8').rstrip("\n\r"))

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=threads) as executor:

        for f in files:
            a = executor.submit(run, f)

