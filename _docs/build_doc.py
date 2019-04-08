import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
print(os.system('sh make.sh'))