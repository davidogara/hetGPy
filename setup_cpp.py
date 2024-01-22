import cppimport
import subprocess
import os

if __name__ == "__main__":
    os.chdir('hetgpy')
    print(f"Importing from: {os.getcwd()}")
    subprocess.run("python -m cppimport build",shell=True)