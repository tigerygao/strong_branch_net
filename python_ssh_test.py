
import subprocess

subprocess.call(["touch", "ssh_from_py_test.txt"]);
subprocess.call(["git", "add", "."]);
subprocess.call(["git", "commit", "-m", "'git via ssh via python'"]);
subprocess.call(["git", "push"]);





