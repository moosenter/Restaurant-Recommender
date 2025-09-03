import os
from os.path import dirname, abspath
import sys
lib_path = dirname(dirname(abspath(__file__)))
sys.path.append(lib_path)
os.chdir(lib_path)