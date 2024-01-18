import os
import cppimport

PROJECT_DIR = os.getcwd()
if PROJECT_DIR.endswith('notebooks'):
    folder = '../hetgpy/'
else:
    folder = ''
cppimport.imp_from_filepath(f'{folder}matern.cpp')
cppimport.imp_from_filepath(f'{folder}gauss.cpp')