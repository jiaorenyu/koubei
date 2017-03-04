import os

def list_files(basedir):
    files = os.listdir(basedir)
    files = map(lambda fn : os.path.join(basedir, fn), files)

    return files
    
