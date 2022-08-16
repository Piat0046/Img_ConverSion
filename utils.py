import os

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            print('Already directory exist')
    except OSError:
        print('Error: Creating directory. '+ directory)