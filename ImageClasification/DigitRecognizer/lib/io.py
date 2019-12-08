import os

# Function that creates a folder if doesn't exist
def createFolder(path_save, kwargs):
    if path_save == '': # If no arguments are given for path_save create default one
        path_save = os.getcwd()  + '/results'

    base_path_save_url = path_save
    i=0
    path_save = base_path_save_url + str(i)
    while(os.path.exists(base_path_save_url + str(i))):
        i += 1
        path_save = base_path_save_url + str(i)

    try:
        if not os.path.exists(path_save):
            os.makedirs(path_save)
    except OSError:
        print ('Error: Creating directory. ' +  path_save)
    
    # Save command executed
    f = open(os.path.join(path_save, "command_executed.txt"),"w+")
    for s in kwargs.keys():
        f.write(s + ': ' + str(kwargs[s]) + '\n')
    f.close()

    return path_save