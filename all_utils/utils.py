import os 
import random 
import shutil
import time 
def fix_path(path):
    path = str(path)
    new_path = path.replace('\\\\','/') 
    return new_path.replace('\\','/')


enable_decorator = False

# Decorator function
def timeit(method):
    def timed(*args, **kw):
        if enable_decorator:
            ts = time.perf_counter()  # Record start time
            result = method(*args, **kw)  # Call the original function
            te = time.perf_counter()  # Record end time
            print(f"'{method.__name__}'{te - ts:.2f} sec")
        else:
            result = method(*args, **kw)
        return result  # Return the result of the original function
    return timed

@timeit
def create_new_source_folder(source_folder, num_save, save_file_path):
    # Taking num_save random elements from the list
    my_list = os.listdir(source_folder)
    random_elements = random.sample(my_list, num_save)
    # sample 7 unique element 
    for i in random_elements:
        shutil.copy(os.path.join(source_folder, i), 
                    os.path.join(save_file_path, i))
