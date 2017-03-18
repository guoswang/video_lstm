from TensorflowToolbox.utility import file_io
import sys

if __name__ == "__main__":
    file_1 = sys.argv[1]
    file_2 = sys.argv[2]

    file_list1 = file_io.read_file(file_1)
    file_list2 = file_io.read_file(file_2)
    for f in file_list1:
        assert(f not in file_list2)
    
    for f in file_list2:
        assert(f not in file_list1)
