from TensorflowToolbox.utility import file_io
import numpy as np
import cv2

ucsd_label_path = "/media/dog/data/UCSD/gtDensities/"
ucsd_image_path = "/media/dog/data/UCSD/images/"
mask_file_name = "/media/dog/data/UCSD/Mask/mask227.npy"


#ucsd_label_path = "/Users/Geoff/Documents/my_git/video_analysis/data_process_script/ucsd/desmap/"
#ucsd_image_path = "/Users/Geoff/Documents/my_git/video_analysis/data_process_script/ucsd/image/"
#mask_file_name = "/Users/Geoff/Documents/my_git/video_analysis/data_process_script/ucsd/mask/mask227.npy"
#
def file_list_to_list(file_list, unroll_num):
    start_index = unroll_num
    end_index = len(file_list)
    new_list = list()

    for i in range(len(file_list) - unroll_num):
        i += unroll_num
        curr_list = " ".join(file_list[i - unroll_num : i])
        new_list.append(curr_list)

    return new_list 

def file_list_to_train_list(file_list):
    file_list = [t + " " + t.replace(image_ext, desmap_ext).\
    replace(ucsd_image_path, ucsd_label_path) + " " \
    + mul_mask_name for t in file_list]
    
    return file_list

def reverse_copy_list(image_list):
    reverse_image_list = list()
    for f in image_list:
        f_l = f.split(" ")
        f_l.reverse()
        f_s = " ".join(f_l)
        reverse_image_list.append(f_s)

    return reverse_image_list

if __name__ == "__main__":
    image_ext = ".jpg"
    desmap_ext = ".desmap"
    mask_ext = "_mask.npy"
    file_list_dir = "../file_list/"
    unroll_num = 5

    mul_mask_name = " ".join([mask_file_name] * 5)

    save_train_file_name = "ucsd_train_list1.txt"
    save_test_file_name = "ucsd_test_list1.txt"
    
    image_list = file_io.get_listfile(ucsd_image_path, image_ext)
    image_list.sort()
    image_list = file_list_to_list(image_list, unroll_num) 

    reverse_list = reverse_copy_list(image_list)

    file_list = file_list_to_train_list(image_list)

    reverse_file_list = file_list_to_train_list(reverse_list)
    
    train_list = reverse_file_list[600:1400]
    #file_list[600:1400] + reverse_file_list[600:1400]

    test_list = file_list[0:600] + file_list[1400:]
    
    file_io.save_file(train_list, file_list_dir + save_train_file_name, True)
    file_io.save_file(test_list, file_list_dir + save_test_file_name, True)
    
    
