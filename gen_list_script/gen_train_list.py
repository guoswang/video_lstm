import os
import random
from TensorflowToolbox.utility import file_io
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "../../data"

    file_list_dir = "../file_list/"
    data_ext = "_resize.jpg"
    label_ext = "_resize.desmap"
    mask_ext = "_msk_resize.npy"    

    unroll_num = 5

    cam_dir_list = file_io.get_dir_list(data_dir)
    train_list = list()
    test_list = list()
    full_file_list = list()
    for cam_dir in cam_dir_list:
        video_list = file_io.get_listfile(cam_dir, ".avi")
        
        data_list = list()
        mask_list = list()
        for file_name in video_list:
            data_dir_name = file_name.replace(".avi", "")
            curr_data_list = file_io.get_listfile(data_dir_name, data_ext)

            mask_list += [data_dir_name + mask_ext] * len(curr_data_list)
            data_list += curr_data_list


        full_file_list += [d + " " + d.replace(data_ext, label_ext) + " " + m \
                        for d, m in zip(data_list, mask_list)]
        
        #partition = 0.7
        #train_data_len = int(len(data_list) * partition)

        #random.shuffle(data_list)
        #train_data = data_list[:train_data_len]
        #test_data = data_list[train_data_len:]

        #train_list += [d + " " + d.replace(data_ext, label_ext) for d in train_data]
        #test_list += [d + " " + d.replace(data_ext, label_ext) for d in test_data]
    
    for i, f in enumerate(full_file_list):
        f_l = f.split(" ")
        f_ll = [f_l[0]] * unroll_num + [f_l[1]] * unroll_num \
                            + [f_l[2]] * unroll_num
        full_file_list[i] = " ".join(f_ll)

    train_file_list_name = 'train_list1.txt'
    train_len = int(len(full_file_list) * 0.7)

    train_file_list = full_file_list[:train_len]
    file_io.save_file(train_file_list, file_list_dir + train_file_list_name, True)

    test_file_list_name = 'test_list1.txt'
    test_file_list = full_file_list[train_len:]
    file_io.save_file(test_file_list, file_list_dir + test_file_list_name, True)


