from TensorflowToolbox.utility import file_io

trancos_data_path = "/media/dog/data/TranCos/TranCos/TranCos/"

def file_list_to_list(file_list, unroll_num):
    start_index = unroll_num
    end_index = len(file_list)
    new_list = list()

    for i in range(len(file_list) - unroll_num):
        i += unroll_num
        curr_list = " ".join(file_list[i - unroll_num : i])
        new_list.append(curr_list)

    return new_list 

def file_list_to_train_list(file_list_name):
    data_list = file_io.read_file(file_list_name)
    data_list.sort()

    data_list = [trancos_data_path + t for t in data_list]

    data_list = file_list_to_list(data_list, unroll_num)

    desmap_list = [d.replace(data_ext, desmap_ext) for d in \
                        data_list]
    
    mask_list = [d.replace(data_ext, mask_ext) for d in \
                        data_list]

    file_list = [i + " " + d + " "  + m for i, d, m in \
                zip(data_list, desmap_list, mask_list)]

    return file_list 

if __name__ == "__main__":
    data_ext = ".jpg"
    desmap_ext = ".desmap"
    mask_ext = "_mask.npy"
    unroll_num = 5

    file_list_dir = "../file_list/"
    save_train_file_name = "trancos_train_list1.txt"
    save_test_file_name = "trancos_test_list1.txt"

    train_file = file_list_to_train_list("../file_list/trancos_org_trainval.txt")
    test_file = file_list_to_train_list("../file_list/trancos_org_test.txt")

    file_io.save_file(train_file, file_list_dir + save_train_file_name, True)
    file_io.save_file(test_file, file_list_dir + save_test_file_name, True)
