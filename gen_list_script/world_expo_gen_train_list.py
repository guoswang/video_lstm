from TensorflowToolbox.utility import file_io


video_analysis_train_list = "/Users/Geoff/Documents/my_git/video_analysis/file_list/world_expo_train_list1.txt"

def file_list_to_list(file_list, unroll_num):
    start_index = unroll_num
    end_index = len(file_list)
    new_list = list()

    for i in range(len(file_list) - unroll_num):
        i += unroll_num
        curr_list = " ".join(file_list[i - unroll_num : i])
        new_list.append(curr_list)

    return new_list 

def reorder_str(file_name):

    f_l = file_name.split(" ")
    new_f_l = list()

    for i in range(3):
        new_f_l += f_l[i::3]
        
    return " ".join(new_f_l)


if __name__ == "__main__":
    unroll_num = 5

    for i in range(2):
        if i == 0:
            fiie_list_name = "/Users/Geoff/Documents/my_git/video_analysis/file_list/world_expo_train_list1.txt"
            save_file_name = "../file_list/world_expo_train_list1.txt"
        else:
            fiie_list_name = "/Users/Geoff/Documents/my_git/video_analysis/file_list/world_expo_test_list1.txt"
            save_file_name = "../file_list/world_expo_test_list1.txt"

        file_list = file_io.read_file(video_analysis_train_list)
        file_list.sort()
        unrolled_list = list()

        for i, f in enumerate(file_list):
            if i == 0:
                start_video = f.split("/")[0][:6]
                start_i = i
            else:
                video = f.split("/")[0][:6]
                if start_video != video or i == len(file_list) - 1:
                    new_list = file_list_to_list(file_list[start_i:i], unroll_num)
                    unrolled_list += new_list
        unrolled_list = [reorder_str(f) for f in unrolled_list]
        file_io.save_file(unrolled_list, save_file_name)

