def save_density_map(file_list, density_map_list, unroll_num, desmap_scale):
    j = 0
    new_density_map_list = list()
    batch, h, w, c = density_map_list[0][-1].shape
    for b in range(batch):
        for i in range(unroll_num):
            new_file_name = file_list[j].replace(".jpg", ".infer_desmap")
            desmap = density_map_list[i][-1][b,:,:,:] / desmap_scale
            desmap.tofile(new_file_name)
            j += 1

    
