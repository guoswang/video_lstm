def save_density_map(file_list, density_map_list):
    j = 0
    new_density_map_list = list()
    batch, h, w, c = density_map_list[0][-1].shape
    for b in range(batch):
        for i in range(5):
            new_file_name = file_list[j].replace(".jpg", ".infer_desmap")
            density_map_list[i][-1][b,:,:,:].tofile(new_file_name)
            j += 1

    
