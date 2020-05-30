def enumerate_electrodes(cnt_coords, center):
    #dist = dist_between_points(cnt_coords[0][0], cnt_coords[0][1], cnt_coords[1][0], cnt_coords[1][1])
    cnt_x = [cnt_coords[i][0] for i in range(len(cnt_coords))]
    cnt_y = [cnt_coords[i][1] for i in range(len(cnt_coords))]
    mean_x = np.mean(cnt_x)
    mean_y = np.mean(cnt_y)
    bottom_right = cnt_coords[np.argmax([cnt_x[i]+cnt_y[i] for i in range(len(cnt_x))])]
    bottom_left = cnt_coords[np.argmax([np.abs(cnt_x[i]+cnt_y[i]) for i in range(len(cnt_x))])]
    spiral_dir = spiral_direction(cnt_coords)
    dist_to_mean = []
    distances = np.empty([len(cnt_coords), len(cnt_coords)])
    elec_coords = []
    for (x, y) in cnt_coords:
        dist_to_mean.append(dist_between_points(mean_x, mean_y, x, y))
    for i in range(len(cnt_coords)):
        for j in range(len(cnt_coords)):
            distances[i, j] = dist_between_points(cnt_coords[i][0], cnt_coords[i][1], cnt_coords[j][0], cnt_coords[j][1])
    median_dist = dist_between_points(cnt_coords[0][0], cnt_coords[0][1], cnt_coords[1][0], cnt_coords[1][1])
    if spiral_dir == "right":
            last_elec = bottom_right
            elec_coords.append(last_elec)
            for i in range(len(cnt_coords)):
                next_elec = (elec_coords[i][0]-median_dist, elec_coords[i][1])
                tree = spatial.KDTree(cnt_coords)
                next_elec_index = tree.query(next_elec)
                next_elec_coord = cnt_coords[next_elec_index[1]]
                if next_elec_coord[0] < elec_coords[i][0]:
                    elec_coords.append(next_elec_coord)
                    cnt_coords.remove(next_elec_coord)
                    continue
                else: 
                    next_elec = (elec_coords[i][0], elec_coords[i][1]-median_dist)
                    tree = spatial.KDTree(cnt_coords)
                    next_elec_index = tree.query(next_elec)
                    next_elec_coord = cnt_coords[next_elec_index[1]]
                    if next_elec_coord[1] < elec_coords[i][1]:
                        elec_coords.append(next_elec_coord)
                        cnt_coords.remove(next_elec_coord)
                        continue
                    else:
                        next_elec = (elec_coords[i][0]+median_dist, elec_coords[i][1])
                        tree = spatial.KDTree(cnt_coords)
                        next_elec_index = tree.query(elec_coords[i])
                        next_elec_coord = cnt_coords[next_elec_index[1]]
                        if next_elec_coord[0] > elec_coords[i][0]:
                            elec_coords.append(next_elec_coord)
                            cnt_coords.remove(next_elec_coord)
                            continue
                        else: 
                            next_elec = (elec_coords[i][0], elec_coords[i][1]+median_dist)
                            tree = spatial.KDTree(cnt_coords)
                            next_elec_index = tree.query(next_elec)
                            next_elec_coord = cnt_coords[next_elec_index[1]]
                            if next_elec_coord[1] > elec_coords[i][1]:
                                elec_coords.append(next_elec_coord)
                                cnt_coords.remove(next_elec_coord)
                                continue
                            else:
                                next_elec = (elec_coords[i][0]-median_dist, elec_coords[i][1]-median_dist)
                                tree = spatial.KDTree(cnt_coords)
                                next_elec_index = tree.query(next_elec)
                                next_elec_coord = cnt_coords[next_elec_index[1]]
                                if np.abs(next_elec[1]-next_elec_coord[1]) < 70:
                                    elec_coords.append(next_elec_coord)
                                    cnt_coords.remove(next_elec_coord)
                                else:
                                    break
                                break
                            break
                        break
                    break
                
            
            
                                    
    return elec_coords
