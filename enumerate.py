def get_next_elec(cnt_coords, elec_coords, direction, median_dist):
    if direction == "left":
        x = -median_dist
        y = 0
    elif direction == "up":
        x = 0
        y = -median_dist
    elif direction == "right":
        x = median_dist
        y = 0
    elif direction == "down":
        x = 0
        y = median_dist
    if len(cnt_coords)>2:
        next_elec = (elec_coords[-1][0]+x, elec_coords[-1][1]+y)
        tree = spatial.KDTree(cnt_coords)
        next_elec_index = tree.query(next_elec)
        next_elec_coord = cnt_coords[next_elec_index[1]]
        elec_coords.append(next_elec_coord)
        cnt_coords.remove(next_elec_coord)
    else:
        next_elec = (elec_coords[-1][0], elec_coords[-1][1])
        tree = spatial.KDTree(cnt_coords)
        next_elec_index = tree.query(next_elec)
        next_elec_coord = cnt_coords[next_elec_index[1]]
        elec_coords.append(next_elec_coord)
        cnt_coords.remove(next_elec_coord)
        #distances = []
        #for i in range(len(cnt_coords)):
            #distances.append(dist_between_points(cnt_coords[i][0], cnt_coords[i][1], elec_coords[-1][0], elec_coords[-1][1]))
            #next_elec_index = np.argmin(distances)
            #next_elec_coord = cnt_coords[next_elec_index]
            #elec_coords.append(next_elec_coord)
            #cnt_coords.remove(next_elec_coord)
    return cnt_coords, elec_coords

def enumerate_electrodes(cnt_coords, center):
    #dist = dist_between_points(cnt_coords[0][0], cnt_coords[0][1], cnt_coords[1][0], cnt_coords[1][1])
    cnt_x = [cnt_coords[i][0] for i in range(len(cnt_coords))]
    cnt_y = [cnt_coords[i][1] for i in range(len(cnt_coords))]
    mean_x = np.mean(cnt_x)
    mean_y = np.mean(cnt_y)
    bottom_right = cnt_coords[np.argmax([cnt_x[i]+cnt_y[i] for i in range(len(cnt_x))])]
    bottom_left = cnt_coords[np.argmax([np.abs(cnt_x[i]+cnt_y[i]) for i in range(len(cent_x))])]
    spiral_dir = spiral_direction(cnt_coords)
    elec_coords = []
    median_dist = dist_between_points(cnt_coords[0][0], cnt_coords[0][1], cnt_coords[1][0], cnt_coords[1][1])
    if spiral_dir == "right":
        last_elec = bottom_right
        elec_coords.append(last_elec)
        cnt_coords.remove(last_elec)
        #while len(cnt_coords)>2:
        leftmost = np.amin([cnt_coords[j][0] for j in range(len(cnt_coords))])
        highest = np.amin([cnt_coords[j][1] for j in range(len(cnt_coords))])
        rightmost = np.amax([cnt_coords[j][0] for j in range(len(cnt_coords))])
        lowest = np.amax([cnt_coords[j][1] for j in range(len(cnt_coords))])
        while elec_coords[-1][0] != leftmost:
            cnt_coords, elec_coords = get_next_elec(cnt_coords, elec_coords, "left", median_dist)
            #break 
        while elec_coords[-1][1] != highest:
            cnt_coords, elec_coords = get_next_elec(cnt_coords, elec_coords, "up", median_dist)
            #break
        while elec_coords[-1][0] != rightmost:
            cnt_coords, elec_coords = get_next_elec(cnt_coords, elec_coords, "right", median_dist)
            #break
        while elec_coords[-1][1] != lowest:
            cnt_coords, elec_coords = get_next_elec(cnt_coords, elec_coords, "down", median_dist)
            #break
        while elec_coords[-1][0] != leftmost:
            cnt_coords, elec_coords = get_next_elec(cnt_coords, elec_coords, "left", median_dist)
            #break
        
    return elec_coords
