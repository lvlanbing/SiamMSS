import os
import json
from collections import defaultdict
from defusedxml.minidom import parse

root_dir = "sequences/"


def get_dict(dir_path):
    anno_dir = dir_path + "/anno/"
    img_dir = dir_path + "/img/"
    anno_dir_txt = dir_path

    dir_dict = {
        "video_dir": dir_path,
        "init_rect": [],
        "img_names": [],
        "gt_rect": [],
        # "camera_motion": [],
        # "illum_change": [],
        # "motion_change": [],
        # "size_change": [],
        # "occlusion": []
    }

    img_list = os.listdir(img_dir)
    img_list.sort()
    for img_name in img_list:
        dir_dict["img_names"].append(dir_path + "/img/" + img_name)

    # xml_list = os.listdir(anno_dir)
    # xml_list.sort()
    # for xml_file in xml_list:
    #     with parse(anno_dir + xml_file) as tree:
    #         root = tree.documentElement
    #
    #         x_min = float(root.getElementsByTagName("xmin")[0].childNodes[0].nodeValue)
    #         y_min = float(root.getElementsByTagName("ymin")[0].childNodes[0].nodeValue)
    #         x_max = float(root.getElementsByTagName("xmax")[0].childNodes[0].nodeValue)
    #         y_max = float(root.getElementsByTagName("ymax")[0].childNodes[0].nodeValue)
    #         dir_dict["gt_rect"].append([x_min, y_min, x_max, y_min, x_min, y_max, x_max, y_max])
    #
    #         occ = int(root.getElementsByTagName("occluded")[0].childNodes[0].nodeValue)
    #         dir_dict["occlusion"].append(occ)

    file_lines = open(anno_dir_txt + '/groundtruth_rect.txt')
    line = file_lines.readline()
    while 1:
        x_min, y_min, w, h = line.split(',')
        x_min, y_min, w, h = float(x_min), float(y_min), float(w), float(h)
        x_max = float(x_min) + float(w)
        y_max = float(y_min) + float(h)
        # dir_dict["gt_rect"].append([x_min, y_min, x_max, y_min, x_min, y_max, x_max, y_max])
        dir_dict["gt_rect"].append([x_min, y_min, w, h])
        line = file_lines.readline()
        if not line:
            break
    file_lines.close()

    dir_dict["init_rect"] = dir_dict["gt_rect"][0]

    return dir_dict


def main():
    ptb_tir = defaultdict()

    dir_list = os.listdir(root_dir)
    dir_list.sort()
    print(len(dir_list))
    for sub_dir in dir_list:
        print(sub_dir)
        try:
            ptb_tir[sub_dir] = get_dict(root_dir + sub_dir)
        except FileNotFoundError:
            continue

    with open('LSOTB.json', 'w') as js:
        json.dump(obj=ptb_tir, fp=js)
    return


if __name__ == "__main__":
    main()
