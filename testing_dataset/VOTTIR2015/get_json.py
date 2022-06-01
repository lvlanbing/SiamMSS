import os
import json
from collections import defaultdict
from defusedxml.minidom import parse

root_dir = "vottir-2015/"


def get_dict(dir_path):
    # print(dir_path)
    anno_dir = dir_path + "/anno/"
    img_dir = dir_path + "/img/"
    anno_dir_txt = dir_path

    dir_dict = {
        "video_dir": dir_path,
        "init_rect": [],
        "img_names": [],
        "gt_rect": [],
        "camera_motion": [],
        "illum_change": [],
        "motion_change": [],
        "size_change": [],
        "occlusion": []
    }

    img_list = os.listdir(img_dir)
    img_list.sort()
    for img_name in img_list:
        dir_dict["img_names"].append(dir_path + "/img/" + img_name)



    file_lines = open(anno_dir_txt + '/groundtruth.txt')
    file_lines2 = open(anno_dir_txt + '/camera_motion.tag')
    file_lines3 = open(anno_dir_txt + '/motion_change.tag')
    file_lines4 = open(anno_dir_txt + '/size.tag')
    file_lines5 = open(anno_dir_txt + '/occlusion.tag')
    line = file_lines.readline()
    line2 = file_lines2.readline()
    line3 = file_lines3.readline()
    line4 = file_lines4.readline()
    line5 = file_lines5.readline()
    while 1:
        x1, y1, x2, y2, x3, y3, x4, y4 = line.split(",")
        dir_dict["gt_rect"].append([(float(x1)), (float(y1)), (float(x2)), (float(y2)), (float(x3)), (float(y3)), (float(x4)), (float(y4))])
        line = file_lines.readline()

        camera = int(line2)
        dir_dict["camera_motion"].append(camera)
        line2 = file_lines2.readline()

        motion = int(line3)
        dir_dict['motion_change'].append(motion)
        line3 = file_lines3.readline()

        size = int(line4)
        dir_dict['size_change'].append(size)
        line4 = file_lines4.readline()

        occlusion = int(line5)
        dir_dict['occlusion'].append(occlusion)
        line5 = file_lines5.readline()
        # print(line)
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

    with open('VOTTIR2015.json', 'w') as js:
        json.dump(obj=ptb_tir, fp=js)
    return


if __name__ == "__main__":
    main()
