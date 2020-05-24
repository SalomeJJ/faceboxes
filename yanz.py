#path = '/home/lxg/codedata/widerFace/wider_face_split/'
path = 'D:/file/pyproject/faceboxes/widerface/wider_face_split/'

with open(path+'wider_face_val_bbx_gt.txt') as f:
    lines = f.readlines()
    nums = len(lines)

output = open(path+'val_list.txt', 'w')
#从数据集的label文件中提取所需的信息
for i in range(nums):
    # if i == 10:
    #     break

    line = lines[i]
    if 'jpg' not in line:
        continue
    im_name = line.strip()
    #face_num = int(lines[i+1].strip())
    #im_name = im_name

    # print(i)    
    output.writelines(im_name+'\n')
    #output.writelines('widerface/WIDER_val/images/' + im_name+'\n')

# create aflw data
# with open('label.txt') as f:
#     lines = f.readlines()
# output = open('label_path.txt', 'w') 
# for line in lines:
#     output.writelines('data/all/' + line)


