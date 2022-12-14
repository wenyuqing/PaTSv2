
import re
import matplotlib.pyplot as plt
import os.path as osp

fullpath = osp.abspath('./logs/2ds_add.txt')
fullpath2 = osp.abspath('./logs/2d_cls.txt')
# mode = {'Loss'}
mode = {'Loss'}

filedir, filename = osp.split(fullpath)
count = 0
Loss, Loss2, x ,y= [], [], [],[]
with open(fullpath, 'r') as f:
    while True:
        line = f.readline()
        if 'INFO 12 views inference' in line:
            break
        if line == '':
            break
        #if not ('INFO Train:' in line and '[7300/7330]' in line):
        if not 'the 19760 test videos:' in line:
            continue
        count += 1
        line = line.replace(' ', '').replace('\t', '')
        #pattern = re.compile(r'[tot_loss]\w*.\w+[(](\w*.\w+)[)]')
        # pattern = re.compile(r'[tot_loss]\w*.\w+[(](\w*.\w+)[)]')
        pattern = re.compile(r'[the19760testvideos:](\d*.\d+)[%]')
        find_list = pattern.findall(line)
        if mode == {'Loss'}:
            Loss.append(float(find_list[0]))
        elif mode == {'Loss', 'CLoss', 'TLoss'}:
            Loss.append(float(find_list[0]))
            # CLoss.append(float(find_list[1]))
            # TLoss.append(float(find_list[2]))
        x.append(count)
        print(Loss)
f.close()
count = 0
with open(fullpath2, 'r') as f:
    while True:
        line = f.readline()
        if 'INFO 12 views inference' in line:
            break
        if line == '':
            break
        #if not ('INFO Train:' in line and '[3650/3665]' in line):
        if not 'the 19760 test videos:' in line:
            continue
        count += 1
        line = line.replace(' ', '').replace('\t', '')
        #pattern = re.compile(r'[tot_loss]\w*.\w+[(](\w*.\w+)[)]')
        pattern = re.compile(r'[the19760testvideos:](\d*.\d+)[%]')
        find_list = pattern.findall(line)
        if mode == {'Loss'}:
            Loss2.append(float(find_list[0]))
        elif mode == {'Loss', 'CLoss', 'TLoss'}:
            Loss2.append(float(find_list[0]))
            # CLoss.append(float(find_list[1]))
            # TLoss.append(float(find_list[2]))
        y.append(count)
        print(Loss2)
f.close()

pngName = filename.split('.')[0]

if mode == {'Loss'}:
    plt.plot(x, Loss, color='red', marker='o', linewidth=0.5, markersize=1)
    plt.plot(y, Loss2, color='green', marker='o', linewidth=0.5, markersize=1)
    plt.legend(labels=('2ds_add','2d_cls'))
elif mode == {'Loss', 'CLoss', 'TLoss'}:
    plt.plot(x, Loss, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=1)
    # plt.plot(x, CLoss, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=1)
    # plt.plot(x, TLoss, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=1)
    plt.legend(labels=('Loss', 'CLoss', 'TLoss'))

plt.savefig(osp.join(filedir, pngName))
plt.show()