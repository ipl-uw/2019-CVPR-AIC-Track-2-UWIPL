from os import listdir, mkdir
from os.path import join, split, isfile, isdir


image_sets = [
    'query',
    'test',
]

dummys = [
    '',
    #'_dummy',
]

models = [
    'v2m100',
]

aic_track2_dir = '/path_to_aic19-track2-reid/'

for model in models:
    for image_set in image_sets:
        for dummy in dummys:
            print((model, image_set, dummy))
            # parse metadata probability from file
            metadatas = []
            with open(aic_track2_dir + 'prob_%s_%s.txt'%(model, image_set), 'r') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if i % 4 == 0:
                        metadatas.append([])
                    else:
                        l = line.rfind('[')
                        r = line.find(']')
                        if l == -1 and r == -1:
                            metadatas[-1].append(line.strip())
                        elif l < r:
                            metadatas[-1].append(line[l+1:r].strip())
                        else:
                            print('invalid line: ' + line)
            if len(metadatas[-1]) == 0:
                metadatas = metadatas[:-1]
            print('images in metadatas: %d' % len(metadatas))

            # read image filenames from file
            img_orders = {}
            with open(aic_track2_dir + 'imglist_%s_%s.txt'%(model, image_set), 'r') as f:
                for i, line in enumerate(f):
                    pos = line.find('.jpg')
                    imgid = line[pos-6:pos]
                    #print(imgid)
                    if imgid in img_orders:
                        print('duplicate images: '+imgid)
                    img_orders[imgid] = i
            print('images in image list: %d' % len(img_orders))


            image_path = aic_track2_dir + 'image_%s_deepreid%s' % (image_set, dummy)
            metadata_path = aic_track2_dir + 'metadata_%s_%s_deepreid%s' % (model, image_set, dummy)
            mkdir(metadata_path)

            pids = [f for f in listdir(image_path) if isdir(join(image_path, f))]
            pids.sort()
            for pid in pids:
                print(pid)
                pid_path = join(metadata_path, pid)
                pid_path_img = join(image_path, pid)
                mkdir(pid_path)
                cids = [f for f in listdir(pid_path_img) if isdir(join(pid_path_img, f))]
                for cid in cids:
                    cid_path = join(pid_path, cid)
                    cid_path_img = join(pid_path_img, cid)
                    mkdir(cid_path)
                    imgs = [f for f in listdir(cid_path_img) if isfile(join(cid_path_img, f)) and f[-4:] == '.jpg']
                    for img in imgs:
                        imgname = img[:-4]
                        imgid = imgname.split('_')[-1]
                        metadata_file = join(cid_path, imgname+'.txt')
                        with open(metadata_file, 'w') as file:
                            for metadata in metadatas[img_orders[imgid]]:
                                file.write(metadata+'\n')
