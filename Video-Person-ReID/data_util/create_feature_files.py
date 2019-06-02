from os import listdir, mkdir
from os.path import join, split, isfile, isdir


image_sets = [
    #'train',
    'query',
    'test',
]

dummys = [
    '',
    #'_dummy',
]

features = [
    'keypoint',
]

aic_track2_dir = '/path_to_aic19-track2-reid/'

for image_set in image_sets:
    for dummy in dummys:
        image_path = aic_track2_dir + 'image_%s_deepreid%s' % (image_set, dummy)
        for feature in features:
            print((image_set, dummy, feature))
            feature_path = aic_track2_dir + '%s_%s_deepreid%s' % (feature, image_set, dummy)
            mkdir(feature_path)

            feature_file = aic_track2_dir + '%s-%s.txt' % (feature, image_set)
            lines = []
            with open(feature_file, 'r') as f:
                lines = f.readlines()

            pids = [f for f in listdir(image_path) if isdir(join(image_path, f))]
            pids.sort()
            for pid in pids:
                print(pid)
                pid_path = join(feature_path, pid)
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
                        feature_file = join(cid_path, imgname+'.txt')
                        with open(feature_file, 'w') as file:
                            file.write(lines[int(imgid)-1])
