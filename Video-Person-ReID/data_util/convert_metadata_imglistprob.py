from os import listdir, mkdir
from os.path import join, split, isfile, isdir


conversions = [
    ('./track2-gallery-query-metadata-v2m100/test-prob-v2m100.log',
     './track2-gallery-query-metadata-v2m100/prob_v2m100.txt',
     './track2-gallery-query-metadata-v2m100/imglist_v2m100.txt'),
]

img_gline = {}
with open('test_track.txt', 'r') as f:
    for gg, line in enumerate(f):
        g_line = gg+1
        print(g_line)

        imgs = line.replace("\n", "").strip().split(" ")
        for i, img in enumerate(imgs):
            img_gline[img] = g_line

img_qline = {}
with open('query_track.txt', 'r') as f:
    for qq, line in enumerate(f):
        q_line = qq+1
        print(q_line)

        imgs = line.replace("\n", "").strip().split(" ")
        for i, img in enumerate(imgs):
            img_qline[img] = q_line
        assert int(imgs[0].replace('.jpg','')) == q_line # make sure is ordered


for raw_filename, prob_filename, imglist_filename in conversions:
    metadatas = []
    with open(raw_filename, 'r') as f:
        buf = ''
        i = 0
        for line in f:
            line = line.strip()
            if i % 4 == 0:
                metadatas.append([])
                i += 1
            else:
                buf = buf + ' ' + line
                #if line[-2:] != ']]':
                #    continue
                #print(buf)
                l = buf.rfind('[[')
                r = buf.find(']]')
                if l == -1 and r == -1:
                    metadatas[-1].append(buf.strip())
                elif l < r:
                    metadatas[-1].append(buf[l+2:r].strip())
                else:
                    print('invalid buf: ' + buf)
                buf = ''
                i += 1
    if len(metadatas[-1]) == 0:
        metadatas = metadatas[:-1]
    print('images in metadatas: %d' % len(metadatas))

    prob_filename_test = prob_filename[:-4] + '_test.txt'
    imglist_filename_test = imglist_filename[:-4] + '_test.txt'
    f_prob = open(prob_filename_test, 'w')
    f_imglist = open(imglist_filename_test, 'w')
    i = 0
    for img in img_gline:
        f_prob.write('%d/%d image\n' % (i, len(img_gline)))
        for metadata in metadatas[img_gline[img]-1 + 1052]:
            f_prob.write(metadata+'\n')
        f_imglist.write(img+'\n')
        i+=1
    f_prob.close()
    f_imglist.close()

    prob_filename_query = prob_filename[:-4] + '_query.txt'
    imglist_filename_query = imglist_filename[:-4] + '_query.txt'
    f_prob = open(prob_filename_query, 'w')
    f_imglist = open(imglist_filename_query, 'w')
    i = 0
    for img in img_qline:
        f_prob.write('%d/%d image\n' % (i, len(img_qline)))
        for metadata in metadatas[img_qline[img]-1]:
            f_prob.write(metadata+'\n')
        f_imglist.write(img+'\n')
        i+=1
    f_prob.close()
    f_imglist.close()







