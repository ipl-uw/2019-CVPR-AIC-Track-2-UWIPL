import os,re
rawlogpath = './test.log'
newlogpath = './newtest.log'
f = open(rawlogpath,'r')
cnt = 0
attr = []

for line in f:
    if line[0:2] == '[[':
        tmp = []
        for x in re.split('\[|\n| ',line):
            if x!='':
                tmp.append(x)
    elif line[0:4]=='test':
        pass
    elif line[0:4]=='data':
        break
    elif line[-3:] == ']]\n':
        for x in re.split('\]|\n| ',line):
            if x!='':
                tmp.append(x)
        attr.append(tmp)
    else:
        for x in re.split('\n| ',line):
            if x!='':
                tmp.append(x)   
    cnt+=1
f.close()

f = open(newlogpath,'w')
for cnt in range(int(len(attr)/3)):
    f.write('test %s image \n'%(cnt))
    for x in attr[cnt*3]:
        f.write(x+' ')
    f.write('\n')
    
    for x in attr[cnt*3+1]:
        f.write(x)
    f.write('\n')
    
    for x in attr[cnt*3+2]:
        f.write(x)
    f.write('\n')

print(len(attr))
f.close()
    