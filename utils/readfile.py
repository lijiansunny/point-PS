import os

file_dir = "../datasets/DiLiGenT/pmsData"
for root, dirs, files in os.walk(file_dir, topdown=False):
    print("当前目录路径:", root)
print("当前目录下所有子目录:", dirs)
dirs1=[dir[:-3] for dir in dirs]
print(dirs1)
with open('objects.txt','w') as f:
   str='\n'
   f.write(str.join(dirs1))
   f.close()


