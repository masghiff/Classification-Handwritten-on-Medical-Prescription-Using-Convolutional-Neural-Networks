from PIL import Image
import os, sys

path = "Data/validation"
dirs = os.listdir( path )

def resize():
    for dr in dirs:
        dirbaca=path+"/"+dr
        print(dirbaca)
        dirs2=os.listdir(dirbaca)
        for item in dirs2 :
            fcheck=path+"/"+dr+"/"+item
            if os.path.isfile(fcheck):
                im = Image.open(fcheck)
                f, e = os.path.splitext(fcheck)
                imResize = im.resize((200,150), Image.ANTIALIAS)
                imResize.save("resized"+path+"/"+dr+"/"+item, 'JPEG', quality=90)
resize()