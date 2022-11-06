import os, sys, shutil, pprint

def main():

    remove_path = sys.argv[1]
    print remove_path
    outfile = sys.argv[2]
    print outfile
    shutil.rmtree(remove_path)
    dir_of_trash_path = os.path.dirname(remove_path)

    print remove_path+'was removed.'
    print 'Remaining files:'
    pprint.pprint(os.listdir(dir_of_trash_path))
    txt_f = open(outfile, 'w')
    # txt_f.write(os.listdir(dir_of_trash_path))
    txt_f.write(remove_path+'was removed.')
    txt_f.close()
    # outfile = txt_f

if __name__ == '__main__':
    main()



