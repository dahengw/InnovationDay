import os


def travel_father_dir(path):
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            rename_files_of_dir(os.path.join(path, dir))


def rename_files_of_dir(dir_path):
    name = 1
    for file in os.listdir(dir_path):
        os.rename(os.path.join(dir_path, file), os.path.join(dir_path, str(name)+'.dcm'))
        name += 1


def main():
    path = './Data/'
    travel_father_dir(path)


if __name__ == '__main__':
    main()
