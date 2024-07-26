import gdown
import threading
import os

def download_file(url, output_path):
    gdown.download(url, output_path)


if __name__ == '__main__':

    train_part_0 = '1EdWafDjZG3VUo7CXTERx7fTx4IqWWNvC'
    train_part_1 = '1Kyn8seqBiU8drJ8uyS2RSgeefLatUCzi'
    train_part_2 = '1FaxOJJu6hbPjkab5dtCu-QWxMVDEcj5T'
    train_part_3 = '1S6G-knZUG2V0Jhobseb_2-_V5n2jhVMS'
    train_part_4 = '1br0-EZfh4aTY5E66_m-PXR9i0v1IA1-8'
    train_part_5 = '17ZGVQ8fGVv9FlEQiLy-j_M9r4NlABY3i'
    train_part_6 = '10vPLCDjv4AC35TkdwAkgUrrAvAhIQJ9v'
    train_part_7 = '1vebIvddC4UG78YSpbGTJlr-Hl1ScZeGV'
    train_part_8 = '1KmVoAofGWnwBJ0EqClYqWNBzENMA8riE'
    train_part_9 = '1N4eiw0DV-HrxWSkhRDmBV6skGTiOUz5B'

    train_parts = [train_part_0, train_part_1, train_part_2, train_part_3, train_part_4,
                   train_part_5, train_part_6, train_part_7, train_part_8, train_part_9]

    mem_path = 'MemoryMaze/MemoryMaze_data/9x9/'

    isExist = os.path.exists(mem_path)
    if not isExist:
        os.makedirs(mem_path)
    
    threads = {}
    for i in range(10):
        data_path = f'MemoryMaze/MemoryMaze_data/9x9/train_mem_maze_9x9_part{i}.zip'
        file_exist = os.path.exists(data_path)
        if not file_exist:
            print(f"File train_mem_maze_9x9_part{i}.zip does not exist. Downloading...")
            threads[i] = threading.Thread(target=download_file, 
                                          args=(f"https://drive.google.com/uc?id={train_parts[i]}&confirm=t", 
                                                f'MemoryMaze/MemoryMaze_data/9x9/train_mem_maze_9x9_part{i}.zip'))
        else:
            print(f"File train_mem_maze_9x9_part{i}.zip exists. It's good!")
            
    for i in range(10):
        if threads.get(i) is not None:
            threads[i].start()

    for i in range(10):
        if threads.get(i) is not None:
            threads[i].join()