import os,shutil
from tqdm import tqdm

ori_root_list = [r'C:\Users\liuji\Downloads\jierui24_final']
dst_root = r'C:\Users\liuji\Downloads\jierui24_final_INF'
def cpoy_seq(source_dir,target_dir):
    # 定义原始目录和目标目录

    # 如果目标目录不存在，创建它
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历原始目录中的所有内容
    for root, dirs, files in os.walk(source_dir):
        # 确定当前目录的相对路径
        relative_path = os.path.relpath(root, source_dir)

        # 构建目标目录路径
        target_path = os.path.join(target_dir, relative_path)

        # 如果目标目录不存在，创建它
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        # 复制文件
        for file in files:
            shutil.copy(os.path.join(root, file), os.path.join(target_path, file))


def transform(ori_root_list,dst_root,type=['train']):
    for idx,prefix in enumerate(ori_root_list):
        dst_dir=os.path.join(dst_root,type[idx])
        os.makedirs(dst_dir,exist_ok=True)
        seq_path_list=[os.path.join(prefix,item,'infrared') for item in os.listdir(prefix)]
        for seq in tqdm(seq_path_list):
            path_parts = os.path.normpath(seq).split(os.sep)
            # 检查路径是否有足够的部分
            if len(path_parts) >= 2:
                second_last_dir = path_parts[-2]
            else:
                second_last_dir = None
            cpoy_seq(seq,os.path.join(dst_dir,second_last_dir))

transform(ori_root_list,dst_root)








