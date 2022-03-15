import os
root_dir = os.getcwd()

root_dir = os.path.dirname(root_dir)
data305_source_dir = os.path.join(root_dir, 'UNet_framework/degrain_examples/305/source')
data305_target_dir = os.path.join(root_dir, 'UNet_framework/degrain_examples/305/target')
data306_source_dir = os.path.join(root_dir, 'UNet_framework/degrain_examples/306/source')
data306_target_dir = os.path.join(root_dir, 'UNet_framework/degrain_examples/306/target')

def get_files(dir_path):
    return os.listdir(dir_path)

def rename(source_data_dir, source_files, target_files):
    for source_file, target_file in zip(source_files, target_files):
        print(source_file)
        print(target_file)
        os.rename(source_data_dir+'/'+source_file,source_data_dir+'/'+target_file)


print(get_files(data305_source_dir))
rename(data305_source_dir,get_files(data305_source_dir),get_files(data305_target_dir))
rename(data306_source_dir,get_files(data306_source_dir),get_files(data306_target_dir))


