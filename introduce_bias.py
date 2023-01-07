import os
import random


import pandas as pd

# dir that contains csv files of lists of data
# Train 1000 (resampled)
# Val 218 (resampled)
# Test 624 (kept unchanged from download site)
correct_dir = './CXR_0_0_csv'


bias0 = {'bias_name': '5_5', 'bias_type': {'0to1': 0.05, '1to0': 0.05}}
bias1 = {'bias_name': '10_10', 'bias_type': {'0to1': 0.1, '1to0': 0.1}}
bias2 = {'bias_name': '15_15', 'bias_type': {'0to1': 0.15, '1to0': 0.15}}
bias3 = {'bias_name': '20_20', 'bias_type': {'0to1': 0.2, '1to0': 0.2}}
bias4 = {'bias_name': '25_25', 'bias_type': {'0to1': 0.25, '1to0': 0.25}}

bias0_0 = {'bias_name': '10_0', 'bias_type': {'0to1': 0.1, '1to0': 0}}
bias1_0 = {'bias_name': '20_0', 'bias_type': {'0to1': 0.2, '1to0': 0}}
bias2_0 = {'bias_name': '30_0', 'bias_type': {'0to1': 0.3, '1to0': 0}}
bias3_0 = {'bias_name': '40_0', 'bias_type': {'0to1': 0.4, '1to0': 0}}
bias4_0 = {'bias_name': '50_0', 'bias_type': {'0to1': 0.5, '1to0': 0}}

bias0_1 = {'bias_name': '0_10', 'bias_type': {'0to1': 0, '1to0': 0.1}}
bias1_1 = {'bias_name': '0_20', 'bias_type': {'0to1': 0, '1to0': 0.2}}
bias2_1 = {'bias_name': '0_30', 'bias_type': {'0to1': 0, '1to0': 0.3}}
bias3_1 = {'bias_name': '0_40', 'bias_type': {'0to1': 0, '1to0': 0.4}}
bias4_1 = {'bias_name': '0_50', 'bias_type': {'0to1': 0, '1to0': 0.5}}
biases = [#bias0, bias1, bias2, bias3, bias4,
bias0_0,
bias1_0,
bias2_0,
bias3_0,
bias4_0,
bias0_1,
bias1_1,
bias2_1,
bias3_1,
bias4_1]
base_folder = 'bias_csv'
def move_img(src_lst, dst_lst, percentage):
    files = src_lst
    no_of_files = int(round(len(files) * percentage))
    for file_name in random.sample(files, no_of_files):
        src_lst.remove(file_name)
        dst_lst.append(file_name)
    return src_lst, dst_lst

def introduce_bias(bias, trial_num):
    bias_name = f"{bias['bias_name']}_trial{trial_num}_csv"
    os.mkdir(os.path.join(base_folder, bias_name))
    for f in ['test', 'train', 'val']:
        os.mkdir(os.path.join(base_folder, bias_name, f))
        corr_normal = pd.read_csv(os.path.join(correct_dir, f, 'normal_names.csv'))['name'].to_list()
        corr_pneu = pd.read_csv(os.path.join(correct_dir, f, 'pneu_names.csv'))['name'].to_list()
        if f == 'train':

            biased_normal = []
            biased_pneu = []
            corr_normal, biased_normal = move_img(corr_normal, biased_normal, bias['bias_type']['0to1'])
            corr_pneu, biased_pneu = move_img(corr_pneu, biased_pneu, bias['bias_type']['1to0'])
            biased_normal, corr_pneu = move_img(biased_normal, corr_pneu, 1)
            biased_pneu, corr_normal = move_img(biased_pneu, corr_normal, 1)

        normal_dict = {'name': corr_normal}
        pneu_dict = {'name': corr_pneu}
        dataframe = pd.DataFrame(normal_dict)
        dataframe.to_csv(os.path.join(base_folder, bias_name, f, f'normal_names.csv'))
        dataframe = pd.DataFrame(pneu_dict)
        dataframe.to_csv(os.path.join(base_folder, bias_name, f, f'pneu_names.csv'))
random.seed(42)

normal = [{'bias_name': '0_0', 'bias_type': {'0to1': 0, '1to0': 0}}]
#for b in normal:
trial_num_lst = [1,2]
if not os.path.exists(base_folder): os.mkdir(base_folder)
for b in biases:
    for t in trial_num_lst:
        introduce_bias(b, t)

#os.mkdir(root_dir)
# for f in ['test','train','val']:
#     os.mkdir(os.path.join(root_dir, f))
#     normal_files = os.listdir(os.path.join('./copy', f, 'NORMAL'))
#     pneu_files = os.listdir(os.path.join('./copy', f, 'PNEUMONIA'))
#
#
#     normal_dict = {'name': normal_files}
#     pneu_dict = {'name': pneu_files}
#     dataframe = pd.DataFrame(normal_dict)
#     dataframe.to_csv(os.path.join(root_dir, f, f'normal_names.csv'))
#     dataframe = pd.DataFrame(pneu_dict)
#     dataframe.to_csv(os.path.join(root_dir, f, f'pneu_names.csv'))
# a = [1,2,3, 33]
# b = []
# print(move_img(a, b, 0.5))
# mean = 0.0
# for images, label in loader:
#     if label[0] == 1:
#         plt.imshow(images[0].permute(1,2,0), cmap='gray')
#         plt.show()
#         break
# for images, _ in loader:
#     batch_samples = images.size(0)
#     images = images.view(batch_samples, images.size(1), -1)
#     mean += images.mean(2).sum(0)
# mean = mean / len(loader.dataset)
#
# var = 0.0
# for images, _ in loader:
#     batch_samples = images.size(0)
#     images = images.view(batch_samples, images.size(1), -1)
#     var += ((images - mean.unsqueeze(1))**2).sum([0,2])
# std = torch.sqrt(var / (len(loader.dataset)*224*224))
# print(mean)
# print(std)

