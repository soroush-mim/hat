from matplotlib import pyplot as plt
import argparse
import os

def extract_adv_accs_from_log(dir, mode):
    with open(dir) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line    
    content = [x.strip() for x in content]
    plot_list = []
    for line in content:
        if mode == 'loss':
            if 'Loss:' in line:
                plot_list.append(float(line.split('\t')[0].split(':')[1].strip()[:-1]))

        elif mode == 'nat-acc':
            if 'Standard' in line and 'Train' in line:
                plot_list.append(float(line.split(':')[2].strip()[:-2]))

        elif mode == 'adv-acc':
            if 'Adversarial' in line and 'Train' in line:
                plot_list.append(float(line.split(':')[2].strip()[:-2]))

        elif mode == 'mem-loss':
            if 'memory Loss' in line:
                plot_list.append(float((line.split(':')[-1].strip())))
        

    if mode == 'adv-acc' or mode == 'nat-acc':
        plot_list = plot_list[:-1]

    return plot_list

parser = argparse.ArgumentParser(description='plot loss, adv accuracy and standard accuracy per epoch for multiple runs')
parser.add_argument('--log-dir', type=str, default='/home/mahdi/hat/logs/')
parser.add_argument('--title', type=str, default='')
parser.add_argument('--descs', nargs='+', help='run descs to to compare', required=True)
parser.add_argument('--mode', default='adv-acc', choices=['adv-acc', 'nat-acc' , 'loss' , 'mem-loss','lr'], type=str)

args = parser.parse_args()


linestyles = ["-.","-","--",":"]

for i,desc in enumerate(args.descs):
    LOG_DIR = os.path.join(args.log_dir, desc)
    log_file_dir = os.path.join(LOG_DIR, 'log-train.log')
    plot_list = extract_adv_accs_from_log(log_file_dir, args.mode)
    # if i < 4:
    #     plt.plot(adv_accs, label = desc, linestyle=linestyles[i])
    # else:
    plt.plot(plot_list, label = desc)

plt.xlabel("epoch")
plt.ylabel(args.mode)
plt.title(args.title)
plt.legend()
plt.savefig('plots/'+args.title + '.png')






