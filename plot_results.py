from matplotlib import pyplot as plt
import argparse
import os

def extract_adv_accs_from_log(dir):
    with open(dir) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line    
    content = [x.strip() for x in content]
    # losses = []
    adv_accs = []
    # clean_accs = []
    memory_losses = []
    for line in content:
        # if 'Loss:' in line:
        #     losses.append(float(line.split('\t')[0].split(':')[1].strip()[:-1]))
        # if 'Standard' in line and 'Train' in line:
        #     clean_accs.append(float(line.split(':')[2].strip()[:-2]))
        if 'Adversarial' in line and 'Train' in line:
            adv_accs.append(float(line.split(':')[2].strip()[:-2]))

        # if 'memory Loss' in line:
        #     memory_losses.append(float((line.split(':')[-1].strip())))

    # clean_accs = clean_accs[:-1]
    adv_accs = adv_accs[:-1]

    return adv_accs

parser = argparse.ArgumentParser(description='plot loss, adv accuracy and standard accuracy per epoch for multiple runs')
parser.add_argument('--log-dir', type=str, default='/home/mahdi/hat/logs/')
parser.add_argument('--title', type=str, default='')
parser.add_argument('--descs', nargs='+', help='run descs to to compare', required=True)

args = parser.parse_args()


linestyles = ["-.","-","--",":"]

for i,desc in enumerate(args.descs):
    LOG_DIR = os.path.join(args.log_dir, desc)
    log_file_dir = os.path.join(LOG_DIR, 'log-train.log')
    adv_accs = extract_adv_accs_from_log(log_file_dir)
    # if i < 4:
    #     plt.plot(adv_accs, label = desc, linestyle=linestyles[i])
    # else:
    plt.plot(adv_accs, label = desc)

plt.xlabel("epoch")
plt.ylabel("adv acc")
plt.title(args.title)
plt.legend()
plt.savefig('plots/'+args.title + '.png')






