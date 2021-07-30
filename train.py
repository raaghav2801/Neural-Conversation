from persona import *
import argparse
import codecs
import torch
from torch import tensor


parser = argparse.ArgumentParser()

parser.add_argument('-df', '--data_folder', type=str, default='data/testing',
					help='the folder that contains your dataset and vocabulary file')
parser.add_argument('-tf', '--train_file', type=str, default='train.txt')
parser.add_argument('-devf', '--dev_file', type=str, default='valid.txt')
parser.add_argument('-D', '--dictPath', type=str, default='vocabulary')
parser.add_argument('-sf', '--save_folder', type=str, default='save/testing')
parser.add_argument('-sp', '--save_prefix', type=str, default='model')
parser.add_argument('-spa', '--save_params', type=str, default='params')
parser.add_argument('-of', '--output_file', type=str, default='log')
parser.add_argument('-ns', '--no_save', action='store_true')
parser.add_argument('-c', '--cpu', action='store_true')

parser.add_argument('-U', '--UNK', type=int, default=0,
					help='the index of UNK. UNK+special_word=3.')
parser.add_argument('-S', '--special_word', type=int, default=3,
					help='default special words include: padding, EOS, EOT.')

parser.add_argument('-ft', '--fine_tuning', action='store_true')
parser.add_argument('-ftm', '--fine_tuning_model', type=str, default='model')

parser.add_argument('-PN', '--PersonaNum', type=int, default=2)
parser.add_argument('-SM', '--SpeakerMode', action='store_true')
parser.add_argument('-AM', '--AddresseeMode', action='store_true')

parser.add_argument('-bs', "--batch_size", type=int, default=256)
parser.add_argument('-sml', "--source_max_length", type=int, default=50)
parser.add_argument('-tml', "--target_max_length", type=int, default=50)
parser.add_argument('-mi', "--max_iter", type=int, default=10)

parser.add_argument('-dim', "--dimension", type=int, default=512)
parser.add_argument('-lay', "--layers", type=int, default=4)
parser.add_argument('-iw', "--init_weight", type=float, default=0.1)

parser.add_argument('-a', "--alpha", type=int, default=1)
parser.add_argument('-sh', "--start_halve", type=int, default=6)
parser.add_argument('-t', "--thres", type=int, default=5)
parser.add_argument('-Dr', "--dropout", type=float, default=0.2)


args = parser.parse_args()
print(args)
print()

if __name__ == '__main__':
	model = persona(args)
	model.train()
	model.evaluate()



