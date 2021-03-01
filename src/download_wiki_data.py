import argparse
import msgpack

import data_utils as du


parser = argparse.ArgumentParser(description='Download wiki40b.')
parser.add_argument('--save', type=str, default='../data/wiki40b/paras_0_10',
                    help='Name of save file')
parser.add_argument('--start', type=int, default=0,
                    help='Start wikipedia split.')
parser.add_argument('--end', type=int, default=10,
                    help='End wikipedia split.')
args = parser.parse_args()

def main(savefile, start, end):
    data = du.get_wiki_data(start, end)
    paragraphs = []
    for d in data:
        paragraphs.extend(du.get_paragraphs(d))

    with open(savefile+".msgpack", "wb") as buf:
        for para in paragraphs:
            buf.write(msgpack.packb(para, use_bin_type=True))

if __name__ == "__main__":
    main(args.save, args.start, args.end)
