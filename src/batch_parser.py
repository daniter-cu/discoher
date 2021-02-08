import argparse
import data_utils as du

parser = argparse.ArgumentParser(description='Batch process wiki40b text into paragraph and SRL.')
parser.add_argument('--save', type=str, default='./wiki40b_srl_%s_%s.pkl',
                    help='Name of save file')
parser.add_argument('--start', type=int, default=0,
                    help='Start wikipedia split.')
parser.add_argument('--end', type=int, default=10,
                    help='End wikipedia split.')
parser.add_argument('--batch', type=int, default=16,
                    help='Batch size for SRL parser.')
args = parser.parse_args()

def main(start, end, fname):
    save_file = fname % (start, end)
    data = du.get_wiki_data(start, end)
    paragraphs = []
    for d in data:
        paragraphs.extend(du.get_paragraphs(d))
    reprs = du.srl_paragraphs_batched(paragraphs, batch_size=16)
    du.serialize_reprs(reprs, save_file)

if __name__ == "__main__":
    main(args.start, args.end, args.save)
