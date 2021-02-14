import argparse
import data_utils as du

parser = argparse.ArgumentParser(description='Batch process wiki40b text into paragraph and SRL.')
parser.add_argument('--save', type=str, default='./xu_ins_srl_%s_%s.pkl',
                    help='Name of save file')
parser.add_argument('--data_path', type=str, default='../../cross_domain_coherence/data/parsed_wsj/test_perm.tsv',
                    help='Name of save file')
parser.add_argument('--batch', type=int, default=16,
                    help='Batch size for SRL parser.')
parser.add_argument('--start', type=int, default=0,
                    help='Start wikipedia split.')
parser.add_argument('--end', type=int, default=10,
                    help='End wikipedia split.')
args = parser.parse_args()

def main(save_file_tmp, path, start, end):
    save_file = save_file_tmp % (start, end)
    paragraphs = du.get_xu_data(path)[start:end]
    reprs = du.srl_paragraphs_batched_xu(paragraphs, batch_size=16)
    du.serialize_reprs(reprs, save_file)

if __name__ == "__main__":
    main(args.save, args.data_path, args.start, args.end)
