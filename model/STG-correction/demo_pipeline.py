import warnings
warnings.filterwarnings("ignore")
from app import Pipeline
import argparse
from config import evaljoint_parse as demo_parse

def read_text(path :str) -> str:
    with open(path, 'r', encoding='utf-8') as file:
        data = file.readline()
    file.close()
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Path options.
    parser.add_argument("--report_title", type=str, default="FCGEC Project Template Test Report V2.0", help="Report Title")
    parser.add_argument("--dataset_path", type=str, default="dataset/demo/demo-v1.0.txt", help="Path of the document data.")
    parser.add_argument("--report_prefix", type=str, default="./", help="Prefix of the document data.")
    parser.add_argument("--report_type", type=str, default="pdf", help="Type for the output of document (docx, pdf).")
    args = parser.parse_args()
    # Read Data
    doc = read_text(args.dataset_path)
    # FCGEC Reporter Pipeline
    # args_binary   = bert_config.parse_args()
    args_binary   = None  # Note: In our repo, binary and type are not available
    args_demo = demo_parse()
    pipecls = Pipeline(args_binary, args_demo)
    export_path = pipecls(doc, args.report_title, args.report_prefix, args.report_type)
    print(">>> Report has been saved at %s" % export_path)