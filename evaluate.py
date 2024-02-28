import os
import csv
import json
import logging
import argparse
from dotenv import load_dotenv
from src.preprocessing import read_json_file_crossre_eval, read_json_file_crossre_DoubleMarkers
from itertools import permutations
from sklearn.metrics import classification_report, f1_score
from src.utils.bucketing import Buckets

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

def parse_arguments():

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--train_path', default=None, help='path to training data for oov computation')
    arg_parser.add_argument('--gold_path', type=str, nargs='?', required=True, help='Path to the gold labels file.')
    arg_parser.add_argument('--out_path', type=str, nargs='?', required=True, help='Path where to save scores.')
    arg_parser.add_argument('--attribute', type=str, help='if specified, test instances are bucketed in respect to the attribute')
    arg_parser.add_argument('--summary_exps', type=str, nargs='?', required=True, help='Path to the summary of the overall experiments.')

    return arg_parser.parse_args()

def get_f1(train_path, gold_path, out_path, attribute):

    # get the labels
    label_types = {label: idx for idx, label in enumerate(os.getenv(f"RELATION_LABELS_CROSSRE").split())}

    ## prepare the bucket
    # retrieve domain to base the buckets on: local->test, aggregate->train
    if attribute in os.getenv("aggregate").split():
        domain = os.path.splitext(os.path.basename(train_path))[0].split('-')[0]
    else:
        domain = os.path.splitext(os.path.basename(gold_path))[0].split('-')[0]
    buckets = Buckets(attribute, domain, label_types, train_path=train_path, multi_label=True)

    # get the gold
    _, _, _, gold = read_json_file_crossre_eval(gold_path, buckets)

    tot_micro_f1 = []

    # evaluate bucket by bucket
    id_bucket = 0
    for bucket in gold:
        # retrieve predicted path
        name_file = f'pred-{attribute}-{id_bucket}'
        id_bucket += 1
        predicted_path = os.path.join(out_path, f'{os.path.splitext(os.path.basename(gold_path))[0]}-{name_file}.csv')
        # get the predicted
        predicted = []
        with open(predicted_path) as predicted_file:
            predicted_reader = csv.reader(predicted_file, delimiter=',')
            next(predicted_reader)
            for line in predicted_reader:
                instance_labels = [0] * len(label_types.keys())
                for elem in line[0].split(' '):
                    instance_labels[label_types[elem]] = 1
                predicted.append(instance_labels)

        # check gold and predicted lengths
        assert len(bucket) == len(predicted), "Length of gold and predicted labels should be equal."

        try:
            report = classification_report(bucket, predicted, target_names=label_types.keys(), output_dict=True, zero_division=0)
            tot_micro_f1.append(report['micro avg']['f1-score'])
        except ValueError:
            # if the bucket is empty
            tot_micro_f1.append(0)

        # save F1 per label
        if attribute == 'none':
            domain = os.path.splitext(os.path.basename(gold_path))[0].split('-')[0]
            for elem in report:
                if elem in label_types.keys():
                    summary_exp_path = os.path.join(args.summary_exps, f"dev-{domain}-{elem}.txt")
                    with open(summary_exp_path, 'a') as file:
                        file.write(f"{str(report[elem]['f1-score'] * 100).replace('.', ',')}; ")


    return tot_micro_f1


if __name__ == '__main__':

    args = parse_arguments()
    logging.info(f"Evaluating {args.gold_path}")

    tot_micro_f1 = get_f1(args.train_path, args.gold_path, args.out_path, args.attribute)

    # saving micro-f1
    domain = os.path.splitext(os.path.basename(args.gold_path))[0].split('-')[0]
    for i in range(len(tot_micro_f1)):
        summary_exp_path_micro = os.path.join(args.summary_exps, f"{domain}-{args.attribute}-{i}-micro.txt")
        logging.info(f"Saving scores to {summary_exp_path_micro} -> Micro F1: {tot_micro_f1[i] * 100}")
        with open(summary_exp_path_micro, 'a') as file:
            file.write(f"Macro F1: {macro_f1 * 100}\n")