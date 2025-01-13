import sys
import os
import argparse
import torch
import torch.cuda
import ujson as json
from transformers import AutoConfig, AutoModel, AutoTokenizer
import pandas as pd

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import DocREModel
from utils.utils_atlop_lace import set_seed, load_data, create_id2rel, get_preds, train, create_entity_types_to_tokens_mapping
from utils.utils import save_arguments, format_and_save_for_submission
from utils.utils_dreeam import get_sentence_labels

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ablation", default="eider", type=str, choices=['atlop', 'eider', 'eider_rule'])
    parser.add_argument("--coref_method", default='none', choices=['hoi', 'none'])

    parser.add_argument("--data_dir", required=True, type=str, help="Directory for the dataset")
    parser.add_argument("--transformer_type", required=True, type=str, help="Type of transformer model (e.g., bert)")
    parser.add_argument("--model_name_or_path", required=True, type=str, help="Name or path to the transformer model (e.g., bert-base-cased)")
    parser.add_argument("--data_file_with_silver_evidence", required=True, type=str, help="File name of the dataset (e.g., train_annotated.json)")
    parser.add_argument("--test_file_with_silver_evidence", default="test.json", type=str)
    parser.add_argument("--save_finetuned_model_path", required=True, type=str, help="Path to save the fine-tuned model")
    parser.add_argument("--rel2id_file", required=True, type=str, help="File containing relation to ID mapping")
    parser.add_argument("--pred_path", required=True, type=str, help="Path for saving predictions")
    parser.add_argument("--results_path", required=True, type=str, help="Path for saving results")
    parser.add_argument("--use_specific_entity_types_markers", action="store_true", help="Flag to use specific entity types markers")
    parser.add_argument("--use_specific_parent_entity_types_markers", action="store_true", help="Flag to use specific entity types markers and parent entity types")
    parser.add_argument("--evidence_construction", choices=['predictions', 'predictions_and_rule_based', 'rule_based'], default='predictions_and_rule_based', type=str, help="How to construct the evidences") 
    parser.add_argument("--multi_label", action="store_true", help="Flag to use the multi label configuration during predictions. Else take the best prediction")

    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--num_labels", default=4, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--lr_added", default=1e-4, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--num_train_epochs", default=30.0, type=float)
    parser.add_argument("--evaluation_steps", default=-1, type=int)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--num_class", type=int, default=97)
    parser.add_argument("--max_sent_num", default=25, type=int, help="Max number of sentences in each document.")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.train_on_full_train_set = True
    args.cross_validation = False

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    train_file = os.path.join(args.data_dir, args.data_file_with_silver_evidence)
    test_file = os.path.join(args.data_dir, args.test_file_with_silver_evidence)
    rel2id_file = os.path.join(args.data_dir, args.rel2id_file)
    rel2id = json.load(open(rel2id_file, 'r'))

    df_train = load_data(train_file)
    df_dev = pd.DataFrame()
    df_test = load_data(test_file)

    df_train = df_train.apply(lambda row : get_sentence_labels(row, args.evidence_construction), axis=1)
    df_test = df_test.apply(lambda row : get_sentence_labels(row, args.evidence_construction), axis=1)

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    entity_types_to_tokens_mapping, tokens_to_add_to_vocab = create_entity_types_to_tokens_mapping(df_train, args.use_specific_entity_types_markers, args.use_specific_parent_entity_types_markers)
    tokenizer.add_tokens(tokens_to_add_to_vocab)
    model.resize_token_embeddings(len(tokenizer))
    config.vocab_size = len(tokenizer)

    set_seed(args)
    model = DocREModel(config, model, num_labels=args.num_labels, ablation=args.ablation, max_sent_num=args.max_sent_num)
    model.to(device)

    id2rel = create_id2rel(rel2id)

    train(args, model, df_train, df_dev, id2rel, eider_model=True)

    # Test
    model.load_state_dict(torch.load(args.save_finetuned_model_path))

    df_test = df_test.apply(lambda row: get_preds(row, args, model, id2rel), axis=1)

    os.makedirs(os.path.dirname(args.pred_path), exist_ok=True)
    format_and_save_for_submission(df_test, args.pred_path)
    
    save_arguments(args)

if __name__ == "__main__":
    main()
