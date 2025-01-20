import argparse
import os
import numpy as np
import torch
import ujson as json
import pandas as pd
from transformers import AutoConfig, AutoModel, AutoTokenizer

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import LACEModel
from utils.utils import save_arguments, format_and_save_for_submission
from utils.utils_atlop_lace import set_seed, print_statistics, load_data, format_data, create_id2rel, train, get_preds, create_entity_types_to_tokens_mapping
from LACE.utils_lace import create_adj_matrix, create_label_embeddings


def reduce_nb_unrelated_relations(example):
    doc_labels = example['labels']
    doc_hts = example['hts']
    pos_label = []
    neg_label = []
    pos_hts = []
    neg_hts = []
    for i in range(len(doc_labels)):
        if doc_labels[i][0] == 0:
            pos_label.append(doc_labels[i])
            pos_hts.append(doc_hts[i])
        else:
            neg_label.append(doc_labels[i])
            neg_hts.append(doc_hts[i])
    neg_label = neg_label[:10*len(pos_label)+5]
    np.random.seed(10)
    np.random.shuffle(neg_hts)
    neg_hts = neg_hts[:10*len(pos_label)+5]
    pos_and_neg_label = pos_label + neg_label
    pos_and_neg_hts = pos_hts + neg_hts
    state = np.random.get_state()
    np.random.shuffle(pos_and_neg_label)
    np.random.set_state(state)
    np.random.shuffle(pos_and_neg_hts)
    example['labels'] = pos_and_neg_label
    example['hts'] = pos_and_neg_hts
    return example

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", required=True, type=str, help="Directory for the dataset")
    parser.add_argument("--transformer_type", required=True, type=str, help="Type of transformer model (e.g., bert)")
    parser.add_argument("--model_name_or_path", required=True, type=str, help="Name or path to the transformer model (e.g., bert-base-cased)")
    parser.add_argument("--data_file", required=True, type=str, help="File name of the dataset (e.g., train_annotated.json)")
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_finetuned_model_path", required=True, type=str, help="Path to save the fine-tuned model")
    parser.add_argument("--rel2id_file", required=True, type=str, help="File containing relation to ID mapping")
    parser.add_argument("--pred_path", required=True, type=str, help="Path for saving predictions")
    parser.add_argument("--results_path", required=True, type=str, help="Path for saving results")
    parser.add_argument("--adjacency_matrix_save_path", required=True, type=str, help="Path where adjacency matrix is or will be saved")
    parser.add_argument("--label_embeddings_save_path", required=True, type=str, help="Path where label embeddings is or will be saved")
    parser.add_argument("--synthetic_data_file", default="", type=str, help="Synthetic data file path")
    parser.add_argument("--add_synthetic_data_for_training", action="store_true", help="Flag to add synthetic data for training")
    parser.add_argument("--use_specific_entity_types_markers", action="store_true", help="Flag to use specific entity types markers")
    parser.add_argument("--use_specific_parent_entity_types_markers", action="store_true", help="Flag to use specific entity types markers and parent entity types")
    parser.add_argument("--use_entity_embedding_layers_mean", action="store_true", help="Flag to use the mean of entity reprensation from all layers. Else take the representation of last layer")
    parser.add_argument("--use_entity_attention_layers_mean", action="store_true", help="Flag to use the mean of entity attention from all layers. Else take the attention of last layer")
    parser.add_argument("--multi_label", action="store_true", help="Flag to use the multi label configuration during predictions. Else take the best prediction")
    parser.add_argument("--type_of_graph", default="GAT", type=str, choices=['GAT', 'GCN'], help="Type of graph to use in the model")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--lr_added", default=1e-4, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    rel2id_file = os.path.join(args.data_dir, args.rel2id_file)
    rel2id = json.load(open(rel2id_file, 'r'))

    train_file = os.path.join(args.data_dir, args.data_file)
    test_file = os.path.join(args.data_dir, args.test_file)

    df_train = load_data(train_file)
    df_dev = pd.DataFrame()
    df_test = load_data(test_file)

    if args.add_synthetic_data_for_training:
        synthetic_file = os.path.join(args.data_dir, args.synthetic_data_file)
        df_synthetic = load_data(synthetic_file)
        df_train = pd.concat([df_train, df_synthetic])
    
    entity_types_to_tokens_mapping, tokens_to_add_to_vocab = create_entity_types_to_tokens_mapping(df_train, args.use_specific_entity_types_markers, args.use_specific_parent_entity_types_markers)
    tokenizer.add_tokens(tokens_to_add_to_vocab)
    model.resize_token_embeddings(len(tokenizer))
    config.vocab_size = len(tokenizer)
    
    df_train = df_train.apply(lambda row: format_data(row, tokenizer, rel2id, entity_types_to_tokens_mapping, args.use_specific_parent_entity_types_markers, max_seq_length=args.max_seq_length, eval=False), axis=1)
    df_dev = df_dev.apply(lambda row: format_data(row, tokenizer, rel2id, entity_types_to_tokens_mapping, args.use_specific_parent_entity_types_markers, max_seq_length=args.max_seq_length, eval=True), axis=1)
    df_test = df_test.apply(lambda row: format_data(row, tokenizer, rel2id, entity_types_to_tokens_mapping, args.use_specific_parent_entity_types_markers, max_seq_length=args.max_seq_length, eval=True), axis=1)

    df_train = df_train.apply(reduce_nb_unrelated_relations, axis=1)
    print_statistics(df_train)


    if not os.path.exists(args.adjacency_matrix_save_path):   
        create_adj_matrix(df_train, args.adjacency_matrix_save_path)
    if not os.path.exists(args.label_embeddings_save_path):
        create_label_embeddings(rel2id, args.label_embeddings_save_path, model, tokenizer)

    set_seed(args)
    model = LACEModel(config, model, args.adjacency_matrix_save_path, args.label_embeddings_save_path, device, num_labels=args.num_class, 
                      use_entity_embedding_layers_mean=args.use_entity_embedding_layers_mean, use_entity_attention_layers_mean=args.use_entity_attention_layers_mean,
                      type_of_graph=args.type_of_graph)
    model.to(device)

    id2rel = create_id2rel(rel2id)

    train(args, model, df_train, df_dev, id2rel, lace_model=True)

    # Get preds on test set
    model.load_state_dict(torch.load(args.save_finetuned_model_path))

    df_test = df_test.apply(lambda row: get_preds(row, args, model, id2rel), axis=1)

    os.makedirs(os.path.dirname(args.pred_path), exist_ok=True)
    format_and_save_for_submission(df_test, args.pred_path)

    save_arguments(args)

if __name__ == "__main__":
    main()
