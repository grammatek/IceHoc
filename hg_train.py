import re
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
import torch
from tqdm import tqdm
import transformers as ppb
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel
import warnings
import argparse

BATCH_SIZE = 512
N_EPOCHS = 50
ALPHA = 0.001
TEST_SIZE = 0.1
VALIDATION_SIZE = 0.1

HOMOGRAPH_RE = r'\[\[(.*?)\]\]'
LABEL_RE = r'[\s\t]+(0|1)$'

warnings.filterwarnings('ignore')


def load_data_from_directory(directory_path):
    """
    Loads data from all files within the specified directory, treating each file as a separate homograph group.

    Args:
        directory_path (str): The path to the directory containing text files.

    Returns:
        pd.DataFrame: A dataframe containing sentences, homographs, labels, and group identifiers.
    """
    print("Loading dataset ...")
    all_data = []
    for file_path in Path(directory_path).rglob('*.*'):
        # Skip all *.pkl files
        if file_path.suffix == '.pkl':
            continue
        # Group ID: file name without suffix
        group_id = file_path.stem
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                homograph_match = re.search(HOMOGRAPH_RE, line)
                label_match = re.search(LABEL_RE, line)
                if homograph_match and label_match:
                    sentence = line[:label_match.start()].strip()
                    homograph = homograph_match.group(1)
                    label = int(label_match.group(1))
                    all_data.append((sentence, homograph, label, group_id))
    if len(all_data) == 0:
        raise RuntimeError(f'No files found in {directory_path}')
    return pd.DataFrame(all_data, columns=['sentence', 'homograph', 'label', 'group_id'])


def balance_labels_within_group(group_df):
    """
    Balances the number of examples for each class within a given homograph group by downsampling.

    Args:
        group_df (pd.DataFrame): A DataFrame containing data for a specific homograph group, including sentences,
                                 homographs, labels, and the group identifier.

    Returns:
        pd.DataFrame: A balanced DataFrame for the group, with equal numbers of examples for each class.
    """
    # Find the minority class and its count
    min_class_count = group_df['label'].value_counts().min()
    # Return a balanced dataframe for the group
    return pd.concat([
        group_df[group_df['label'] == 0].sample(n=min_class_count, random_state=42),
        group_df[group_df['label'] == 1].sample(n=min_class_count, random_state=42)
    ]).sample(frac=1)


def create_balanced_splits(df, test_size=TEST_SIZE, valid_size=VALIDATION_SIZE, show_histogram=False):
    all_train_dfs = []
    all_valid_dfs = []
    all_test_dfs = []

    for group_id in df['group_id'].unique():
        group_df = df[df['group_id'] == group_id]
        balanced_group_df = balance_labels_within_group(group_df)

        try:
            # Splitting balanced group data into train+valid and test
            train_valid_df, test_df = train_test_split(balanced_group_df, test_size=test_size, random_state=42,
                                                       stratify=balanced_group_df['label'])

            # Now, split train_valid_df into actual train and valid sets
            train_df, valid_df = train_test_split(train_valid_df, test_size=valid_size / (1 - test_size),
                                                  random_state=42, stratify=train_valid_df['label'])
            all_train_dfs.append(train_df)
            all_valid_dfs.append(valid_df)
            all_test_dfs.append(test_df)
        except Exception as e:
            print(f'Exception: {e}')

    # Combine and shuffle the final datasets
    final_train_df = pd.concat(all_train_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    final_valid_df = pd.concat(all_valid_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    final_test_df = pd.concat(all_test_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

    if show_histogram:
        print_group_id_frequencies(final_train_df)
        print_group_id_frequencies(final_valid_df)
        print_group_id_frequencies(final_test_df)

    return final_train_df, final_valid_df, final_test_df


def print_group_id_frequencies(df):
    """
    Prints the frequency of each group_id in the given dataframe.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the 'group_id' column.

    Returns:
    None
    """
    # Calculate the frequency of each group_id
    group_id_counts = df['group_id'].value_counts()
    print(group_id_counts)


def precalculate_embeddings(bert_model, device, train_set, valid_set, test_set):
    bert_model.eval()
    datasets = {'train': train_set, 'valid': valid_set, 'test': test_set}

    for name, df in datasets.items():
        combined_embeddings = []

        for i in tqdm(range(0, len(df), BATCH_SIZE), desc=f"Creating embeddings for {name} set"):
            batch = df.iloc[i:i + BATCH_SIZE]
            input_ids = torch.cat(batch['processed_data'].map(lambda x: x['input_ids']).tolist()).to(device)
            attention_mask = torch.cat(batch['processed_data'].map(lambda x: x['attention_mask']).tolist()).to(device)
            positions = torch.nn.utils.rnn.pad_sequence(batch['processed_data'].map(
                lambda x: torch.tensor(x['homograph_positions'], dtype=torch.long)).tolist(), batch_first=True,
                                                        padding_value=-1).to(device)

            with torch.no_grad():
                outputs = bert_model(input_ids, attention_mask=attention_mask)
                sequence_output = outputs.last_hidden_state

            # Directly calculate combined embeddings on GPU and then move to CPU
            for j in range(sequence_output.size(0)):
                cls_embedding = sequence_output[j, 0, :]
                valid_positions = positions[j][positions[j] != -1]
                if valid_positions.numel() > 0:
                    # Ensure there are valid positions
                    homograph_embedding = sequence_output[j, valid_positions].mean(dim=0)
                    combined_embedding = torch.cat((cls_embedding.unsqueeze(0), homograph_embedding.unsqueeze(0)),
                                                   dim=1)
                    combined_embeddings.append(combined_embedding.cpu().numpy())
                else:
                    raise RuntimeError(f"No valid homograph embedding detected !")

            # Clearing unused GPU memory
            del input_ids, attention_mask, positions, sequence_output
            torch.cuda.empty_cache()

        datasets[name]['combined_embeddings'] = pd.Series(combined_embeddings)

    return datasets['train'], datasets['valid'], datasets['test']


def load_and_preprocess_data(bert_model, bert_tokenizer, device, directory_path, around, force_reprocess=False,
                             show_histogram=False):
    """
    Loads, preprocesses, and splits the dataset. It adjusts sentences around homographs, tokenizes them,
    balances the dataset, and finally calculates embeddings.

    Args:
        bert_model (transformers.PreTrainedModel): A BERT model for embedding calculation.
        bert_tokenizer (transformers.PreTrainedTokenizer): A tokenizer corresponding to bert_model.
        device (torch.device): The device to run the model on.
        directory_path (str): The path to the directory containing text files for the dataset.
        around (int): The number of words around each homograph to consider in the adjusted sentence.
        force_reprocess (bool): If True, forces the reprocessing of the dataset even if cached data exists.
        show_histogram (bool): If True, shows histogram of all homographs after dataset balancing

    Returns:
        tuple: Three DataFrames corresponding to the training, validation, and test sets, each with sentences,
               labels, group IDs, and calculated embeddings.
    """
    model_name = bert_model.name_or_path.replace('/', '_')
    cache_file = Path(directory_path) / f"{model_name}_{around}_processed_dataset.pkl"

    if cache_file.exists() and not force_reprocess:
        print("Loading preprocessed dataset from cache ...")
        df = pd.read_pickle(cache_file)
    else:
        df = load_data_from_directory(directory_path)
        print("Adjust sentence and remove homograph labels ...")
        df['adjusted_sentence'] = df['sentence'].apply(lambda x: get_context_around_homograph(x, around))
        print("Pre-tokenize sentences ...")
        processed_data_list = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Tokenizing and processing"):
            try:
                processed_data = preprocess_and_tokenize_data(row, bert_tokenizer)
                processed_data_list.append(processed_data)
            except Exception as e:
                print(e)
                continue

        # Assign the list of processed data back to the DataFrame
        df['processed_data'] = processed_data_list
        print("Saving preprocessed dataset to cache ...")
        df.to_pickle(cache_file)
    print("Create balanced splits of the data ...")
    train_set, valid_set, test_set = create_balanced_splits(df, TEST_SIZE, VALIDATION_SIZE, show_histogram)

    return precalculate_embeddings(bert_model, device, train_set, valid_set, test_set)


def preprocess_and_tokenize_data(row, tokenizer):
    # Tokenize the adjusted sentence
    encoded = tokenizer.encode_plus(row['adjusted_sentence'], add_special_tokens=True, return_tensors='pt',
                                    padding='max_length', truncation=True, max_length=512)
    input_ids = encoded['input_ids']

    # Find homograph in the original sentence and tokenize
    homograph = re.search(HOMOGRAPH_RE, row['sentence']).group(1)
    homograph_tokens = tokenizer.tokenize(homograph)
    homograph_ids = tokenizer.convert_tokens_to_ids(homograph_tokens)

    # Identify positions of homograph sub-tokens
    positions = [i for i, token_id in enumerate(input_ids[0]) if token_id in homograph_ids]
    if not positions:
        print("The following line has no valid homograph:")
        print(f"{row['sentence']}")
        raise ValueError(f"No valid positions for the homograph {homograph} were found.")

    return {'input_ids': input_ids, 'attention_mask': encoded['attention_mask'], 'homograph_positions': positions}


def detect_tokenizer_behavior(tokenizer):
    """
    Detects whether the tokenizer uses a special prefix for tokens that are not at the beginning of a new word.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to check.

    Returns:
        bool: True if the tokenizer uses a special prefix, False otherwise.
    """
    # Tokenize a simple multi-word text to detect special character usage
    test_tokens = tokenizer.tokenize("test token")

    # Check if any token besides the first starts with a special character like "Ġ"
    uses_special_prefix = any(token.startswith("Ġ") for token in test_tokens[1:])

    return uses_special_prefix


def compute_weights(df):
    """
    Computes class weights based on the imbalance in the dataset.

    Args:
        df (pd.DataFrame): The dataframe containing the labels for each class.

    Returns:
        dict: A dictionary with class weights.
    """
    print("Compute weights ...")
    class_weights = compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'])
    return {i: weight for i, weight in enumerate(class_weights)}


def get_model_and_tokenizer(model_name):
    """
    Retrieves the model and tokenizer based on the specified model name.

    Args:
        model_name (str): The name of the model to retrieve. Supports a range of models, including 'distilbert'
         and others.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    if model_name == 'distilbert':
        model_class, tokenizer_class, pretrained_weights = (
            ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    elif model_name in ['sbert-ruquad', 'icelandic-ner-bert', 'icebert', 'macocu-is', 'macocu-is-base',
                        'labse', 'convbert', 'convbert-small']:
        model_class, tokenizer_class = AutoModel, AutoTokenizer
        pretrained_weights = {
            'sbert-ruquad': 'language-and-voice-lab/sbert-ruquad',
            'icelandic-ner-bert': 'grammatek/icelandic-ner-bert',
            'icebert': 'mideind/IceBERT',
            'macocu-is': 'MaCoCu/XLMR-MaCoCu-is',
            'macocu-is-base': 'MaCoCu/XLMR-base-MaCoCu-is',
            'labse': 'setu4993/LaBSE',
            'convbert': 'jonfd/convbert-base-igc-is',
            'convbert-small': 'jonfd/convbert-small-igc-is'
        }[model_name]
    else:
        raise ValueError(f"Unsupported model {model_name}")

    print(f"Loading {model_name} model ...")
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    return model, tokenizer


def evaluate(dataset, classifier_model, group_id=None):
    """
    Evaluates the classifier model on a given dataset, optionally filtering for a specific homograph group.

    Args:
        dataset (pd.DataFrame): The dataset to evaluate, containing embeddings and labels.
        classifier_model (Any): The classifier model to use for predictions.
        group_id (str, optional): If provided, filters the dataset to evaluate only the specified homograph group.

    Returns:
        float: The weighted F1 score of the model's predictions.
    """
    if group_id is not None:
        dataset = dataset[dataset['group_id'] == group_id]

    all_predictions = []
    all_actual = dataset['label'].tolist()

    combined_embeddings = np.vstack(dataset['combined_embeddings'].tolist())

    # Predict in batches due to potential memory constraints
    for i in tqdm(range(0, len(combined_embeddings), BATCH_SIZE), desc="Evaluating dataset"):
        combined_embeddings_batch = combined_embeddings[i:i + BATCH_SIZE]

        # Prediction using the classifier model
        predictions = classifier_model.predict(combined_embeddings_batch)

        # Store predictions for later evaluation
        all_predictions.extend(predictions)

    # Calculate the F1 score on the test set
    f1 = f1_score(all_actual, all_predictions, average='weighted')
    return f1


def evaluate_groups(test_df, classifier_model):
    for group_id in test_df['group_id'].unique():
        f1_score_group = evaluate(test_df, classifier_model, group_id=group_id)
        print(f"F1 score for group {group_id}: {f1_score_group:.4f}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Homograph classification using a BERT model + logistic regression")
    parser.add_argument("--model", type=str,
                        choices=["distilbert", "sbert-ruquad", "icebert", "icelandic-ner-bert", "macocu-is",
                                 "macocu-is-base", "labse", "convbert", "convbert-small"],
                        required=True, help="Model type to use")
    parser.add_argument("--directory", type=str, required=True, help="Path to input training files.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training if available")
    parser.add_argument('--around', type=int, default=None, help='Number of words around the homograph to consider')
    parser.add_argument('-f', '--force', action='store_true', help='Force reprocessing of the dataset')
    parser.add_argument('--histogram', action='store_true', default=False, help='Show histogram of individual homographs')
    return parser.parse_args()


def evaluate_and_save_model(lr_clf, test_df, args):
    f1_val = evaluate(test_df, lr_clf)

    print(f"F1 score on test set (Logistic Regression): {f1_val:.4f}")

    model_filename = f"classifier/trained_clf_{args.model}_{args.around}_{f1_val:.3f}.pkl"
    with open(model_filename, 'wb') as model_file:
        pickle.dump(lr_clf, model_file)
    print(f"Model trained and saved as {model_filename} in the current directory.")
    evaluate_groups(test_df, lr_clf)


def get_context_around_homograph(sentence, around):
    """
    Adjusts the sentence to focus on the context around the marked homograph.

    Args:
        sentence (str): The original sentence containing a marked homograph with [[...]].
        around (int): The maximum number of tokens around the homograph to include.

    Returns:
        str: The adjusted sentence focusing on the specified context.
    """
    # Find the marked homograph using a regular expression
    match = re.search(HOMOGRAPH_RE, sentence)
    if not match:
        raise RuntimeError(f"No homograph marking in {sentence}")

    homograph = match.group(1)
    start_pos, end_pos = match.span()

    # Split the sentence into words for context extraction
    words_before = sentence[:start_pos].split()
    words_after = sentence[end_pos:].split()

    # Calculate the number of words to include before and after the homograph
    num_words_before = min(len(words_before), around)
    num_words_after = min(len(words_after), around)

    # Extract the context around the homograph
    context_before = ' '.join(words_before[-num_words_before:])
    context_after = ' '.join(words_after[:num_words_after])

    # Reconstruct the sentence with the desired context around the homograph
    adjusted_sentence = f"{context_before} {homograph} {context_after}".strip()

    return adjusted_sentence


def train_and_evaluate(lr_clf, train_df, valid_df, args):
    best_f1_val = 0.0
    best_model_filename = f"best_trained_classifier_{args.model}_{args.around}.pkl"

    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch + 1}/{N_EPOCHS}")

        # Shuffle the training data at the beginning of each epoch
        train_df_shuffled = train_df.sample(frac=1).reset_index(drop=True)

        for i in tqdm(range(0, len(train_df_shuffled), BATCH_SIZE), desc="Training batches"):
            batch = train_df_shuffled.iloc[i:i + BATCH_SIZE]
            combined_embeddings = np.vstack(batch['combined_embeddings'].tolist())
            batch_labels = batch['label'].values

            lr_clf.partial_fit(combined_embeddings, batch_labels, classes=np.unique(train_df['label'].values))

        # Evaluate on the validation set after each epoch
        f1_val = evaluate(valid_df, lr_clf)
        print(f"Validation F1 score (after epoch {epoch + 1}): {f1_val:.4f}")

        # Save the model if it improves on the best F1 score observed so far
        if f1_val > best_f1_val:
            best_f1_val = f1_val
            with open(best_model_filename, 'wb') as model_file:
                pickle.dump(lr_clf, model_file)

    return best_model_filename


def main():
    args = parse_arguments()
    path = args.directory

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    bert_model, bert_tokenizer = get_model_and_tokenizer(args.model)

    bert_model = bert_model.to(device)
    train_df, valid_df, test_df = load_and_preprocess_data(bert_model=bert_model, bert_tokenizer=bert_tokenizer,
                                                           device=device,  directory_path=path, around=args.around,
                                                           force_reprocess=args.force, show_histogram=args.histogram)
    class_weights = compute_weights(train_df)

    # Initialize the logistic regression classifier with SGD
    lr_clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, alpha=ALPHA, class_weight=class_weights)

    # best classifier is saved and then evaluated on test set, finally being stored with resulting F1-score as part of
    # the filename
    best_model_filename = train_and_evaluate(lr_clf, train_df, valid_df, args)

    with open(best_model_filename, 'rb') as model_file:
        lr_clf = pickle.load(model_file)

    evaluate_and_save_model(lr_clf, test_df, args)


if __name__ == "__main__":
    main()
