import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import re
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import transformers as ppb

BATCH_SIZE = 512
TEST_SIZE = 0.1
VALIDATION_SIZE = 0.1
HOMOGRAPH_RE = r'\[\[(.*?)\]\]'
LABEL_RE = r'[\s\t]+(0|1)$'
HOMOGRAPHS = \
        ["alla", "bolla", "böll", "bollum", "böllum", "bollunum", "dalla", "dill", "dillum", "drolla",
         "ella", "elli", "gallana", "gallanna", "gallann", "gallans", "gallanum", "gallar", "galla", "gallinn",
         "galli", "gella", "gellur", "göllum", "grilla", "grillir", "grilli", "gulla", "gulli", "gullu",
         "halla", "halli", "halló", "holla", "holli", "holl", "hollum", "kalla", "kalli", "kolla", "kollu",
         "lalla", "lalli", "mallar", "malla", "malli", "milli", "möllum", "ollu", "palla", "palli",
         "pollana", "pollanna", "pollarnir", "pollar", "polla", "pollinn", "polli", "pollum", "ullar", "villanna",
         "villan", "villa", "villi", "villum", "villuna", "villunnar", "villunni", "villunum", "villurnar", "villur",
         "villu"]

# extremely seldom and/or different UTF-8 encoding
EXTRA_HOMOGRAPHS = ["gallarnir", "göllunum", "böllum", "halló", "möllum"]


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
    for file_path in Path(directory_path).glob('*.*'):
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


def load_and_preprocess_data(bert_model, bert_tokenizer, device, directory_path, around, force_reprocess=False,
                             show_histogram=False, args=None):
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
        args: if given, uses them for determining the value for --no-cls

    Returns:
        tuple: Three DataFrames corresponding to the training, validation, and test sets, each with sentences,
               labels, group IDs, and calculated embeddings.
    """
    model_name = bert_model.name_or_path.replace('/', '_')
    # Adjust cache file naming based on args
    additional_cache_info = f"{'_no_cls' if args and args.no_cls else ''}"
    embeddings_cache_file = Path(directory_path) / f"{model_name}_{around}_embeddings{additional_cache_info}.pkl"

    # Check for existing embeddings cache
    if embeddings_cache_file.exists() and not force_reprocess:
        print(f"Loading embeddings from cache file {embeddings_cache_file} ...")
        with open(embeddings_cache_file, 'rb') as ecf:
            train_set, valid_set, test_set, valid_set_unbalanced, test_set_unbalanced = pickle.load(ecf)
    else:
        df = load_data_from_directory(directory_path)
        print("Adjust sentence and remove homograph labels ...")
        df['adjusted_sentence'] = df['sentence'].apply(lambda x: get_context_around_homograph(x, around))
        df['adjusted_sentence_marked'] = \
            df['sentence'].apply(lambda x: get_context_around_homograph(x, around, True))
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

        # drop only temporarily needed columns
        df.drop(columns=['adjusted_sentence_marked', 'sentence'], inplace=True)

        # Calculate embeddings
        print("Create splits (balanced, unbalanced) of the data ...")
        train_set, valid_set, test_set, valid_set_unbalanced, test_set_unbalanced = \
            create_balanced_and_unbalanced_splits(df, TEST_SIZE, VALIDATION_SIZE, show_histogram)

        train_set, valid_set, test_set, valid_set_unbalanced, test_set_unbalanced = \
            precalculate_embeddings(bert_model, device, train_set, valid_set, test_set,
                                    valid_set_unbalanced, test_set_unbalanced, BATCH_SIZE, args.no_cls)

        # Save calculated embeddings to cache
        print("Saving calculated embeddings to cache ...")
        with open(embeddings_cache_file, 'wb') as ecf:
            pickle.dump((train_set, valid_set, test_set, valid_set_unbalanced, test_set_unbalanced), ecf)

    return train_set, valid_set, test_set, valid_set_unbalanced, test_set_unbalanced


def preprocess_and_tokenize_data(row, tokenizer):
    # Tokenize the adjusted sentence without markup
    encoded_adjusted = tokenizer.encode_plus(row['adjusted_sentence'], add_special_tokens=True, return_tensors='pt',
                                             padding='max_length', truncation=True, max_length=512)
    input_ids_adjusted = encoded_adjusted['input_ids'].squeeze()

    # Extract the homograph from the adjusted marked sentence
    marked_homograph_match = re.search(HOMOGRAPH_RE, row['adjusted_sentence_marked'])
    if not marked_homograph_match:
        raise ValueError("Marked homograph not found in the adjusted marked sentence.")

    # Tokenize the homograph to find its token IDs
    homograph = marked_homograph_match.group(1)
    homograph_tokens = tokenizer.tokenize(homograph)
    homograph_ids = tokenizer.convert_tokens_to_ids(homograph_tokens)

    # Tokenize the part of the sentence up to the homograph to estimate start position
    pre_homograph_text = row['adjusted_sentence_marked'][:marked_homograph_match.start(1)]
    pre_homograph_encoded = tokenizer.encode_plus(pre_homograph_text, add_special_tokens=True, return_tensors='pt',
                                                  truncation=True)
    pre_homograph_ids = pre_homograph_encoded['input_ids'].squeeze()

    # -3 to account for the separation special token at the end and the 2 opening brackets for the homograph markup
    # itself
    homograph_start_pos = len(pre_homograph_ids) - 3

    # Verify the expected tokens match the actual tokens at calculated positions
    actual_homograph_ids = input_ids_adjusted[homograph_start_pos:homograph_start_pos + len(homograph_ids)]
    if not all(actual == expected for actual, expected in zip(actual_homograph_ids, homograph_ids)):
        raise ValueError(
            f"Mismatch in token IDs for homograph '{homograph}' at calculated positions in"
            f" sentence: '{row['adjusted_sentence']}'")

    # Calculate positions assuming each token ID in homograph_ids occupies one position
    homograph_positions = list(range(homograph_start_pos, homograph_start_pos + len(homograph_ids)))

    return {
        'input_ids': input_ids_adjusted.unsqueeze(0),
        'attention_mask': encoded_adjusted['attention_mask'],
        'homograph_positions': homograph_positions
    }


def get_context_around_homograph(sentence, around, do_mark=False):
    """
    Adjusts the sentence to focus on the context around the marked homograph.

    Args:
        sentence (str): The original sentence containing a marked homograph with [[...]].
        around (int): The maximum number of tokens around the homograph to include.
        do_mark(bool): If true, return the adjusted sentence with the homograph marked as in the input sentence

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
    if do_mark:
        adjusted_sentence = f"{context_before} [[{homograph}]] {context_after}".strip()
    else:
        adjusted_sentence = f"{context_before} {homograph} {context_after}".strip()
    return adjusted_sentence


def precalculate_embeddings(bert_model, device, train_set, valid_set, test_set, unbalanced_valid_set,
                            unbalanced_test_set, batch_size, no_cls=False):
    bert_model.eval()
    datasets = {'train': train_set, 'valid': valid_set, 'test': test_set, 'unbalanced_valid': unbalanced_valid_set,
                'unbalanced_test': unbalanced_test_set}

    for name, df in datasets.items():
        combined_embeddings = []

        for i in tqdm(range(0, len(df), batch_size), desc=f"Creating embeddings for {name} set"):
            batch = df.iloc[i:i + batch_size]
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

                    if no_cls:
                        combined_embedding = homograph_embedding.unsqueeze(0)
                    else:
                        combined_embedding = torch.cat((cls_embedding.unsqueeze(0), homograph_embedding.unsqueeze(0)),
                                                       dim=1)
                    combined_embeddings.append(combined_embedding.cpu().numpy())
                else:
                    raise RuntimeError(f"No valid homograph embedding detected !")

            # Clearing unused GPU memory
            del input_ids, attention_mask, positions, sequence_output
            torch.cuda.empty_cache()

        datasets[name]['combined_embeddings'] = pd.Series(combined_embeddings)

    return datasets['train'], datasets['valid'], datasets['test'], \
        datasets['unbalanced_valid'], datasets['unbalanced_test']


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
    elif model_name in ['sbert-ruquad', 'icelandic-ner-bert', 'labse', 'convbert']:
        model_class, tokenizer_class = AutoModel, AutoTokenizer
        pretrained_weights = {
            'sbert-ruquad': 'language-and-voice-lab/sbert-ruquad',
            'icelandic-ner-bert': 'grammatek/icelandic-ner-bert',
            'labse': 'setu4993/LaBSE',
            'convbert': 'jonfd/convbert-base-igc-is',
        }[model_name]
    else:
        raise ValueError(f"Unsupported model {model_name}")

    print(f"Loading {model_name} model ...")
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    return model, tokenizer


def find_max_tokens(homographs, tokenizer):
    """
    Determines the maximum number of tokens for any homograph in the list.

    Args:
        homographs (list of str): A list of homograph strings.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer corresponding to the BERT model being used.

    Returns:
        int: The maximum number of tokens found for any homograph in the list.
    """
    max_tokens = 0
    for homograph in homographs:
        # Tokenize the homograph and count its tokens
        tokens = tokenizer.tokenize(homograph)
        num_tokens = len(tokens)
        # Update max_tokens if this homograph has more tokens
        if num_tokens > max_tokens:
            max_tokens = num_tokens
    return max_tokens


def compute_weights(df):
    """
    Computes class weights based on the imbalance in the dataset.

    Args:
        df (pd.DataFrame): The dataframe containing the labels for each class.

    Returns:
        dict: A dictionary with class weights.
    """
    class_weights = compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'])
    return {i: weight for i, weight in enumerate(class_weights)}


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
    balanced_group_df = pd.concat([
        group_df[group_df['label'] == 0].sample(n=min_class_count, random_state=42),
        group_df[group_df['label'] == 1].sample(n=min_class_count, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_group_df


def adjusted_train_test_split(group_id, *arrays, **options):
    stratify = options.get('stratify', None)
    if stratify is not None:
        unique, counts = np.unique(stratify, return_counts=True)
        min_counts = np.min(counts)

        # Check if stratification is feasible
        if min_counts < 2:
            raise RuntimeError("Stratification not feasible due to class with less than 2 instances.")
        else:
            test_size = options.get('test_size', 0.25)
            # Ensure there's at least one instance of each class in both splits
            required_min_size = len(unique) * 2 / len(stratify)
            if isinstance(test_size, float) and test_size < required_min_size:
                adjusted_test_size = max(test_size, required_min_size)
                print(
                    f"{group_id}: Adjusting test_size from {test_size} to {adjusted_test_size}"
                    f" due to stratification requirements.")
                options['test_size'] = adjusted_test_size
            elif isinstance(test_size, int):
                # When test_size is an int, ensure the dataset can be split accordingly
                adjusted_test_size = min(test_size, max(1, len(stratify) - len(unique)))
                if test_size != adjusted_test_size:
                    print(f"{group_id}: Adjusting test_size from {test_size} to {adjusted_test_size}"
                          f" due to dataset size and stratification requirements.")
                    options['test_size'] = adjusted_test_size

    return train_test_split(*arrays, **options)


def create_balanced_and_unbalanced_splits(df, test_size=TEST_SIZE, valid_size=VALIDATION_SIZE, show_histogram=False):
    all_train_dfs_balanced = []
    all_valid_dfs_balanced = []
    all_test_dfs_balanced = []
    all_valid_dfs_unbalanced = []
    all_test_dfs_unbalanced = []

    for group_id in df['group_id'].unique():
        group_df = df[df['group_id'] == group_id]

        try:
            # Splitting group data into unbalanced train+valid and test with stratification
            train_valid_df_unbalanced, test_df_unbalanced = train_test_split(
                group_df,
                test_size=test_size,
                random_state=42,
                stratify=group_df['label']
            )

            # Now, split unbalanced train_valid_df into actual train and valid sets with stratification
            train_df_unbalanced, valid_df_unbalanced = train_test_split(
                train_valid_df_unbalanced,
                test_size=valid_size / (1 - test_size),
                random_state=42,
                stratify=train_valid_df_unbalanced['label']
            )
        except Exception as e:
            print(f"{group_id}: skipped, {e}")
            continue

        # Balance the train, valid, and test sets
        try:
            train_df_balanced = balance_labels_within_group(train_df_unbalanced)
            valid_df_balanced = balance_labels_within_group(valid_df_unbalanced)
            test_df_balanced = balance_labels_within_group(test_df_unbalanced)
        except Exception as e:
            print(f"{group_id}: skipped, due to exception for balancing training set: {e}")
            continue

        # Append to respective lists
        all_train_dfs_balanced.append(train_df_balanced)
        all_valid_dfs_balanced.append(valid_df_balanced)
        all_test_dfs_balanced.append(test_df_balanced)
        all_valid_dfs_unbalanced.append(valid_df_unbalanced)
        all_test_dfs_unbalanced.append(test_df_unbalanced)

    # Combine and shuffle the final datasets
    final_train_df_balanced = pd.concat(all_train_dfs_balanced).sample(frac=1, random_state=42).reset_index(drop=True)
    final_valid_df_balanced = pd.concat(all_valid_dfs_balanced).sample(frac=1, random_state=42).reset_index(drop=True)
    final_test_df_balanced = pd.concat(all_test_dfs_balanced).sample(frac=1, random_state=42).reset_index(drop=True)
    final_valid_df_unbalanced = pd.concat(all_valid_dfs_unbalanced).sample(frac=1, random_state=42).reset_index(
        drop=True)
    final_test_df_unbalanced = pd.concat(all_test_dfs_unbalanced).sample(frac=1, random_state=42).reset_index(drop=True)

    if show_histogram:
        print("Balanced Training Set Group ID Frequencies:")
        print_group_id_frequencies(final_train_df_balanced)
        print("Balanced Validation Set Group ID Frequencies:")
        print_group_id_frequencies(final_valid_df_balanced)
        print("Balanced Test Set Group ID Frequencies:")
        print_group_id_frequencies(final_test_df_balanced)
        print("Unbalanced Validation Set Group ID Frequencies:")
        print_group_id_frequencies(final_valid_df_unbalanced)
        print("Unbalanced Test Set Group ID Frequencies:")
        print_group_id_frequencies(final_test_df_unbalanced)

    return (final_train_df_balanced, final_valid_df_balanced, final_test_df_balanced, final_valid_df_unbalanced,
            final_test_df_unbalanced)


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
