# src/data_loader.py

import pandas as pd
import os
from typing import Optional, Tuple, Dict, Any
import json

FEWREL_LABELS = [
    "P1001",
    "P101",
    "P102",
    "P105",
    "P106",
    "P118",
    "P123",
    "P127",
    "P1303",
    "P131",
    "P1344",
    "P1346",
    "P135",
    "P136",
    "P137",
    "P140",
    "P1408",
    "P1411",
    "P1435",
    "P150",
    "P156",
    "P159",
    "P17",
    "P175",
    "P176",
    "P178",
    "P1877",
    "P1923",
    "P22",
    "P241",
    "P264",
    "P27",
    "P276",
    "P306",
    "P31",
    "P3373",
    "P3450",
    "P355",
    "P39",
    "P400",
    "P403",
    "P407",
    "P449",
    "P4552",
    "P460",
    "P466",
    "P495",
    "P527",
    "P551",
    "P57",
    "P58",
    "P6",
    "P674",
    "P706",
    "P710",
    "P740",
    "P750",
    "P800",
    "P84",
    "P86",
    "P931",
    "P937",
    "P974",
    "P991",
]

politics_labels = ['O', 'B-country', 'B-politician', 'I-politician', 'B-election', 'I-election', 'B-person', 'I-person', 'B-organisation', 'I-organisation', 'B-location', 'B-misc', 'I-location', 'I-country', 'I-misc', 'B-politicalparty', 'I-politicalparty', 'B-event', 'I-event']
science_labels = ['O', 'B-scientist', 'I-scientist', 'B-person', 'I-person', 'B-university', 'I-university', 'B-organisation', 'I-organisation', 'B-country', 'I-country', 'B-location', 'I-location', 'B-discipline', 'I-discipline', 'B-enzyme', 'I-enzyme', 'B-protein', 'I-protein', 'B-chemicalelement', 'I-chemicalelement', 'B-chemicalcompound', 'I-chemicalcompound', 'B-astronomicalobject', 'I-astronomicalobject', 'B-academicjournal', 'I-academicjournal', 'B-event', 'I-event', 'B-theory', 'I-theory', 'B-award', 'I-award', 'B-misc', 'I-misc']
music_labels = ['O', 'B-musicgenre', 'I-musicgenre', 'B-song', 'I-song', 'B-band', 'I-band', 'B-album', 'I-album', 'B-musicalartist', 'I-musicalartist', 'B-musicalinstrument', 'I-musicalinstrument', 'B-award', 'I-award', 'B-event', 'I-event', 'B-country', 'I-country', 'B-location', 'I-location', 'B-organisation', 'I-organisation', 'B-person', 'I-person', 'B-misc', 'I-misc']
literature_labels = ["O", "B-book", "I-book", "B-writer", "I-writer", "B-award", "I-award", "B-poem", "I-poem", "B-event", "I-event", "B-magazine", "I-magazine", "B-literarygenre", "I-literarygenre", 'B-country', 'I-country', "B-person", "I-person", "B-location", "I-location", 'B-organisation', 'I-organisation', 'B-misc', 'I-misc']
ai_labels = ["O", "B-field", "I-field", "B-task", "I-task", "B-product", "I-product", "B-algorithm", "I-algorithm", "B-researcher", "I-researcher", "B-metrics", "I-metrics", "B-programlang", "I-programlang", "B-conference", "I-conference", "B-university", "I-university", "B-country", "I-country", "B-person", "I-person", "B-organisation", "I-organisation", "B-location", "I-location", "B-misc", "I-misc"]

domain2labels = {"politics": politics_labels, "science": science_labels, "music": music_labels, "literature": literature_labels, "ai": ai_labels}



class Dataset:
    """
    Represents a dataset, handling loading and preprocessing for use with
    models like those in simpletransformers.

    Attributes:
        name (str): An identifier for the dataset (e.g., 'imdb', 'custom_csv').
                    Determines the loading strategy.
        config (Dict[str, Any]): Configuration details needed for loading,
                                 e.g., file paths, column names.
        train_df (Optional[pd.DataFrame]): DataFrame for training data.
                                           Expected columns: 'text', 'labels'.
        test_df (Optional[pd.DataFrame]): DataFrame for testing/evaluation data.
                                          Expected columns: 'text', 'labels'.
        num_labels (Optional[int]): The number of unique labels in the training data.
        error (Optional[str]): Stores an error message if loading fails.
    """
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initializes the Dataset object.

        Args:
            name (str): The identifier for the dataset.
            config (Dict[str, Any]): Configuration dictionary. Contents depend
                                     on the dataset name/type.
                                     Example for CSV:
                                     {
                                         'train_path': 'path/to/train.csv',
                                         'test_path': 'path/to/test.csv',
                                         'sentence_col': 'original_text_column_name',
                                         'label_col': 'original_label_column_name',
                                         'separator': ',',
                                         'encoding': 'utf-8',
                                         'nrows_train': None, # Optional: limit rows read
                                         'nrows_test': None   # Optional: limit rows read
                                     }
        """
        self.name = name
        self.config = config
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.num_labels: Optional[int] = None
        self.error: Optional[str] = None
        print(f"Dataset object created for '{self.name}'. Call load() to load data.")

    def load(self) -> bool:
        """
        Loads the training and testing data based on the dataset name and config.

        Populates self.train_df, self.test_df, and self.num_labels.
        Sets self.error if loading fails.

        Returns:
            bool: True if data loading was successful, False otherwise.
        """
        print(f"Attempting to load data for dataset: '{self.name}'")
        try:
            # --- Select loading strategy based on name ---
            if self.name == 'custom_csv':
                self._load_csv_data()

            elif self.name == 'sst2':
                self._load_sst2_data()

            elif self.name == 'fewrel':
                self._load_fewrel_data()

            elif self.name == 'crossner':
                self._load_crossner_data()

            else:
                self.error = f"Error: Unknown dataset name '{self.name}'. No loading strategy defined."
                print(self.error)
                return False

            # --- Post-loading checks and processing ---
            if self.error: # Check if private loader set an error
                 print(f"Loading failed for '{self.name}': {self.error}")
                 return False

            if self.train_df is None:
                self.error = "Error: Training data was not loaded."
                print(self.error)
                return False # Training data is essential

            # Ensure required columns exist
            if 'sentence' not in self.train_df.columns or 'label' not in self.train_df.columns:
                 self.error = "Error: Loaded training data missing required 'sentence' or 'labels' columns."
                 print(self.error)
                 return False
            if self.test_df is not None and ('sentence' not in self.test_df.columns or 'label' not in self.test_df.columns):
                 self.error = "Error: Loaded test data missing required 'sentence' or 'label' columns."
                 print(self.error)
                 # Allow loading to succeed even if test data is missing/invalid,
                 # but log the error. Training might still be possible.
                 # return False # Uncomment if test data is strictly required

            # Calculate number of labels from training data
            self.num_labels = self.train_df['label'].nunique()
            print(f"Calculated number of unique labels: {self.num_labels}")

            print(f"Data loading successful for '{self.name}'.")
            print(f"  Training samples: {len(self.train_df)}")
            if self.test_df is not None:
                print(f"  Testing samples: {len(self.test_df)}")
            else:
                print("  Testing samples: 0 (or not loaded)")

            return True

        except Exception as e:
            self.error = f"An unexpected error occurred during loading: {e}"
            import traceback
            print(f"{self.error}\n{traceback.format_exc()}")
            return False

    def _load_single_csv(
        self,
        file_path: Optional[str],
        text_col: str,
        label_col: str,
        separator: str,
        encoding: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Internal helper to load and process a single CSV file.
        (Adapted from the previous standalone load_data function)
        """
        if not file_path:
            return None, "Info: No file path provided for this split (e.g., test set might be optional)."

        # --- File Existence Check ---
        if not os.path.exists(file_path):
            return None, f"Error: File not found at path: {file_path}"

        # --- Data Loading ---
        try:
            print(f"  Loading from: {file_path}")
            df = pd.read_csv(
                file_path,
                sep=separator,
                encoding=encoding
            )
            print(f"  Successfully loaded {len(df)} rows from {os.path.basename(file_path)}.")

        except FileNotFoundError:
            return None, f"Error: File not found at path: {file_path}"
        except pd.errors.EmptyDataError:
            return None, f"Error: File is empty: {file_path}"
        except Exception as e:
            return None, f"Error loading file {file_path}: {e}"

        # --- Column Validation ---
        if text_col not in df.columns:
            return None, f"Error: Text column '{text_col}' not found in {file_path}."
        if label_col not in df.columns:
            return None, f"Error: Label column '{label_col}' not found in {file_path}."

        # --- Select and Rename Columns ---
        df_processed = df[[text_col, label_col]].copy()
        df_processed.rename(columns={text_col: 'sentence', label_col: 'label'}, inplace=True)
        required_cols = ['sentence', 'label']

        # --- Data Type Conversion (Important for simpletransformers) ---
        # df_processed['sentence'] = df_processed['sentence'].astype(str)
        # Labels are often integers, but could be strings for multi-label
        # Let simpletransformers handle label type specifics later, or adjust here if needed.
        # df_processed['label'] = df_processed['label'].astype(int) # Example if labels are always ints

        return df_processed, None

    def _load_sst2_data(self):
        """
        Loads training and testing data for SST2 data based on self.config.
        """
        print("Loading SST2 data...")
        # --- Get config parameters ---
        train_path  = os.path.join(self.config.get('train_path'), 'train.tsv')
        test_path   = os.path.join(self.config.get('test_path'), 'test.tsv') # Test path is optional
        text_col    = 'sentence'
        label_col   = 'label'
        separator   = '\t'
        encoding    = 'utf-8'

        if not train_path or not test_path:
            self.error = "Error: Missing required config for data loading: 'train_path', 'test_path'."
            print(self.error)
            return

        # --- Load Training Data ---
        self.train_df, train_error = self._load_single_csv(
            train_path, text_col, label_col, separator, encoding
        )
        if train_error:
            print(train_error)
            return

        # --- Load Test Data ---
        self.test_df, test_error = self._load_single_csv(
            test_path, text_col, label_col, separator, encoding
        )
        if test_error:
            print(test_error)
            return

    def _load_fewrel_data(self):
        print("Loading FewRel data...")
        # --- Get config parameters ---
        train_path  = os.path.join(self.config.get('train_path'), 'train.json')
        test_path   = os.path.join(self.config.get('test_path'), 'test.json') # Test path is optional
        text_col    = 'sentence'
        label_col   = 'label'
        separator   = '\t'
        encoding    = 'utf-8'

        if not train_path or not test_path:
            self.error = "Error: Missing required config for data loading: 'train_path', 'test_path'."
            print(self.error)
            return

        # --- Load Training Data --- #
        self.train_df, train_error = self._process_fewrel_data(train_path)

        # --- Load Test Data ---
        self.test_df, test_error = self._process_fewrel_data(test_path)

        if train_error:
            print(train_error)
            return

        if test_error:
            print(test_error)
            return

    def _load_csv_data(self):
        """
        Loads training and testing data from CSV files based on self.config.
        """
        print("Executing CSV loading strategy...")
        # --- Get config parameters ---
        train_path = self.config.get('train_path')
        test_path = self.config.get('test_path') # Test path is optional
        text_col = self.config.get('text_col')
        label_col = self.config.get('label_col')
        separator = self.config.get('separator', ',')
        encoding = self.config.get('encoding', 'utf-8')
        nrows_train = self.config.get('nrows_train', None)
        nrows_test = self.config.get('nrows_test', None)

        # --- Validate required config ---
        if not train_path or not text_col or not label_col:
            self.error = "Error: Missing required config for CSV loading: 'train_path', 'text_col', 'label_col'."
            print(self.error)
            return

        # --- Load Training Data ---
        self.train_df, train_error = self._load_single_csv(
            train_path, text_col, label_col, separator, encoding, nrows_train
        )
        if train_error:
            # Distinguish between file not found (error) and other issues
            if "File not found" in train_error:
                 self.error = train_error # Make it a critical error
            else:
                 print(f"Warning during train data loading: {train_error}") # Log non-critical errors
                 # Decide if partial success is okay, e.g., if train_df is partially loaded
                 if self.train_df is None:
                     self.error = f"Failed to load essential training data: {train_error}"
                     return # Stop if training data failed critically

        # --- Load Testing Data (Optional) ---
        if test_path:
            self.test_df, test_error = self._load_single_csv(
                test_path, text_col, label_col, separator, encoding, nrows_test
            )
            if test_error:
                # Log error but don't necessarily stop the whole process
                print(f"Warning during test data loading: {test_error}")
                self.test_df = None # Ensure test_df is None if loading failed
        else:
            print("Info: No 'test_path' provided in config. Skipping test data loading.")
            self.test_df = None


    # ------ CrossNER Data Functions ------ #
    def load_crossner_sentences(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()

        sentences = []
        current_sentence = []

        for line in lines:
            if line.strip() == '':  # Empty line indicates end of sentence
                if current_sentence:
                    sentences.append(' '.join(current_sentence))
                    current_sentence = []
            else:
                word, _ = line.strip().split()  # Extract word, ignore label
                current_sentence.append(word)

        # Handle potential last sentence without an empty line
        if current_sentence:
            sentences.append(' '.join(current_sentence))

        return pd.DataFrame({"sentence": sentences})

    def load_crossner_train(self, path, dev=False):
        with open(path, 'r') as f:
            lines = f.readlines()

        data = []
        sentence_id = 0
        if dev: 
            sentence_id += 2700

        for line in lines:
            if line.strip() == '':  # Empty line indicates end of sentence
                sentence_id += 1
            else:
                token, label = line.strip().split()  # Extract word, ignore label
                data.append([sentence_id, token, label])
        return data


    # ------ FewRel Data Functions ------- #


    def _process_fewrel_data(self, path):
        pairs = []
        with open(path) as f:
            raw = json.load(f)
            for label, lst in raw.items():
                y = FEWREL_LABELS.index(label)
                for sample in lst:
                    text, head, tail = read_sample_dict(sample)
                    x = linearize_input(text, head, tail)
                    pairs.append((x, y))

        df = pd.DataFrame(pairs)
        df.columns = ["sentence", "label"]
        df = df.sample(frac=1)  # Shuffle
        print(dict(path=path, data=df.shape, unique_labels=len(set(df["label"].tolist()))))
        return df, None


def linearize_input(text, head, tail):
    return f"Head Entity : {head} , Tail Entity : {tail} , Context : {text}"

def read_sample_dict(sample):
    tokens = sample["tokens"]
    head = " ".join([tokens[i] for i in sample["h"][2][0]])
    tail = " ".join([tokens[i] for i in sample["t"][2][0]])
    return " ".join(tokens), head, tail

# --- Example Usage ---
if __name__ == '__main__':
    # --- Setup Dummy Data ---
    dummy_train_data = {
        'review_text': ['Loved it!', 'Absolutely terrible.', 'It was okay.', 'Best movie ever.', None],
        'rating': [1, 0, 0, 1, 0],
        'user_id': [101, 102, 103, 104, 105]
    }
    dummy_test_data = {
        'review_text': ['Decent watch.', 'Would not recommend.', 'Amazing!'],
        'rating': [0, 0, 1],
        'user_id': [201, 202, 203]
    }
    train_file = 'dummy_train.csv'
    test_file = 'dummy_test.csv'
    pd.DataFrame(dummy_train_data).to_csv(train_file, index=False)
    pd.DataFrame(dummy_test_data).to_csv(test_file, index=False)

    # --- Configuration for the dummy CSV dataset ---
    csv_config = {
        'train_path': train_file,
        'test_path': test_file,
        'text_col': 'review_text',
        'label_col': 'rating',
        'separator': ',',
        'encoding': 'utf-8'
    }

    print("\n--- Testing Dataset class with 'custom_csv' ---")
    # 1. Instantiate the Dataset
    my_dataset = Dataset(name='custom_csv', config=csv_config)

    # 2. Load the data
    success = my_dataset.load()

    # 3. Check results
    if success:
        print("\n--- Loading Successful ---")
        print("Dataset Name:", my_dataset.name)
        print("Number of Labels:", my_dataset.num_labels)

        print("\nTrain DataFrame Head:")
        if my_dataset.train_df is not None:
            print(my_dataset.train_df.head())
            print("\nTrain DataFrame Info:")
            my_dataset.train_df.info()
        else:
            print("Train DataFrame is None.")

        print("\nTest DataFrame Head:")
        if my_dataset.test_df is not None:
            print(my_dataset.test_df.head())
            print("\nTest DataFrame Info:")
            my_dataset.test_df.info()
        else:
            print("Test DataFrame is None or was not loaded.")

    else:
        print("\n--- Loading Failed ---")
        print("Error:", my_dataset.error)

    # --- Test unknown dataset name ---
    print("\n--- Testing unknown dataset name ---")
    unknown_dataset = Dataset(name='unknown_format', config={})
    unknown_success = unknown_dataset.load()
    print(f"Loading success status: {unknown_success}")
    print(f"Error message: {unknown_dataset.error}")


    # --- Clean up dummy files ---
    print("\n--- Cleaning up dummy files ---")
    try:
        if os.path.exists(train_file):
            os.remove(train_file)
            print(f"Removed: {train_file}")
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"Removed: {test_file}")
    except OSError as e:
        print(f"Error removing dummy files: {e}")
