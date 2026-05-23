"""
Data preprocessing module for NLP ML Pipeline.
Loads the data
cleans the data
Splits the data to test and training data depending on how we want to train and evaluate
Balances the training data
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string
import numpy as np

from utils.config import (
    TRAIN_TEST_SPLIT_SIZE,
    RANDOM_STATE,
    DATA_PATH,
    SMS_FILE,
    EMAIL_FILE,
    MESSAGE,
    LABEL
)

from utils.logger import get_logger, LogLevel


class DataProcessor:
    """
    Data processor for NLP datasets.
    
    Handles loading, cleaning, and preprocessing of NLP data
    """
    
    def __init__(self):
        """Initialize the data processor."""
        self.train_data = None
        self.test_data = None

    def load_data(self):
        """
        Load training and test datasets from CSV files.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        logger = get_logger()
        logger.substep("Loading Data")
        
        # Charger en sautant les lignes malformées
        self.train_data = pd.read_csv(f"{DATA_PATH}/{SMS_FILE}", sep=';', usecols=[0,1])
        self.train_data.columns = [LABEL, MESSAGE]
        self.test_data = pd.read_csv(f"{DATA_PATH}/{EMAIL_FILE}")

        # Logging
        with logger.indent():
            logger.dataframe_info(self.train_data, "SMS training data")
            logger.dataframe_info(self.test_data, "Email test data")
        
        logger.success("Data loading completed")
        return self.train_data.copy(), self.test_data.copy()
    
    def drop_duplicates(self, data):
        """
        Drops the duplicates from the data
        
        Returns:
            the data with no duplicates
        """
        data = data.drop_duplicates(keep='first', ignore_index=True)
        return data
    
    def extract_msgs_labels(self, data):
        """
        Extracts messages and labels from the data
        Returns:
            a tuple of messages and labels
        """
        return data[MESSAGE], data[LABEL]
    
    def train_eval_same_data(self, data_messages, data_labels):
        """
        Split the dataset data into training and testing sets.
    
        Creates a standard train-test split for the data.
        Uses stratified sampling to maintain class distribution across splits.
    
        Returns
        -------
        tuple
            Tuple containing (train_messages, test_messages, train_labels, test_labels)
        
        Notes
        -----
        Uses global TRAIN_TEST_SPLIT_SIZE and RANDOM_STATE for consistency
        across all experiments.
        """
        data_msg_train, data_msg_test, data_lable_train, data_lable_test = train_test_split(
            data_messages, data_labels, test_size=TRAIN_TEST_SPLIT_SIZE, 
            random_state=RANDOM_STATE, stratify=data_labels
        )
        return (data_msg_train, data_msg_test, data_lable_train, data_lable_test)

    def train_data1_eval_data2(self, data1_messages, data1_labels, data2_messages, data2_labels):
        """
        Prepare data for cross-domain transfer learning experiment.
    
        Uses entire data1 dataset for training and entire data2 dataset for testing.
        This setup evaluates model generalization across different text domains
        and communication channels.
    
        Returns
        -------
        tuple
            Tuple containing (train_messages, test_messages, train_labels, test_labels)
            where training data comes from data1 and testing data from data2
        
        Notes
        -----
        No random splitting is performed as we use complete datasets for 
        cross-domain evaluation.
        """
        return (data1_messages, data1_labels, data2_messages, data2_labels)

    def train_eval_combined(self, data1, data2):
        """
        Combine SMS (data1) and email(data2) datasets for unified training and testing.
    
        Merges both datasets and creates a mixed train-test split. This approach
        evaluates whether combining different text domains improves overall
        spam detection performance.
    
        Returns
        -------
        tuple
            Tuple containing (train_messages, test_messages, train_labels, test_labels)
            from the combined dataset
        
        Notes
        -----
        Uses pandas.concat to merge datasets while preserving all data points.
        Maintains class balance across the combined dataset.
        """
        merged = pd.concat([data1, data2], axis=0, ignore_index=True)
        merged_messages = merged[MESSAGE]
        merged_labels = merged[LABEL]
        merged_msg_train, merged_msg_test, merged_lable_train, merged_lable_test = train_test_split(
            merged_messages, merged_labels, test_size=TRAIN_TEST_SPLIT_SIZE, 
            random_state=RANDOM_STATE, stratify=merged_labels
        )
        return (merged_msg_train, merged_msg_test, merged_lable_train, merged_lable_test)

    def balance(self, training_messages, training_labels):
        """
        Balance training data by oversampling the minority class.
    
        Addresses class imbalance by randomly sampling additional instances
        from the underrepresented class until both classes have equal frequency.
        This prevents model bias toward the majority class.
    
        Parameters
        ----------
        training_messages : pandas.Series
            Training text messages
        training_labels : pandas.Series  
            Corresponding class labels (0 for ham, 1 for spam)
        
        Returns
        -------
        tuple
            Tuple containing (balanced_messages, balanced_labels) with equal
            class representation
        
        Notes
        -----
        Uses random sampling with replacement to increase minority class size.
        Preserves original data distribution while achieving balance.
        """
        logger = get_logger()
        logger.substep("Balancing training data", level=LogLevel.NORMAL)
    
        # Label counts before balancing
        counts_before = training_labels.value_counts()
        with logger.indent():
            logger.info(f"Ham (0): {counts_before.get(0, 0)} samples")
            logger.info(f"Spam (1): {counts_before.get(1, 0)} samples")
       
        counts = training_labels.value_counts()
        if counts[1] > counts[0]:
            label_to_oversample = 0
            diff = counts[1] - counts[0]
        else:
            label_to_oversample = 1
            diff = counts[0] - counts[1]

        logger.info(f"Oversampling class {label_to_oversample} by {diff} samples")

        training_data = pd.concat([training_messages, training_labels], axis=1)
        draw_from = training_data[training_data[LABEL] == label_to_oversample]

        for i in range(diff):
            sample = draw_from.sample(random_state=RANDOM_STATE)
            training_data = pd.concat([training_data, sample])

        training_messages = training_data[MESSAGE]
        training_labels = training_data[LABEL]

        # Label counts after balancing
        counts_after = training_labels.value_counts()
        with logger.indent():
            logger.info(f"Ham (0): {counts_after.get(0, 0)} samples")
            logger.info(f"Spam (1): {counts_after.get(1, 0)} samples")
    
        logger.success("Balancing completed")
    
        return training_messages, training_labels
