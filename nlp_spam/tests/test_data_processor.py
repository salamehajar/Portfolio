"""
Technical validation tests for DataProcessor class (NLP Pipeline).

These tests validate the technical implementation without focusing on ML performance.
They check that methods execute correctly and produce expected data structures.
"""
# To run data_processor test 
#uv run python scripts/run_tests.py --module data_processor --verbose
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import tempfile

# Assuming the package structure from the provided code
from pipeline.data_processor import DataProcessor
from utils.config import (
    MESSAGE,
    LABEL,
    TRAIN_TEST_SPLIT_SIZE,
    RANDOM_STATE
)


class TestDataProcessor:
    """Test suite for DataProcessor technical validation."""
    
    @pytest.fixture
    def sample_spam_data(self):
        """Create sample spam/ham data for testing."""
        # Create realistic SMS-like messages
        messages = [
            "Free entry in a prize draw! Click here to win!",
            "Hey, are we still meeting for lunch tomorrow?",
            "WINNER! You have been selected for a cash prize",
            "Can you pick up milk on your way home?",
            "Congratulations! You've won a free vacation",
            "Meeting rescheduled to 3pm in conference room B",
            "URGENT! Your account needs verification NOW",
            "Thanks for yesterday, had a great time!",
            "Claim your FREE gift card worth $500 today",
            "Don't forget mom's birthday is next week"
        ] * 10  # Repeat to get 100 samples
        
        # Create labels (0 = ham, 1 = spam)
        labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 10
        
        df = pd.DataFrame({
            MESSAGE: messages,
            LABEL: labels
        })
        
        return df
    
    @pytest.fixture
    def imbalanced_data(self):
        """Create imbalanced dataset for testing balancing method."""
        # 80 ham messages, 20 spam messages (80/20 split)
        ham_messages = ["This is a legitimate message"] * 80
        spam_messages = ["FREE MONEY CLICK HERE!!!"] * 20
        
        messages = ham_messages + spam_messages
        labels = [0] * 80 + [1] * 20
        
        df = pd.DataFrame({
            MESSAGE: messages,
            LABEL: labels
        })
        
        return df
    
    @pytest.fixture
    def data_with_duplicates(self):
        """Create dataset with duplicate entries."""
        messages = [
            "Hello there",
            "Hello there",  # duplicate
            "Free prize!",
            "Meeting at 3pm",
            "Free prize!",  # duplicate
            "Hello there",  # duplicate
        ]
        labels = [0, 0, 1, 0, 1, 0]
        
        df = pd.DataFrame({
            MESSAGE: messages,
            LABEL: labels
        })
        
        return df
    
    @pytest.fixture
    def processor_with_data(self, sample_spam_data, tmp_path, monkeypatch):
        """Create DataProcessor with sample data files."""
        
        # Split data into train and test
        train_data = sample_spam_data.iloc[:80].copy()
        test_data = sample_spam_data.iloc[80:].copy()
        
        # Save to temporary files
        train_path = tmp_path / "train.csv"
        test_path = tmp_path / "test.csv"
        
        # Save to temporary files with the correct separator
        train_data.to_csv(train_path, index=False, sep=';')
        test_data.to_csv(test_path, index=False)
        
        # Patch the config module
        monkeypatch.setattr('pipeline.data_processor.DATA_PATH', tmp_path)
        monkeypatch.setattr('pipeline.data_processor.SMS_FILE', "train.csv")
        monkeypatch.setattr('pipeline.data_processor.EMAIL_FILE', "test.csv")
        
        # Create processor after patching
        processor = DataProcessor()
        
        return processor, train_data, test_data

    def test_load_data_success(self, processor_with_data):
        """Test that data loading works correctly."""
        processor, expected_train, expected_test = processor_with_data
        
        # Test loading
        train_df, test_df = processor.load_data()
        
        # Assertions
        assert isinstance(train_df, pd.DataFrame), "Should return DataFrame for training data"
        assert isinstance(test_df, pd.DataFrame), "Should return DataFrame for test data"
        assert len(train_df) > 0, "Training data should not be empty"
        assert len(test_df) > 0, "Test data should not be empty"
        assert train_df.shape[0] == expected_train.shape[0], "Training data row count should match"
        assert MESSAGE in train_df.columns, f"Message column '{MESSAGE}' should be in training data"
        assert LABEL in train_df.columns, f"Label column '{LABEL}' should be in training data"

    def test_load_data_stores_internally(self, processor_with_data):
        """Test that loaded data is stored in processor attributes."""
        processor, _, _ = processor_with_data
        
        # Initially should be None
        assert processor.train_data is None
        assert processor.test_data is None
        
        # After loading
        processor.load_data()
        assert processor.train_data is not None, "train_data should be stored"
        assert processor.test_data is not None, "test_data should be stored"

    def test_drop_duplicates_functionality(self, data_with_duplicates):
        """Test duplicate removal."""
        processor = DataProcessor()
        
        original_count = len(data_with_duplicates)
        
        # Apply drop_duplicates
        cleaned_data = processor.drop_duplicates(data_with_duplicates.copy())
        
        # Assertions
        assert len(cleaned_data) < original_count, "Should remove duplicate rows"
        # Should keep only unique message-label combinations
        assert len(cleaned_data) == 3, "Should have 3 unique rows"

    def test_extract_msgs_labels_functionality(self, sample_spam_data):
        """Test message and label extraction."""
        processor = DataProcessor()
        
        messages, labels = processor.extract_msgs_labels(sample_spam_data)
        
        # Assertions
        assert isinstance(messages, pd.Series), "Messages should be a pandas Series"
        assert isinstance(labels, pd.Series), "Labels should be a pandas Series"
        assert len(messages) == len(sample_spam_data), "Message count should match dataset size"
        assert len(labels) == len(sample_spam_data), "Label count should match dataset size"
        assert messages.name == MESSAGE, f"Messages series should be named '{MESSAGE}'"
        assert labels.name == LABEL, f"Labels series should be named '{LABEL}'"

    def test_train_eval_same_data_split(self, sample_spam_data):
        """Test train-test split on same dataset."""
        processor = DataProcessor()
        
        messages, labels = processor.extract_msgs_labels(sample_spam_data)
        
        # Apply split
        msg_train, msg_test, lbl_train, lbl_test = processor.train_eval_same_data(
            messages, labels
        )
        
        # Assertions
        assert isinstance(msg_train, pd.Series), "Training messages should be Series"
        assert isinstance(msg_test, pd.Series), "Test messages should be Series"
        assert isinstance(lbl_train, pd.Series), "Training labels should be Series"
        assert isinstance(lbl_test, pd.Series), "Test labels should be Series"
        
        # Check split proportions
        total_size = len(messages)
        expected_test_size = int(total_size * TRAIN_TEST_SPLIT_SIZE)
        assert len(msg_test) == expected_test_size, "Test set size should match TRAIN_TEST_SPLIT_SIZE"
        assert len(msg_train) == total_size - expected_test_size, "Train set should have remaining data"
        
        # Check that train and test don't overlap
        assert len(msg_train) + len(msg_test) == total_size, "No data should be lost"

    def test_train_eval_same_data_stratification(self, sample_spam_data):
        """Test that stratification maintains class distribution."""
        processor = DataProcessor()
        
        messages, labels = processor.extract_msgs_labels(sample_spam_data)
        
        # Original distribution
        original_spam_ratio = (labels == 1).sum() / len(labels)
        
        # Apply split
        msg_train, msg_test, lbl_train, lbl_test = processor.train_eval_same_data(
            messages, labels
        )
        
        # Check stratification
        train_spam_ratio = (lbl_train == 1).sum() / len(lbl_train)
        test_spam_ratio = (lbl_test == 1).sum() / len(lbl_test)
        
        # Ratios should be similar (within 5% tolerance)
        assert abs(train_spam_ratio - original_spam_ratio) < 0.05, "Train set should maintain class distribution"
        assert abs(test_spam_ratio - original_spam_ratio) < 0.05, "Test set should maintain class distribution"

    def test_train_data1_eval_data2_cross_domain(self, sample_spam_data):
        """Test cross-domain training setup."""
        processor = DataProcessor()
        
        # Create two separate datasets
        data1 = sample_spam_data.iloc[:50].copy()
        data2 = sample_spam_data.iloc[50:].copy()
        
        data1_msgs, data1_lbls = processor.extract_msgs_labels(data1)
        data2_msgs, data2_lbls = processor.extract_msgs_labels(data2)
        
        # Apply cross-domain setup
        train_msgs, train_lbls, test_msgs, test_lbls = processor.train_data1_eval_data2(
            data1_msgs, data1_lbls, data2_msgs, data2_lbls
        )
        
        # Assertions
        assert len(train_msgs) == len(data1_msgs), "Training should use all of data1"
        assert len(test_msgs) == len(data2_msgs), "Testing should use all of data2"
        assert len(train_lbls) == len(data1_lbls), "Training labels should match data1"
        assert len(test_lbls) == len(data2_lbls), "Testing labels should match data2"

    def test_train_eval_combined_merging(self, sample_spam_data):
        """Test dataset combination and splitting."""
        processor = DataProcessor()
        
        # Create two separate datasets
        data1 = sample_spam_data.iloc[:40].copy()
        data2 = sample_spam_data.iloc[60:].copy()
        
        # Apply combined approach
        train_msgs, test_msgs, train_lbls, test_lbls = processor.train_eval_combined(
            data1, data2
        )
        
        # Assertions
        total_size = len(data1) + len(data2)
        assert len(train_msgs) + len(test_msgs) == total_size, "Should combine both datasets"
        assert isinstance(train_msgs, pd.Series), "Should return Series for messages"
        assert isinstance(train_lbls, pd.Series), "Should return Series for labels"
        
        # Check split proportions on combined data
        expected_test_size = int(total_size * TRAIN_TEST_SPLIT_SIZE)
        assert abs(len(test_msgs) - expected_test_size) <= 1, "Test size should match TRAIN_TEST_SPLIT_SIZE"

    def test_balance_minority_class_oversampling(self, imbalanced_data):
        """Test that balancing increases minority class to match majority."""
        processor = DataProcessor()
        
        messages, labels = processor.extract_msgs_labels(imbalanced_data)
        
        # Check initial imbalance
        initial_spam_count = (labels == 1).sum()
        initial_ham_count = (labels == 0).sum()
        assert initial_spam_count < initial_ham_count, "Dataset should be imbalanced"
        
        # Apply balancing
        balanced_msgs, balanced_lbls = processor.balance(messages, labels)
        
        # Check that classes are now balanced
        final_spam_count = (balanced_lbls == 1).sum()
        final_ham_count = (balanced_lbls == 0).sum()
        
        assert final_spam_count == final_ham_count, "Classes should be balanced"
        assert final_spam_count >= initial_spam_count, "Minority class should not decrease"
        assert final_ham_count == initial_ham_count, "Majority class should remain unchanged"

    def test_balance_preserves_data_structure(self, imbalanced_data):
        """Test that balancing preserves data types and structure."""
        processor = DataProcessor()
        
        messages, labels = processor.extract_msgs_labels(imbalanced_data)
        
        # Apply balancing
        balanced_msgs, balanced_lbls = processor.balance(messages, labels)
        
        # Assertions
        assert isinstance(balanced_msgs, pd.Series), "Messages should remain a Series"
        assert isinstance(balanced_lbls, pd.Series), "Labels should remain a Series"
        assert balanced_msgs.name == MESSAGE, f"Messages should be named '{MESSAGE}'"
        assert balanced_lbls.name == LABEL, f"Labels should be named '{LABEL}'"
        assert len(balanced_msgs) == len(balanced_lbls), "Messages and labels should have same length"

    def test_balance_already_balanced_data(self, sample_spam_data):
        """Test balancing on already balanced data (should not change much)."""
        processor = DataProcessor()
        
        messages, labels = processor.extract_msgs_labels(sample_spam_data)
        
        initial_size = len(messages)
        
        # Apply balancing
        balanced_msgs, balanced_lbls = processor.balance(messages, labels)
        
        # Should have minimal change if already balanced
        final_size = len(balanced_msgs)
        assert final_size >= initial_size, "Size should not decrease"
        
        # Classes should be equal
        spam_count = (balanced_lbls == 1).sum()
        ham_count = (balanced_lbls == 0).sum()
        assert spam_count == ham_count, "Should achieve perfect balance"

    def test_balance_uses_random_state(self, imbalanced_data):
        """Test that balancing is reproducible with RANDOM_STATE."""
        processor = DataProcessor()
        
        messages, labels = processor.extract_msgs_labels(imbalanced_data)
        
        # Apply balancing twice
        balanced_msgs1, balanced_lbls1 = processor.balance(messages.copy(), labels.copy())
        balanced_msgs2, balanced_lbls2 = processor.balance(messages.copy(), labels.copy())
        
        # Results should be identical due to random_state
        assert len(balanced_msgs1) == len(balanced_msgs2), "Sizes should be identical"
        assert balanced_msgs1.equals(balanced_msgs2), "Messages should be identical"
        assert balanced_lbls1.equals(balanced_lbls2), "Labels should be identical"

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame."""
        processor = DataProcessor()
        
        empty_df = pd.DataFrame({MESSAGE: [], LABEL: []})
        
        messages, labels = processor.extract_msgs_labels(empty_df)
        
        assert len(messages) == 0, "Should handle empty DataFrame"
        assert len(labels) == 0, "Should handle empty DataFrame"

    def test_single_class_data(self):
        """Test handling of data with only one class."""
        processor = DataProcessor()
        
        # Only ham messages
        single_class_df = pd.DataFrame({
            MESSAGE: ["Normal message"] * 10,
            LABEL: [0] * 10
        })
        
        messages, labels = processor.extract_msgs_labels(single_class_df)
        
        # Should not crash on balancing (though result may be trivial)
        try:
            balanced_msgs, balanced_lbls = processor.balance(messages, labels)
            # If it runs, check basic properties
            assert len(balanced_msgs) >= len(messages), "Should handle single class"
        except (KeyError, ValueError):
            # It's acceptable to raise an error for single-class data
            pass

    def test_data_integrity_after_operations(self, sample_spam_data):
        """Test that data integrity is maintained through multiple operations."""
        processor = DataProcessor()
        
        # Extract messages and labels
        messages, labels = processor.extract_msgs_labels(sample_spam_data)
        
        # Split data
        msg_train, msg_test, lbl_train, lbl_test = processor.train_eval_same_data(
            messages, labels
        )
        
        # Balance training data
        balanced_msgs, balanced_lbls = processor.balance(msg_train, lbl_train)
        
        # Check data integrity
        assert len(balanced_msgs) == len(balanced_lbls), "Messages and labels should match"
        assert balanced_msgs.isnull().sum() == 0, "Should not introduce null values"
        assert balanced_lbls.isnull().sum() == 0, "Should not introduce null values"
        assert set(balanced_lbls.unique()).issubset({0, 1}), "Labels should only be 0 or 1"


def run_dataprocessor_tests():
    """
    Function to run all DataProcessor tests programmatically.
    Useful for quick validation during development.
    """
    import pytest
    
    # Run tests with verbose output
    result = pytest.main([__file__, "-v", "--tb=short"])
    
    if result == 0:
        print("✅ All DataProcessor tests passed!")
        return True
    else:
        print("❌ Some DataProcessor tests failed!")
        return False


if __name__ == "__main__":
    # Allow running tests directly
    run_dataprocessor_tests()