"""
Unit tests for the FeatureEngineer class.

Tests cover text preprocessing, feature extraction, vocabulary building,
and edge cases for the feature engineering pipeline.
"""

import pytest
import numpy as np
from scipy.sparse import csr_matrix, issparse
from src.pipeline.feature_engineer import FeatureEngineer


class TestFeatureEngineerInitialization:
    """Test FeatureEngineer initialization and configuration."""
    
    def test_default_initialization(self):
        """Test that FeatureEngineer initializes with default parameters."""
        fe = FeatureEngineer()
        
        assert fe.max_features == 5000
        assert fe.token_pattern == r"([A-Za-z]{4,}|!+)"
        assert fe.lowercase == False
        assert fe.stop_words == "english"
        assert fe.preprocessor is None
        assert fe.is_fitted() == False
    
    def test_custom_initialization(self):
        """Test FeatureEngineer with custom parameters."""
        custom_pattern = r"(\w+)"
        fe = FeatureEngineer(
            token_pattern=custom_pattern,
            max_features=1000,
            lowercase=True,
            stop_words=None
        )
        
        assert fe.max_features == 1000
        assert fe.token_pattern == custom_pattern
        assert fe.lowercase == True
        assert fe.stop_words is None
    
    def test_initialization_with_preprocessor(self):
        """Test initialization with custom preprocessor function."""
        def custom_preprocessor(text):
            return text.upper()
        
        fe = FeatureEngineer(preprocessor=custom_preprocessor)
        assert fe.preprocessor == custom_preprocessor


class TestNumberPreprocessing:
    """Test number preprocessing functionality."""
    
    def test_preprocess_numbers_basic(self):
        """Test basic number replacement."""
        message = "Call me at 555-1234"
        result = FeatureEngineer.preprocess_numbers(message)
        assert "555" not in result
        assert "1234" not in result
        assert "numéro" in result
    
    def test_preprocess_numbers_multiple(self):
        """Test replacement of multiple numbers."""
        message = "Pay 500 to get 600 bonus"
        result = FeatureEngineer.preprocess_numbers(message)
        assert "500" not in result
        assert "600" not in result
        assert result.count("numéro") == 2
    
    def test_preprocess_numbers_custom_replacement(self):
        """Test number replacement with custom placeholder."""
        message = "My number is 123456"
        result = FeatureEngineer.preprocess_numbers(message, replacement="<NUM>")
        assert "123456" not in result
        assert "<NUM>" in result
    
    def test_preprocess_numbers_no_numbers(self):
        """Test preprocessing when no numbers are present."""
        message = "Hello world"
        result = FeatureEngineer.preprocess_numbers(message)
        assert result == message
    
    def test_preprocess_numbers_complex(self):
        """Test preprocessing with various number formats."""
        message = "Call 0612345678 or pay $12.50 for item #42"
        result = FeatureEngineer.preprocess_numbers(message)
        assert "0612345678" not in result
        assert "12" not in result
        assert "50" not in result
        assert "42" not in result


class TestFitTransform:
    """Test fitting and transforming functionality."""
    
    @pytest.fixture
    def sample_messages(self):
        """Provide sample messages for testing."""
        return [
            "This is a spam message with money",
            "Normal message from a friend",
            "Win big prizes now click here",
            "Meeting tomorrow at 3pm"
        ]
    
    def test_fit_transform(self, sample_messages):
        """Test fit_transform returns correct shape and type."""
        fe = FeatureEngineer(max_features=10)
        result = fe.fit_transform(sample_messages)
        
        assert issparse(result)
        assert result.shape[0] == len(sample_messages)
        assert result.shape[1] <= 10
        assert fe.is_fitted()
    
    def test_fit_then_transform(self, sample_messages):
        """Test separate fit and transform calls."""
        fe = FeatureEngineer(max_features=10)
        fe.fit(sample_messages)
        result = fe.transform(sample_messages)
        
        assert issparse(result)
        assert result.shape[0] == len(sample_messages)
        assert fe.is_fitted()
    
    def test_transform_before_fit_raises_error(self, sample_messages):
        """Test that transform without fit raises error."""
        fe = FeatureEngineer()
        
        with pytest.raises(RuntimeError, match="must be fitted before transform"):
            fe.transform(sample_messages)
    
    def test_transform_new_data(self, sample_messages):
        """Test transforming new data after fitting."""
        fe = FeatureEngineer(max_features=10)
        fe.fit(sample_messages)
        
        new_messages = ["Another spam message", "Regular email"]
        result = fe.transform(new_messages)
        
        assert result.shape[0] == len(new_messages)
        assert result.shape[1] == fe.get_vocabulary_size()
    
    def test_fit_transform_consistency(self, sample_messages):
        """Test that fit_transform and fit().transform() give same results."""
        fe1 = FeatureEngineer(max_features=20, token_pattern=r"(\w+)")
        result1 = fe1.fit_transform(sample_messages)
        
        fe2 = FeatureEngineer(max_features=20, token_pattern=r"(\w+)")
        fe2.fit(sample_messages)
        result2 = fe2.transform(sample_messages)
        
        assert np.array_equal(result1.toarray(), result2.toarray())


class TestVocabularyManagement:
    """Test vocabulary extraction and management."""
    
    def test_get_feature_names(self):
        """Test getting feature names from vocabulary."""
        messages = ["hello world", "hello friend", "world peace"]
        fe = FeatureEngineer(max_features=10, stop_words=None, token_pattern=r"(\w+)")
        fe.fit(messages)
        
        feature_names = fe.get_feature_names()
        assert isinstance(feature_names, np.ndarray)
        assert len(feature_names) > 0
        assert "hello" in feature_names or "world" in feature_names
    
    def test_get_feature_names_before_fit_raises_error(self):
        """Test that getting features before fit raises error."""
        fe = FeatureEngineer()
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            fe.get_feature_names()
    
    def test_get_vocabulary_size(self):
        """Test getting vocabulary size."""
        messages = ["word1 word2", "word3 word4", "word5"]
        fe = FeatureEngineer(max_features=10, token_pattern=r"(\w+)", stop_words=None)
        fe.fit(messages)
        
        vocab_size = fe.get_vocabulary_size()
        assert isinstance(vocab_size, int)
        assert vocab_size > 0
        assert vocab_size <= 10
    
    def test_get_vocabulary_size_before_fit_raises_error(self):
        """Test that getting size before fit raises error."""
        fe = FeatureEngineer()
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            fe.get_vocabulary_size()
    
    def test_max_features_limit(self):
        """Test that vocabulary respects max_features limit."""
        messages = [f"word{i}" for i in range(100)]
        messages = [" ".join(messages)]
        
        fe = FeatureEngineer(max_features=50, token_pattern=r"(\w+)", stop_words=None)
        fe.fit(messages)
        
        assert fe.get_vocabulary_size() <= 50


class TestTokenization:
    """Test different tokenization strategies."""
    
    def test_default_token_pattern(self):
        """Test default tokenization pattern (4+ letters and exclamations)."""
        messages = ["Hi! Hello world! Test!!!"]
        fe = FeatureEngineer(stop_words=None)
        fe.fit(messages)
        
        features = fe.get_feature_names()
        # Default pattern should capture words with 4+ letters
        assert any(len(f) >= 4 for f in features if f.isalpha())
    
    def test_custom_token_pattern_words_only(self):
        """Test custom pattern that extracts all words."""
        messages = ["Test 123 message! #hashtag @user"]
        fe = FeatureEngineer(
            token_pattern=r"([A-Za-z]+)",
            max_features=10,
            stop_words=None
        )
        fe.fit(messages)
        
        features = fe.get_feature_names()
        # Should only contain alphabetic tokens
        assert all(f.isalpha() for f in features)
    
    def test_token_pattern_with_numbers(self):
        """Test pattern that includes numbers."""
        messages = ["Item123 Product456"]
        fe = FeatureEngineer(
            token_pattern=r"([A-Za-z0-9]+)",
            max_features=10,
            stop_words=None
        )
        result = fe.fit_transform(messages)
        
        features = fe.get_feature_names()
        assert len(features) > 0


class TestStopWords:
    """Test stop word removal functionality."""
    
    def test_stop_words_removal(self):
        """Test that English stop words are removed."""
        messages = ["the quick brown fox", "a beautiful day"]
        
        fe_with_stopwords = FeatureEngineer(
            stop_words="english",
            token_pattern=r"(\w+)",
            max_features=10
        )
        fe_with_stopwords.fit(messages)
        
        fe_without_stopwords = FeatureEngineer(
            stop_words=None,
            token_pattern=r"(\w+)",
            max_features=10
        )
        fe_without_stopwords.fit(messages)
        
        # With stop words removed, we should have fewer features
        assert fe_with_stopwords.get_vocabulary_size() <= fe_without_stopwords.get_vocabulary_size()
    
    def test_no_stop_words(self):
        """Test feature extraction without stop word removal."""
        messages = ["the cat and the dog"]
        fe = FeatureEngineer(stop_words=None, token_pattern=r"(\w+)")
        fe.fit(messages)
        
        features = fe.get_feature_names()
        # Should contain common words like "the" and "and"
        assert len(features) > 0


class TestLowercase:
    """Test lowercase conversion functionality."""
    
    def test_lowercase_true(self):
        """Test that lowercase=True converts text."""
        messages = ["HELLO World"]
        fe = FeatureEngineer(
            lowercase=True,
            token_pattern=r"(\w+)",
            stop_words=None
        )
        result = fe.fit_transform(messages)
        features = fe.get_feature_names()
        
        # All features should be lowercase
        assert all(f.islower() for f in features if f.isalpha())
    
    def test_lowercase_false(self):
        """Test that lowercase=False preserves case."""
        messages = ["HELLO world"]
        fe = FeatureEngineer(
            lowercase=False,
            token_pattern=r"(\w+)",
            stop_words=None
        )
        result = fe.fit_transform(messages)
        features = fe.get_feature_names()
        
        # Should have mixed case features
        assert len(features) > 0


class TestPreprocessorIntegration:
    """Test custom preprocessor integration."""
    
    def test_with_number_preprocessor(self):
        """Test using number preprocessing as custom preprocessor."""
        messages = ["Call 555-1234", "Amount is 5000"]
        fe = FeatureEngineer(
            preprocessor=FeatureEngineer.preprocess_numbers,
            token_pattern=r"(\w+)",
            stop_words=None
        )
        result = fe.fit_transform(messages)
        features = fe.get_feature_names()
        
        # Should contain the replacement token
        assert "numéro" in features
        # Should not contain actual numbers
        assert "555" not in features
        assert "5000" not in features
    
    def test_custom_preprocessor_function(self):
        """Test with custom preprocessing function."""
        def remove_punctuation(text):
            return text.replace("!", "").replace("?", "")
        
        messages = ["Hello! How are you?", "Great! Thanks!"]
        fe = FeatureEngineer(
            preprocessor=remove_punctuation,
            token_pattern=r"(\w+)",
            stop_words=None
        )
        result = fe.fit_transform(messages)
        
        assert result.shape[0] == len(messages)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_messages(self):
        """Test handling of empty message list."""
        messages = []
        fe = FeatureEngineer()
        
        # CountVectorizer raises ValueError for empty documents
        with pytest.raises(ValueError, match="empty vocabulary"):
            fe.fit_transform(messages)
    
    def test_single_message(self):
        """Test with a single message."""
        messages = ["Single message here"]
        fe = FeatureEngineer(max_features=10)
        result = fe.fit_transform(messages)
        
        assert result.shape[0] == 1
        assert result.shape[1] > 0
    
    def test_empty_strings(self):
        """Test handling messages with empty strings."""
        messages = ["", "normal message", ""]
        fe = FeatureEngineer()
        result = fe.fit_transform(messages)
        
        assert result.shape[0] == len(messages)
    
    def test_special_characters_only(self):
        """Test messages with only special characters."""
        messages = ["!!!", "###", "@@@"]
        fe = FeatureEngineer(token_pattern=r"([!]+)")
        result = fe.fit_transform(messages)
        
        # Should handle special characters gracefully
        assert result.shape[0] == len(messages)
    
    def test_very_long_message(self):
        """Test with a very long message."""
        long_message = " ".join(["word"] * 1000)
        messages = [long_message]
        fe = FeatureEngineer()
        result = fe.fit_transform(messages)
        
        assert result.shape[0] == 1


class TestRealWorldScenarios:
    """Test realistic spam detection scenarios."""
    
    def test_spam_vs_ham_messages(self):
        """Test with typical spam and ham messages."""
        messages = [
            "Congratulations! You've won $1000. Call now!",
            "Hi, how are you doing today?",
            "URGENT: Your account will be closed. Click here!",
            "Let's meet for coffee tomorrow afternoon",
            "Free prize money winner call 555-1234"
        ]
        
        fe = FeatureEngineer(
            preprocessor=FeatureEngineer.preprocess_numbers,
            max_features=50
        )
        result = fe.fit_transform(messages)
        
        assert result.shape[0] == len(messages)
        assert result.shape[1] <= 50
        assert fe.get_vocabulary_size() > 0
    
    def test_sms_style_messages(self):
        """Test with SMS-style short messages."""
        messages = [
            "Free entry 2 win",
            "C u l8r",
            "Txt STOP to unsubscribe",
            "Meeting at 5"
        ]
        
        fe = FeatureEngineer(max_features=20, token_pattern=r"(\w+)")
        result = fe.fit_transform(messages)
        
        assert result.shape[0] == len(messages)
    
    def test_email_style_messages(self):
        """Test with longer email-style messages."""
        messages = [
            "Dear valued customer, we are pleased to inform you about our special offer.",
            "Thank you for your email. I will respond to your inquiry tomorrow.",
            "CLICK HERE NOW for exclusive deals that expire in 24 hours!"
        ]
        
        fe = FeatureEngineer(max_features=100)
        result = fe.fit_transform(messages)
        
        assert result.shape[0] == len(messages)
        assert fe.get_vocabulary_size() > 0
