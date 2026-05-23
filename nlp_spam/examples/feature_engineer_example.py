"""
Example usage of the FeatureEngineer class for spam detection.

This script demonstrates how to use the FeatureEngineer to preprocess
text data and extract features for machine learning models.
"""

from src.pipeline.feature_engineer import FeatureEngineer


def main():
    """Demonstrate FeatureEngineer usage with sample spam/ham messages."""
    
    # Sample messages (spam and ham)
    messages = [
        "Congratulations! You've won $1000. Call 555-1234 now!",
        "Hey, want to grab coffee tomorrow afternoon?",
        "URGENT: Your account will be closed. Click here immediately!",
        "Meeting rescheduled to 3pm on Friday",
        "FREE prize money!!! Text WIN to 12345",
        "Thanks for your email. I'll respond tomorrow.",
    ]
    
    labels = [1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = ham
    
    print("=" * 70)
    print("FEATURE ENGINEER EXAMPLE")
    print("=" * 70)
    
    # Example 1: Basic usage with default settings
    print("\n1. Basic Usage (Default Settings)")
    print("-" * 70)
    
    fe_basic = FeatureEngineer()
    features_basic = fe_basic.fit_transform(messages)
    
    print(f"Number of messages: {features_basic.shape[0]}")
    print(f"Number of features: {features_basic.shape[1]}")
    print(f"First 10 features: {fe_basic.get_feature_names()[:10]}")
    
    # Example 2: With number preprocessing
    print("\n2. With Number Preprocessing")
    print("-" * 70)
    
    fe_numbers = FeatureEngineer(
        preprocessor=FeatureEngineer.preprocess_numbers,
        max_features=50
    )
    features_numbers = fe_numbers.fit_transform(messages)
    
    print(f"Number of features: {features_numbers.shape[1]}")
    feature_names = fe_numbers.get_feature_names()
    print(f"'numéro' in features: {'numéro' in feature_names}")
    print(f"Sample features: {feature_names[:15]}")
    
    # Example 3: Custom tokenization pattern
    print("\n3. Custom Tokenization (All Words)")
    print("-" * 70)
    
    fe_custom = FeatureEngineer(
        token_pattern=r"(\w+)",  # Match all word characters
        max_features=30,
        stop_words=None,  # Keep all words including stop words
        lowercase=True
    )
    features_custom = fe_custom.fit_transform(messages)
    
    print(f"Number of features: {features_custom.shape[1]}")
    print(f"Features (all lowercase): {fe_custom.get_feature_names()[:20]}")
    
    # Example 4: Real-world scenario - train/test split
    print("\n4. Train/Test Split Scenario")
    print("-" * 70)
    
    # Split into train/test
    train_messages = messages[:4]
    test_messages = messages[4:]
    
    fe_train = FeatureEngineer(
        preprocessor=FeatureEngineer.preprocess_numbers,
        max_features=100
    )
    
    # Fit on training data
    train_features = fe_train.fit_transform(train_messages)
    print(f"Training features shape: {train_features.shape}")
    print(f"Vocabulary size: {fe_train.get_vocabulary_size()}")
    
    # Transform test data using same vocabulary
    test_features = fe_train.transform(test_messages)
    print(f"Test features shape: {test_features.shape}")
    print(f"Same number of features: {train_features.shape[1] == test_features.shape[1]}")
    
    # Example 5: Feature inspection
    print("\n5. Feature Inspection")
    print("-" * 70)
    
    fe_inspect = FeatureEngineer(
        preprocessor=FeatureEngineer.preprocess_numbers,
        max_features=20,
        token_pattern=r"([A-Za-z]{3,}|!+)"  # Words 3+ letters or exclamations
    )
    features_inspect = fe_inspect.fit_transform(messages)
    
    # Show feature matrix for first message
    feature_names = fe_inspect.get_feature_names()
    first_msg_features = features_inspect[0].toarray()[0]
    
    print(f"Message: '{messages[0]}'")
    print(f"\nExtracted features (count > 0):")
    for i, count in enumerate(first_msg_features):
        if count > 0:
            print(f"  '{feature_names[i]}': {count}")
    
    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
