import re
from typing import Optional, Callable
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import mlflow
from utils.config import TOKEN_REGEX, LOWERCASE, stop_words, NB_FEATURES
import numpy as np


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        token_pattern: str = TOKEN_REGEX,
        max_features: int = NB_FEATURES,
        lowercase: bool = LOWERCASE,
        stop_words: Optional[str] = stop_words,
        preprocessor: Optional[Callable] = None,
        ngram_range: tuple = (1, 1)
    ):
        self.token_pattern = token_pattern
        self.max_features = max_features
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.preprocessor = preprocessor
        self.ngram_range = ngram_range

        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            token_pattern=self.token_pattern,
            stop_words=self.stop_words,
            lowercase=self.lowercase,
            preprocessor=self.preprocessor,
            ngram_range=self.ngram_range
        )
        self._is_fitted = False

    @staticmethod
    def preprocess_numbers(message: str, replacement: str = "numéro") -> str:
        """Remplace tous les nombres par un token générique."""
        return re.sub(r'\d+', replacement, message)

    def fit(self, X, y=None):
        """Fit le vectorizer et log les paramètres dans MLflow."""
        self.vectorizer.fit(X)
        self._is_fitted = True

        # MLflow logging sécurisé
        if mlflow.active_run() is not None:
            self._mlflow_safe_log("max_features", self.max_features)
            self._mlflow_safe_log("lowercase", self.lowercase)
            self._mlflow_safe_log("token_pattern", self.token_pattern)
            self._mlflow_safe_log("stop_words", self.stop_words)
            self._mlflow_safe_log("ngram_range", self.ngram_range)
            self._mlflow_safe_log("vocabulary_size", len(self.vectorizer.vocabulary_), metric=True)

        return self

    def transform(self, X):
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform.")
        return self.vectorizer.transform(X).astype(np.float32)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.vectorizer.transform(X).astype(np.float32)

    def get_feature_names_out(self):
        """Retourne les noms des features (méthode scikit-learn >=1.0)."""
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before getting feature names.")
        return self.vectorizer.get_feature_names_out()

    def get_feature_names(self):
        """Compatibilité avec les tests unitaires existants."""
        return self.get_feature_names_out()

    def get_vocabulary_size(self):
        """Retourne la taille du vocabulaire."""
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before getting vocabulary size.")
        return len(self.vectorizer.vocabulary_)

    def is_fitted(self):
        """Indique si le vectorizer est déjà fit."""
        return self._is_fitted

    def _mlflow_safe_log(self, key, value, metric=False):
        """
        Log un paramètre ou une métrique dans MLflow seulement s'il n'existe pas déjà.
        Évite les erreurs `Changing param values is not allowed`.
        """
        from mlflow.tracking import MlflowClient

        run = mlflow.active_run()
        if run is None:
            return

        client = MlflowClient()
        run_id = run.info.run_id
        params = client.get_run(run_id).data.params

        try:
            if metric:
                mlflow.log_metric(key, value)
            elif key not in params:
                mlflow.log_param(key, value)
        except Exception as e:
            # Sécurité : ignore si le paramètre est déjà loggé
            pass
