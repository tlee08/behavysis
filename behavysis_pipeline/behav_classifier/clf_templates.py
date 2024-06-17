"""
_summary_
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from sklearn.ensemble import RandomForestClassifier


from keras.layers import Dense, Dropout, Input
from keras.models import Model

if TYPE_CHECKING:
    from .behav_classifier import BehavClassifier


class ClfTemplates:
    """
    Model templates for `BehavClassifier.clf`.
    """

    @staticmethod
    def rf(model: BehavClassifier) -> RandomForestClassifier:
        """
        x features is (samples, features).
        y outcome is (samples, class).
        """
        # Creating Gradient Boosting Classifier
        # clf = GradientBoostingClassifier(
        #     n_estimators=200,
        #     learning_rate=0.1,
        #     # max_depth=3,
        #     random_state=0,
        #     verbose=1,
        # )
        clf = RandomForestClassifier(
            n_estimators=2000,
            max_depth=3,
            random_state=0,
            n_jobs=16,
            verbose=1,
        )
        # Returning classifier model
        return clf

    # @staticmethod
    # def cnn_1(model: BehavClassifier) -> Model:
    #     """
    #     x features is (samples, window, features).
    #     y outcome is (samples, class).
    #     """
    #     # Input layers
    #     # 546 is number of SimBA features
    #     inputs = Input(shape=(model.configs.window_frames * 2 + 1, 546))
    #     # Hidden layers
    #     x = Conv1D(32, 3, activation="relu")(inputs)
    #     x = MaxPooling1D(2)(x)
    #     x = Conv1D(64, 3, activation="relu")(x)
    #     x = MaxPooling1D(2)(x)
    #     x = Flatten()(x)
    #     x = Dense(64, activation="relu")(x)
    #     x = Dropout(0.5)(x)
    #     # Binary classification problem (probability output)
    #     outputs = Dense(1, activation="sigmoid")(x)
    #     # Create the model
    #     model.clf = Model(inputs=inputs, outputs=outputs)
    #     # Compile the model
    #     model.clf.compile(
    #         optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    #     )
    #     # Returning classifier model
    #     return model.clf

    @staticmethod
    def dnn_1(model: BehavClassifier) -> Model:
        """
        x features is (samples, features).
        y outcome is (samples, class).
        """
        # Input layers
        # 546 is number of SimBA features
        input_shape = (546,)
        inputs = Input(shape=input_shape)
        # Hidden layers
        # l = Dense(256, activation="relu")(inputs)  # 32, 64, 256
        # l = Dropout(0.5)(l)
        x = Dense(64, activation="relu")(inputs)  # 32, 64, 256
        x = Dropout(0.5)(x)
        # Binary classification problem (probability output)
        outputs = Dense(1, activation="sigmoid")(x)
        # Create the model
        clf = Model(inputs=inputs, outputs=outputs)
        # Compiling model
        clf.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        # Returning classifier model
        return clf

    @staticmethod
    def dnn_2(model: BehavClassifier) -> Model:
        """ """
        # FRAME DNN MODEL #2
        # Create the model
        inputs = Input(shape=(546,))
        l = Dense(32, activation="relu")(inputs)  # 32, 64, 256
        l = Dropout(0.5)(l)
        outputs = Dense(1, activation="sigmoid")(l)
        clf = Model(inputs=inputs, outputs=outputs)
        clf.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return clf

    @staticmethod
    def dnn_3(model: BehavClassifier) -> Model:
        """ """
        # TESTING FRAME DNN MODEL #3
        # Create the model
        inputs = Input(shape=(546,))
        l = Dense(256, activation="relu")(inputs)  # 32, 64, 256
        l = Dropout(0.5)(l)
        l = Dense(64, activation="relu")(inputs)  # 32, 64, 256
        l = Dropout(0.5)(l)
        outputs = Dense(1, activation="sigmoid")(l)
        clf = Model(inputs=inputs, outputs=outputs)
        clf.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return clf
