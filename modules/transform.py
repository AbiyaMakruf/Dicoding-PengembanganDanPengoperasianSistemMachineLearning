
"""
Author: abiyamf
Date: 1/8/2023
This is the transfom.py module.
Usage:
- Preprocess input features into transformed features.
"""

# Import library
import tensorflow as tf
import tensorflow_transform as tft


# Daftar numerical fitur pada dataset
NUMERICAL_FEATURES = [
    "Age",
    "Gender",
    "EducationLevel",
    "ExperienceYears",
    "PreviousCompanies",
    "DistanceFromCompany",
    "InterviewScore",
    "SkillScore",
    "PersonalityScore",
    "RecruitmentStrategy",
]

# Label key
LABEL_KEY = "HiringDecision"

# Fungsi untuk mengubuah nama fitur yang sudah di transform


def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"


# Fungsi untuk melakukan preprocessing
def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features

    Args:
        inputs: map from feature keys to raw features.

    Return:
        outputs: map from feature keys to transformed features.
    """

    outputs = {}

    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
