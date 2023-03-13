# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import os

# Define the base path to the input dataset and then
# use it to derive the path to the images directory
# and  the annotation CSV file
BASE_PATH = "Dataset-Tensorflow"
TRAIN_IMAGES_PATH = os.path.sep.join([BASE_PATH, "train"])
TEST_IMAGES_PATH = os.path.sep.join([BASE_PATH, "test"])
VALID_IMAGES_PATH = os.path.sep.join([BASE_PATH, "valid"])

TRAIN_CSV_PATH = os.path.sep.join([TRAIN_IMAGES_PATH, "_annotations.csv"])
TEST_CSV_PATH = os.path.sep.join([TEST_IMAGES_PATH, "_annotations.csv"])
VALID_CSV_PATH = os.path.sep.join([VALID_IMAGES_PATH, "_annotations.csv"])

# Define the path to the base output directory
BASE_OUTPUT = "output"
