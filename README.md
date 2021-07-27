# Spam_Classification
Determines a SMS/mail/sentence occuring in natural language to be a spam or not

1. Install the required dependencies from requirements.txt
    ```python
    pip3 install -r requirements.txt

2. Download the data from Kaggle and save it in data/
    ```bash
    kaggle datasets download -d uciml/sms-spam-collection-dataset
    ```

3. To train the model, run the script as follows:
    ```python
    python3 spam_detection.py --text "URGENT! You have won the prize of a million dollars" --train True --data_path data/spam.csv --model_type MultinomialNB
    ```

4. To only test, run the script as follows:
    ```python
    python3 spam_detection.py --model_path models/MultinomialNB.model --text "URGENT! You have won the prize of a million dollars"
    ```