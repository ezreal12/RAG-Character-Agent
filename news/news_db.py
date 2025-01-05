import os
from tinydb import TinyDB, Query


def insert_unique_return_new(db_path, data_list):
    """
    Inserts data into the TinyDB only if the title is unique.

    Args:
        db: TinyDB instance
        data_list: List of dictionaries to be inserted

    Returns:
        List of dictionaries that were successfully inserted.
    """
    db = initialize_db(db_path)
    new_inserts = []
    Title = Query()

    for data in data_list:
        if not db.contains(Title.title == data['title']):
            db.insert(data)
            new_inserts.append(data)

    return new_inserts

def insert_unique(db_path, data_list):
    """
    Inserts data into the TinyDB only if the title is unique.

    Args:
        db: TinyDB instance
        data_list: List of dictionaries to be inserted

    Returns:
        List of dictionaries that were not inserted due to duplicate titles.
    """
    db = initialize_db(db_path)
    failed_inserts = []
    Title = Query()

    for data in data_list:
        if not db.contains(Title.title == data['title']):
            db.insert(data)
        else:
            failed_inserts.append(data)
            
    print("Database Content:", db.all())
    db.close()
    return failed_inserts

def initialize_db(db_path):
    """
    Initializes the TinyDB database. Creates directories if they do not exist.

    Args:
        db_path: Absolute path to the database file

    Returns:
        TinyDB instance
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return TinyDB(db_path)

# Example usage
if __name__ == "__main__":
    # Specify the absolute path to the database
    db_path = "./news_db/test.json"

    # Example data to insert
    data_to_insert = [
        {'title': 'Unique Title 1', 'content': {'text': 'Content 1', 'images': []}},
        {'title': 'Unique Title 2', 'content': {'text': 'Content 2', 'images': []}},
        {'title': 'Unique Title 1', 'content': {'text': 'Duplicate Content', 'images': []}},
    ]

    # Call the function
    failed = insert_unique(db_path, data_to_insert)

    # Print results
    print("Failed Inserts:", failed)
