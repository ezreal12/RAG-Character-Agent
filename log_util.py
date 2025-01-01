
import datetime
import os

def log(msg, save_to_file=True):
    """
    Logs a message with timestamp and optionally saves it to a file.
    
    Args:
        msg: The message to log
        save_to_file (bool): Whether to save the log to a file
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {str(msg)}"
    print(log_message)
    
    if save_to_file:
        log_dir = "./logs"
        log_file = os.path.join(log_dir, "log.txt")
        
        # Create directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Append message to file (creates file if it doesn't exist)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')