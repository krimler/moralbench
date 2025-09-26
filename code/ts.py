from datetime import datetime

def ts():
    current_datetime = datetime.now()
    timestamp_string = current_datetime.strftime("%d%H%M%S")
    print(timestamp_string)
    return timestamp_string

ts()
