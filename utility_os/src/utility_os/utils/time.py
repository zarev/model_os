from datetime import datetime

def convert_millis(m) -> str:
    """ Convert milliseconds into hh:mm:ss format"""
    seconds=(m // 1000) % 60
    seconds=seconds if seconds > 9 else f"0{seconds}"

    minutes=(m // (1000*60)) % 60
    minutes=minutes if minutes > 9 else f"0{minutes}"

    hours=(m // (1000*60*60)) % 24
    hours=hours if hours > 9 else f"0{hours}"

    return f"{hours}:{minutes}:{seconds}"

def curr_time():
    return datetime.now().ctime()

def to_ISO(t):
    return datetime.fromisoformat(t)