from datetime import datetime, timedelta
import holidays

us_holidays = holidays.US(years=range(2024, 2031))

def is_trading_day(dt: datetime) -> bool:
    if dt.weekday() >= 5:
        return False
    if dt.date() in us_holidays:
        return False
    return True

def next_trading_day(current_date: datetime = None) -> datetime:
    if current_date is None:
        current_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    if is_trading_day(current_date):
        return current_date
    next_day = current_date + timedelta(days=1)
    while not is_trading_day(next_day):
        next_day += timedelta(days=1)
    return next_day

def format_next_trading_day() -> str:
    nxt = next_trading_day()
    return nxt.strftime("%Y-%m-%d (%A)")
