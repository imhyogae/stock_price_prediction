from src.long_short_term_memory import LongShortTermMemory
from datetime import datetime
from dateutil.relativedelta import relativedelta

if __name__ == "__main__":
    start_date = (datetime.today() - relativedelta(years=3)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    LongShortTermMemory('TSLA', start_date, end_date, 60).run_prediction(False)
