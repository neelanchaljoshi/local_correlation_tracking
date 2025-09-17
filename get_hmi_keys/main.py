from datetime import datetime
from process import process_year

if __name__ == "__main__":
    T = [datetime.now()]
    for yr in range(2019, 2020):  # Change range as needed
        process_year(yr)
        T.append(datetime.now())
        print(f"Total time for {yr}: {T[-1] - T[-2]}")
