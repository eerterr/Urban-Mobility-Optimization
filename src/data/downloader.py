from pathlib import Path # чтобы не клеить пути через //
from zipfile import ZipFile # раскрытие зип архивов
from io import BytesIO # работа с зипами прямо из памяти без отдельного сохранения

import requests # обработка HTTP запросов

# параметры загрузки
BASE_URL = "https://divvy-tripdata.s3.amazonaws.com"
RAW_DIR = Path("data/raw/divvy")
RAW_DIR.mkdir(parents=True, exist_ok=True)

YEAR = 2024 # год загрузки данных

#функция скачивания месяца
def download_month(month: int) -> None:

    filename = f"{YEAR}{month:02d}-divvy-tripdata.zip" # формирование имени одного файла
    url = f"{BASE_URL}/{filename}"

    print(f"Downloading {filename}...")

    # HHTP-запрос
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    with ZipFile(BytesIO(response.content)) as archive:
        archive.extractall(RAW_DIR)

    print(f"Saved to {RAW_DIR}")

def main():
    for month in range (1, 13):
        download_month(month)

    print("\nAll files downloaded successfully!")

if __name__ == "__main__": main()