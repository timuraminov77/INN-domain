import json
import argparse
from dotenv import load_dotenv
from pipeline import SiteFinderPipeline

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Поиск основного сайта организации по ИНН")
    parser.add_argument("--inn", type=str, required=True, help="ИНН организации (например, 7721581040)")
    args = parser.parse_args()

    pipeline = SiteFinderPipeline()
    result = pipeline.run(args.inn)

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()