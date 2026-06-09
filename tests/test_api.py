import requests
import pandas as pd
import xml.etree.ElementTree as ET
import time

API_KEY = "4eb0658c70fc463a89da"
SERVICE_ID = "I2710"
DATA_TYPE = "xml"

BASE_URL = "http://openapi.foodsafetykorea.go.kr/api"
PAGE_SIZE = 1000
OUTPUT_FILE = "건강기능식품_품목분류정보_I2710.xlsx"


def fetch_xml(start_idx, end_idx):
    url = f"{BASE_URL}/{API_KEY}/{SERVICE_ID}/{DATA_TYPE}/{start_idx}/{end_idx}"

    print(f"\n요청 URL: {url}")

    try:
        response = requests.get(url, timeout=30)

        print("HTTP 상태코드:", response.status_code)
        print("응답 앞부분:")
        print(response.text[:500])

        response.raise_for_status()

        return response.text

    except requests.exceptions.RequestException as e:
        print(f"[요청 오류] {start_idx}~{end_idx}: {e}")
        return None


def get_text(element, tag_name):
    target = element.find(tag_name)
    if target is not None and target.text is not None:
        return target.text.strip()
    return ""


def get_total_count():
    xml_text = fetch_xml(1, 1)

    if not xml_text:
        return 0

    try:
        root = ET.fromstring(xml_text)
        total_count = root.find(".//total_count")

        if total_count is None:
            print("total_count를 찾을 수 없습니다.")
            print(xml_text[:1000])
            return 0

        return int(total_count.text)

    except Exception as e:
        print("XML 파싱 오류:", e)
        return 0


def parse_rows(xml_text):
    rows = []

    root = ET.fromstring(xml_text)

    for row in root.findall(".//row"):
        item = {
            "품목명": get_text(row, "PRDCT_NM"),
            "섭취시주의사항": get_text(row, "IFTKN_ATNT_MATR_CN"),
            "주된기능성": get_text(row, "PRIMARY_FNCLTY"),
            "일일섭취량 하한": get_text(row, "DAY_INTK_LOWLIMIT"),
            "일일섭취량 상한": get_text(row, "DAY_INTK_HIGHLIMIT"),
            "단위": get_text(row, "INTK_UNIT"),
            "REMARK": get_text(row, "INTK_MEMO"),
            "성분명": get_text(row, "SKLL_IX_IRDNT_RAWMTRL"),
            "최초등록일": get_text(row, "CRET_DTM"),
            "최종수정일": get_text(row, "LAST_UPDT_DTM"),
        }

        rows.append(item)

    return rows


def collect_all_data():
    total_count = get_total_count()

    if total_count == 0:
        print("가져올 데이터가 없습니다.")
        return []

    print(f"\n전체 데이터 수: {total_count}건")

    all_rows = []

    for start_idx in range(1, total_count + 1, PAGE_SIZE):
        end_idx = min(start_idx + PAGE_SIZE - 1, total_count)

        print(f"\n수집 중: {start_idx} ~ {end_idx}")

        xml_text = fetch_xml(start_idx, end_idx)

        if not xml_text:
            continue

        try:
            rows = parse_rows(xml_text)
            all_rows.extend(rows)
            print(f"누적 수집: {len(all_rows)}건")

        except Exception as e:
            print(f"XML row 파싱 오류: {e}")

        time.sleep(0.2)

    return all_rows


def save_to_excel(rows):
    if not rows:
        print("저장할 데이터가 없습니다.")
        return

    df = pd.DataFrame(rows)

    ordered_columns = [
        "품목명",
        "성분명",
        "주된기능성",
        "섭취시주의사항",
        "일일섭취량 하한",
        "일일섭취량 상한",
        "단위",
        "REMARK",
        "최초등록일",
        "최종수정일",
    ]

    df = df[ordered_columns]
    df.to_excel(OUTPUT_FILE, index=False)

    print(f"\n엑셀 저장 완료: {OUTPUT_FILE}")
    print(f"총 저장 건수: {len(df)}건")


if __name__ == "__main__":
    rows = collect_all_data()
    save_to_excel(rows)