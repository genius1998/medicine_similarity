from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


NS = {"hp": "http://www.hancom.co.kr/hwpml/2011/paragraph"}
ET.register_namespace("hp", NS["hp"])


PRODUCT_SECTION_START = 164


PARAGRAPH_UPDATES = {
    169: "바른성분연구소",
    171: "성분콕",
    173: (
        "성분콕은 건강기능식품의 성분 정보를 기반으로 제품 간 유사도를 계산하고 유사 제품을 추천하는 서비스이다. "
        "기존 DB 제품 추천과 이미지 업로드 기반 OCR 추천을 함께 제공하여, 소비자가 제품명보다 실제 기능성 원료 중심으로 제품을 비교하고 탐색할 수 있도록 지원한다."
    ),
    176: "웹",
    181: "",
    183: "",
    185: "",
    187: "http://127.0.0.1:8000",
    189: "FastAPI + Jinja2 기반 프로토타입, OCR 및 성분 유사도 추천 기능 구현",
    191: "◦ 건강기능식품 개별인정형 정보(식품의약품안전처)",
    192: "◦ 건강기능식품 품목분류정보(식품의약품안전처)",
    193: "◦ 식품의약품안전처_건강기능식품정보(식품의약품안전처)",
    199: (
        "◦ 건강기능식품 시장 확대와 함께 소비자는 유사한 효능을 표방하는 수많은 제품 중에서 자신에게 맞는 제품을 선택해야 하지만, "
        "실제 구매 단계에서는 제품명과 광고 문구에 의존하는 경우가 많아 성분 구성이 비슷한 제품을 객관적으로 비교하기 어렵다."
    ),
    200: (
        "- 성분콕은 이러한 문제를 해결하기 위해 공공데이터 기반 성분 체계와 OCR 기술을 결합하여, "
        "기존 제품은 물론 사용자가 촬영한 제품 이미지까지 성분 단위로 분석하고 유사한 건강기능식품을 추천하는 서비스를 기획하였다."
    ),
    201: "",
    202: "",
    203: "",
    204: "",
    205: "",
    208: (
        "◦ 성분콕은 건강기능식품의 핵심 원료 구성을 분석하여 유사 제품을 추천하는 웹 기반 서비스로, "
        "기존 DB 제품 기준 추천과 이미지 업로드 기반 OCR 추천을 모두 지원한다."
    ),
    209: (
        "- 사용자는 제품 등록번호나 제품 정보를 기준으로 유사 제품을 조회할 수 있고, 제품 라벨 이미지를 업로드하면 OCR로 추출한 원료명을 정규화한 뒤 "
        "가중치 자카드 유사도로 비슷한 제품을 추천받을 수 있다. 또한 OCR 결과를 사용자가 수정한 뒤 재추천받을 수 있어 정확도와 실사용성을 함께 확보하였다."
    ),
    210: "",
    211: "",
    212: "",
    213: "",
    215: (
        "◦ 건강기능식품 개별인정형 정보와 건강기능식품 품목분류정보는 기능성 원료 표준화와 카테고리 체계 구성에 활용하였고, "
        "식품의약품안전처_건강기능식품정보는 약 4만 5천여 건의 추천 대상 제품 DB 구축에 활용하였다."
    ),
    216: (
        "- 기술적으로는 공공데이터를 바탕으로 제품별 기능성 원료 벡터를 생성하고, 주원료·부원료·보조원료의 역할 차이를 반영한 가중치 자카드 유사도 계산을 적용하였다. "
        "이미지 업로드의 경우 OCR로 추출한 라벨 텍스트를 동일한 공공데이터 기반 성분 체계에 매핑하여 기존 DB와 같은 기준으로 추천이 가능하도록 구현하였다."
    ),
    217: "",
    218: "",
    219: "",
    220: "",
    223: (
        "◦ 기존 건강기능식품 탐색 서비스가 상품명 검색, 브랜드 중심 탐색, 리뷰 기반 추천에 집중하는 것과 달리, "
        "성분콕은 실제 기능성 성분 구성을 중심으로 제품 간 유사도를 계산한다는 점에서 차별성이 크다."
    ),
    224: (
        "- 특히 제품 이미지만으로도 OCR을 통해 성분을 추출하고 추천까지 연결할 수 있으며, 공공데이터 기반 성분 표준화와 추천 이유 설명 기능을 함께 제공한다. "
        "이는 소비자가 제품명을 몰라도 성분 중심으로 대체 상품을 찾을 수 있게 해주는 독창적인 사용자 경험이다."
    ),
    225: "",
    226: "",
    227: "",
    228: "",
    229: "",
    230: "",
    235: (
        "◦ 건강기능식품 시장은 제품 수가 많고 비교 탐색 수요가 높아 성분 기반 추천 서비스의 시장성이 충분하다. "
        "소비자는 현재 복용 중인 제품과 유사한 대체 상품을 찾고자 하며, 판매 플랫폼은 유사 제품 추천을 통해 탐색 편의성과 전환율을 높일 수 있다."
    ),
    236: (
        "- 성분콕은 공공데이터 기반 구조로 초기 구현 비용 대비 실현 가능성이 높고, 향후 B2C 비교 추천 서비스와 B2B 추천 API 제공으로 확장할 수 있다. "
        "또한 가격 비교, 목적별 추천, 중복 성분 안내 기능까지 확장 가능해 건강기능식품 탐색 플랫폼으로 성장할 잠재력이 크다."
    ),
    237: "",
    238: "",
    239: "",
    240: "",
    241: "",
    244: (
        "◦ 바른성분연구소는 성분콕을 소비자용 건강기능식품 비교·추천 서비스로 고도화한 뒤, 이커머스·약국·헬스케어 플랫폼과 제휴하는 방식으로 사업화를 추진하고자 한다."
    ),
    245: (
        "- 초기에는 웹 MVP를 중심으로 추천 정확도와 사용자 경험을 개선하고, 이후 추천 API 제공, 제휴 수수료, 프리미엄 성분 비교 기능 등으로 수익모델을 구체화할 계획이다. "
        "장기적으로는 가격 정보, 맞춤형 추천, 복용 주의 성분 안내까지 확장하여 성분 중심 건강기능식품 탐색 표준 서비스를 구축하는 것을 목표로 한다."
    ),
    246: "",
    247: "",
    248: "",
    249: "",
    250: "",
    251: "",
    252: "",
}


def set_paragraph_text(paragraph: ET.Element, text: str) -> None:
    text_nodes = paragraph.findall(".//hp:t", NS)
    if not text_nodes:
        return
    text_nodes[0].text = text
    for node in text_nodes[1:]:
        node.text = ""


def build_output_hwpx(source_hwpx: Path, output_hwpx: Path) -> None:
    work_dir = output_hwpx.with_suffix("")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    zip_copy = work_dir / "source.zip"
    shutil.copy2(source_hwpx, zip_copy)

    with ZipFile(zip_copy, "r") as archive:
        archive.extractall(work_dir / "unzipped")

    section_path = work_dir / "unzipped" / "Contents" / "section0.xml"
    tree = ET.parse(section_path)
    root = tree.getroot()
    paragraphs = root.findall(".//hp:p", NS)

    for idx, value in PARAGRAPH_UPDATES.items():
        if idx >= len(paragraphs):
            continue
        set_paragraph_text(paragraphs[idx], value)

    tree.write(section_path, encoding="utf-8", xml_declaration=True)

    if output_hwpx.exists():
        output_hwpx.unlink()
    with ZipFile(output_hwpx, "w", compression=ZIP_DEFLATED) as archive:
        for file_path in sorted((work_dir / "unzipped").rglob("*")):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(work_dir / "unzipped").as_posix())


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_hwpx = Path(r"C:\Users\com\Downloads\2026년 경진대회 기획서.hwpx")
    output_hwpx = repo_root / "output" / "성분콕_제품및서비스개발_기획서.hwpx"
    build_output_hwpx(source_hwpx, output_hwpx)
    print(output_hwpx)


if __name__ == "__main__":
    main()
