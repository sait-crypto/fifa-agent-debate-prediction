"""百度体育分析数据提取工具

从百度体育比赛页面 HTML 中提取结构化分析数据，保存为 JSON 文件。
文件名按 赛次_主队vs客队_比赛日期_生成日期 命名，防止冲突。

用法：
    python tools/BaiduTiyuAbstract.py
    python tools/BaiduTiyuAbstract.py path/to/page_source.html

如未指定输入文件，默认读取 tools/page_source.html。
"""

import re
import json
import sys
from datetime import datetime
from pathlib import Path


def extract_analysis_data(html_path: str) -> tuple[dict, str]:
    """从 HTML 文件提取分析数据，返回 (analysis_data, output_filename)

    输出文件名格式：{赛次}_{主队}vs{客队}_{比赛日期}_{生成日期}.json
    例：532539_韩国vs捷克_20260604_20260607.json
    """
    html_path = Path(html_path)
    if not html_path.exists():
        print(f"文件不存在：{html_path}")
        sys.exit(1)

    with open(html_path, encoding="utf-8") as f:
        html = f.read()

    # 提取 s-data 注释中的 JSON
    match = re.search(r"<!--s-data:({.*})-->", html, re.DOTALL)
    if not match:
        print("未找到 s-data 数据")
        sys.exit(1)

    data = json.loads(match.group(1))
    analysis_data = data["data"]["tabsList"][0]["data"]

    # ── 构建文件名 ──
    # 赛次编号
    match_number = analysis_data.get("result", {}).get("num", "unknown")

    # 主客队名
    igence = analysis_data.get("igence", [])
    home_team = away_team = "unknown"
    if igence:
        first_intel = igence[0].get("intelligence", {})
        home_team = first_intel.get("intelligenceTeamInfo", {}).get("name", "unknown")
        away_team = first_intel.get("intelligenceteamLeaterInfo", {}).get("name", "unknown")

    # 比赛日期：取主队近期最近一场比赛日期
    match_date = "nodate"
    home_records = analysis_data.get("homeRecord", [])
    if len(home_records) > 1:
        hist_list = home_records[1].get("history", {}).get("list", [])
        if hist_list:
            raw_date = hist_list[0].get("date", "")
            match_date = raw_date.replace("-", "")

    # 生成日期
    gen_date = datetime.now().strftime("%Y%m%d")

    filename = f"{match_number}_{home_team}vs{away_team}_{match_date}_{gen_date}.json"

    return analysis_data, filename


def save_with_collision_protection(output_dir: Path, filename: str, data: dict) -> Path:
    """保存 JSON 文件，如文件已存在则追加后缀防止覆盖"""
    output_path = output_dir / filename
    counter = 2
    while output_path.exists():
        stem = filename.rsplit(".", 1)[0]
        new_filename = f"{stem}_{counter}.json"
        output_path = output_dir / new_filename
        counter += 1

    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(data, out, ensure_ascii=False, indent=2)

    return output_path


def main() -> None:
    # 输入文件：命令行参数或默认
    html_path = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "page_source.html"

    analysis_data, filename = extract_analysis_data(str(html_path))
    output_dir = Path(__file__).parent
    output_path = save_with_collision_protection(output_dir, filename, analysis_data)

    print(f"提取完成：{output_path.name}")


if __name__ == "__main__":
    main()
