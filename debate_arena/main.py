"""CLI 入口 — 多智能体辩论预测系统"""

import argparse
import datetime
import re
from pathlib import Path

from openai import OpenAI

from .config import load_llm_config, load_system_config, list_available_presets
from .match_config import load_match_config, quick_match_config, merge_debate_config
from .predict import MatchPredictor
from .renderer import Renderer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="多智能体辩论预测系统",
    )

    parser.add_argument(
        "match", nargs="?", default=None,
        help='比赛对阵，如 "巴西 vs 阿根廷"（快速模式）',
    )
    parser.add_argument(
        "--config", default=None,
        help="比赛配置 YAML 文件路径（matches/ 目录下）",
    )
    parser.add_argument(
        "--preset", default=None,
        help="LLM 预设名（覆盖 system.yaml 中的 llm_preset）",
    )
    parser.add_argument(
        "--list-presets", action="store_true",
        help="列出 .env 中所有可用的 LLM 预设",
    )

    args = parser.parse_args()

    # 列出预设
    if args.list_presets:
        presets = list_available_presets()
        print("\n可用的 LLM 预设（在 .env 中定义）：")
        for p in presets:
            print(f"  - {p}")
        print("\n使用方法：在 system.yaml 中设置 llm_preset，或使用 --preset 参数")
        return

    # 加载 LLM 配置
    llm = load_llm_config(preset=args.preset)

    # 加载系统配置（辩论参数的默认值来源）
    sys_config = load_system_config()

    # 创建 LLM 客户端
    client = OpenAI(api_key=llm.api_key, base_url=llm.base_url)

    # 解析比赛配置
    if args.config:
        match_config = load_match_config(args.config)
        print(f"\n⚽  启动比赛预测：{match_config.display_title}")
        print(f"   配置文件：{args.config}")
    elif args.match:
        parts = args.match.split(" vs ")
        if len(parts) != 2:
            print('❌ 对阵格式错误，请使用 "队A vs 队B" 格式')
            return
        match_config = quick_match_config(parts[0].strip(), parts[1].strip())
        print(f"\n⚽  启动比赛预测：{match_config.display_title}")
    else:
        parser.print_help()
        return

    # 合并辩论参数：系统默认 → 比赛配置覆盖
    # 系统配置的 debate 来自 system_default.yaml + system.yaml
    # 比赛配置的 debate 来自 matches/*.yaml（可选择性覆盖）
    if sys_config.debate is not None:
        match_config.debate = merge_debate_config(
            sys_config.debate,
            match_config.debate,
        )
    elif match_config.debate is None:
        # 极端情况：系统配置也没有 debate（不应该发生）
        print("  ⚠️  系统配置缺少辩论参数，请检查 system_default.yaml")
        return

    # 用 LLM 配置的温度填充（如果 YAML 未指定 temperature）
    if match_config.debate.temperature is None:
        match_config.debate.temperature = llm.temperature

    print(f"   辩论模式：{'正反方' if match_config.debate.mode == 'pro_con' else '圆桌'}")
    print(f"   LLM：{llm.model} @ {llm.base_url}")

    # 运行预测
    predictor = MatchPredictor(match_config, client, llm.model, sys_config=sys_config)

    # 生成统一的输出目录和时间戳
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    safe_topic = re.sub(r'[\\/:*?"<>|\s]', '_', match_config.display_title)[:40]
    run_dir = Path("debate_output") / safe_topic
    run_dir.mkdir(parents=True, exist_ok=True)

    # 传给 predictor（用于证据池dump）
    predictor._output_dir = run_dir
    predictor._run_timestamp = run_timestamp

    result = predictor.run()

    # 渲染输出
    pro_names = [a.name for a in predictor.pro_agents]
    con_names = [a.name for a in predictor.con_agents]
    all_agents = predictor.pro_agents + predictor.con_agents + predictor.roundtable_agents

    # 从系统配置读取输出参数
    output_cfg = sys_config.output
    cli_truncate = output_cfg.cli_truncate if output_cfg and output_cfg.cli_truncate is not None else 300
    filename_topic_length = output_cfg.filename_topic_length if output_cfg and output_cfg.filename_topic_length is not None else 40

    renderer = Renderer(
        result,
        evidence_pool=predictor.evidence_pool,
        pro_names=pro_names,
        con_names=con_names,
        agents=all_agents,
        match_context=predictor.match_context,
        team_info=match_config.get_team_info(),
        match_meta=match_config.get_match_meta(),
        cli_truncate=cli_truncate,
        filename_topic_length=filename_topic_length,
    )
    renderer.print_live()

    md_path = renderer.save_markdown(path=str(run_dir / f"{run_timestamp}_{safe_topic}.md"))
    json_path = renderer.save_json(path=str(run_dir / f"{run_timestamp}_{safe_topic}.json"), agents=all_agents)
    print(f"📝 Markdown：{md_path}")
    print(f"📊 JSON：{json_path}")

    # 导演提炼：为动画查看器生成浓缩文本（使用与辩论相同的 client 和 model）
    try:
        from .director import process_json_file
        print("🎬 导演正在提炼竞技场展示文本...")
        process_json_file(json_path, client=client, model=llm.model, verbose=True)
        print(f"✅ 导演处理完成：{json_path}")
    except Exception as e:
        print(f"⚠️ 导演处理跳过：{e}")


if __name__ == "__main__":
    main()
