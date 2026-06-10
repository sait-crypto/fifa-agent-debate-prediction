# Debate Arena — 多智能体辩论预测系统

基于 LLM 的多智能体辩论框架，通过不同视角的 Agent 对抗辩论，对体育赛事（如 2026 FIFA World Cup）进行预测。

## 快速开始

### 1. 安装依赖

```bash
# 使用 uv（推荐）
uv sync

# 或 pip
pip install -e .
```

### 2. 配置 LLM

编辑 `.env` 文件，填入你的 API Key：

```bash
# DeepSeek（默认预设）
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

内置预设：`openai` / `deepseek` / `deepseek-reasoner` / `siliconflow` / `ollama`

可在 `.env` 中按 `{前缀}_API_KEY` / `{前缀}_BASE_URL` / `{前缀}_MODEL` 格式添加自定义预设。

### 3. 运行辩论

```bash
# 快速模式 — 只需队名即可启动
python -m debate_arena "巴西 vs 阿根廷"

# 配置模式 — 使用比赛配置文件（推荐）
python -m debate_arena --config matches/A_1st_round_kor_ vs_cze.yaml

# 指定 LLM 预设
python -m debate_arena --preset deepseek --config matches/semi_bra_vs_arg.yaml

# 查看可用预设
python -m debate_arena --list-presets
```

---

## 配置体系

配置分三层，高优先级覆盖低优先级：

```
system_default.yaml  →  system.yaml  →  matches/*.yaml
   （默认值，勿改）      （用户覆盖）     （比赛专属覆盖）
```

### 系统配置

**`system_default.yaml`** — 所有默认值，不要直接修改。

**`system.yaml`** — 仅写需要修改的项：

```yaml
llm_preset: "deepseek"

debate:
  mode: "roundtable"           # "pro_con" | "roundtable"
```

### 可配置参数一览

| 分组 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| **debate** | `mode` | `pro_con` | 辩论模式：正反方 / 圆桌 |
| | `pro_count` / `con_count` | 2 / 2 | 正反方各几个 Agent |
| | `agent_count` | 5 | 圆桌模式 Agent 数 |
| | `max_rounds` | 4 | 最大辩论轮次 |
| | `agent_search` | true | Agent 是否可自主检索证据 |
| | `agent_speech_hint` | 800 | 发言字数软限制提示（0=不限） |
| **output** | `cli_verbose` | false | true 时实时输出 LLM 完整回复 |
| | `cli_truncate` | 300 | 终端显示截断字数 |
| | `dump_evidence_pool` | false | 辩论中实时导出完整证据池文件 |
| **limits** | `other_agent_summary` | 300 | 非交互 Agent 消息摘要长度 |
| | `free_challenge_retries` | 2 | 自由质疑重试次数 |
| **fetch** | `max_content_length` | 5000 | 网页抓取最大字符数 |
| | `fetch_timeout` | 10 | HTTP 超时秒数 |
| | `search_max_results` | 5 | 搜索结果数上限 |
| **temperatures** | `arbitrator` | 0.3 | 仲裁者温度 |
| | `thinking` | 0.3 | Agent CoT 思考温度 |
| | `evidence_curator` | 0.1 | 证据审核员温度 |
| | `round_summarizer` | 0.1 | 场记员温度 |

---

## 比赛配置

### 创建比赛配置

复制模板并填写：

```bash
cp matches/_template.yaml matches/my_match.yaml
```

**只有 `match.home` 和 `match.away` 必填**，其余全部可选。

```yaml
tournament:
  name: "2026 FIFA World Cup"
  stage: "group"
  group: "A组"

match:
  home: "韩国"           # 必填
  away: "捷克"           # 必填
  date: "2026-06-12"

home_team:
  formation: "4-3-3"
  lineup: []
  injuries: []
  suspensions: []

away_team:
  formation: "4-2-3-1"
  lineup: []
  injuries: []
  suspensions: []

# 百度体育分析数据（见下方说明）
analysis_file: "tools/532539_韩国vs捷克_20260604_20260608.json"

# 本地证据文件
evidence_files:
  - "bra_vs_arg_brazil"
  - "bra_vs_arg_argentina"
  - "bra_vs_arg_h2h"
```

### 分析数据（analysis_file）

百度体育分析数据可自动注入情报和战绩，省去手动填写历史记录：

- 设置 `analysis_file` 后，**有利/不利情报**自动归入证据池（高置信度 0.9）
- **战绩记录**自动替代 YAML 中的 `home_history` / `away_history` / `h2h`
- 所有胜负预测数据会被**自动排除**，不影响辩论客观性

```yaml
# 指向 tools/ 目录下的 JSON 文件
analysis_file: "532539_韩国vs捷克_20260604_20260608.json"
```

### 本地证据文件

在 `evidence/` 目录下创建 YAML 文件，内容为纯字符串列表：

```yaml
# evidence/bra_vs_arg_brazil.yaml
- "巴西近5场比赛4胜1平，内马尔伤愈回归状态出色"
- "巴西本届赛事防守稳固，仅失2球"
```

在比赛配置中引用（可省略 `.yaml` 后缀）：

```yaml
evidence_files:
  - "bra_vs_arg_brazil"
  - "bra_vs_arg_argentina"
```

---

## 数据提取工具

### BaiduTiyuAbstract.py — 从百度体育页面提取分析数据

```bash
# 默认读取 tools/page_source.html
python tools/BaiduTiyuAbstract.py

# 指定 HTML 文件
python tools/BaiduTiyuAbstract.py path/to/page.html
```

**使用流程：**

1. 在浏览器打开百度体育比赛页面
2. 保存页面源代码（Ctrl+U → 全选复制 → 保存为 `tools/page_source.html`）
3. 运行提取脚本 → 自动生成 JSON 文件到 `tools/` 目录

**自动命名格式：** `{赛次}_{主队}vs{客队}_{比赛日期}_{生成日期}.json`

例：`532539_韩国vs捷克_20260604_20260608.json`

然后在比赛配置中绑定：

```yaml
analysis_file: "532539_韩国vs捷克_20260604_20260608.json"
```

---

## 辩论模式

### 正反方模式（pro_con）

- 分为正方（支持主队胜）和反方（支持客队胜）
- 配置：`pro_count` / `con_count` 控制 Agent 数量
- 支持质疑环节：`pro_con_challenge_enabled`

### 圆桌模式（roundtable）

- 所有 Agent 自由选择立场
- 两阶段质疑：
  - **自由质疑**（phase1）：Agent 自选质疑对象
  - **分配质疑**（phase2）：系统分配对立观点者互相质疑
- 配置：`agent_count`、`phase1_challenge_count`、`phase2_challenge_count`

---

## 输出

辩论结束后自动生成三份输出到 `debate_output/` 目录：

| 格式 | 文件 | 内容 |
|------|------|------|
| 🖥️ 终端 | 实时彩色输出 | 截断显示（`cli_truncate` 可调） |
| 📝 Markdown | `{日期}_{话题}.md` | 完整辩论记录 + 仲裁结论 |
| 📊 JSON | `{日期}_{话题}.json` | 结构化完整记录（含 Agent 视角、预测链、证据池） |

开启 `output.dump_evidence_pool: true` 后，辩论过程中还会实时维护证据池完整导出文件。

---

## 项目结构

```
├── .env                          # LLM 预设（API Key / Base URL / Model）
├── system_default.yaml           # 系统默认配置（勿改）
├── system.yaml                   # 用户覆盖配置
├── matches/                      # 比赛配置
│   ├── _template.yaml            # 模板
│   ├── semi_bra_vs_arg.yaml      # 半决赛：巴西 vs 阿根廷
│   └── A_1st_round_kor_ vs_cze.yaml  # A组：韩国 vs 捷克
├── evidence/                     # 本地证据文件（YAML 字符串列表）
├── tools/                        # 数据提取工具 + 分析数据
│   ├── BaiduTiyuAbstract.py      # 百度体育页面提取器
│   ├── page_source.html          # 页面源代码（输入）
│   └── *.json                    # 提取的分析数据（输出）
├── debate_output/                # 辩论结果输出（自动创建）
└── debate_arena/                 # 核心代码
    ├── main.py                   # CLI 入口
    ├── config.py                 # 系统配置 + LLM 预设加载
    ├── match_config.py           # 比赛配置 + 分析数据集成
    ├── analysis_parser.py        # 百度体育分析数据解析
    ├── agent.py                  # Agent（CoT + 证据感知 + 检索）
    ├── predict.py                # 辩论编排器
    ├── evidence.py               # 证据池（共用/私有/临时）
    ├── prompts.py                # 所有 Prompt 模板
    ├── researcher.py             # 搜索 + 抓取 + LLM 降级检索
    ├── fetcher.py                # URL 内容抓取 + HTML 清洗
    ├── renderer.py               # 终端 + Markdown + JSON 输出
    └── models.py                 # 核心数据模型
```
#   f i f a - a g e n t - d e b a t e - p r e d i c t i o n  
 