"""
Fortune App (Bazi / Four Pillars) — Streamlit + OpenAI

Quick start (local):
1) Python 3.10+ recommended
2) pip install -r requirements.txt  (requirements content shown below in this file)
3) Set env var:  export OPENAI_API_KEY=your_key_here
   (optional) export OPENAI_MODEL=gpt-4o-mini  # or gpt-4o / o3-mini, etc.
4) streamlit run app.py

Notes:
- Accurate sexagenary pillars (year/month/day/hour) are computed via the `lunar_python` library.
- If the library isn't available, the app will show a friendly error. Please install dependencies.
- This is an MVP; you can adjust prompt, tone, and feature flags in the CONFIG section below.
"""

from __future__ import annotations
import os
import sys
import json
import traceback
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import streamlit as st

# Try to import lunar_python for precise Bazi computation
try:
    from lunar_python import Solar  # type: ignore
    LUNAR_AVAILABLE = True
except Exception:
    LUNAR_AVAILABLE = False

# Try to import OpenAI SDK (modern)
try:
    from openai import OpenAI  # type: ignore
    OPENAI_SDK = "v1"
except Exception:
    # fallback older import name
    try:
        import openai  # type: ignore
        OPENAI_SDK = "legacy"
    except Exception:
        OPENAI_SDK = None

# ======================
# CONFIG
# ======================
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SYSTEM_TONE = (
    "你是一位严谨而温和的八字命理师，擅长以现代、通俗、可执行的建议来解释命理。"
    "请保持积极、中立、避免宿命论；输出需落地且不涉迷信恐吓。"
)
OUTPUT_SECTIONS = [
    ("总体格局", "概述命局特点、五行强弱与平衡重点。"),
    ("事业与学业", "优势、阻碍、适合的方向与具体建议。"),
    ("财运与规划", "正财/偏财倾向、理财建议与风险提示。"),
    ("感情与人际", "沟通方式、择友/择偶建议、可改进点。"),
    ("健康与作息", "体质倾向、作息饮食、运动建议。"),
    ("当下周期", "以近1年—3年的运势节奏给出注意事项与关键月份/季度。"),
    ("行动清单", "3—5条可执行的To-Do，需具体到行为与频率。")
]

HEAVENLY_STEMS = list("甲乙丙丁戊己庚辛壬癸")
EARTHLY_BRANCHES = list("子丑寅卯辰巳午未申酉戌亥")
GAN_TO_WUXING = {
    "甲": "木", "乙": "木",
    "丙": "火", "丁": "火",
    "戊": "土", "己": "土",
    "庚": "金", "辛": "金",
    "壬": "水", "癸": "水",
}
ZHI_TO_WUXING = {
    "子": "水", "丑": "土", "寅": "木", "卯": "木",
    "辰": "土", "巳": "火", "午": "火", "未": "土",
    "申": "金", "酉": "金", "戌": "土", "亥": "水",
}

@dataclass
class Bazi:
    year: str
    month: str
    day: str
    hour: str

    def to_dict(self) -> Dict[str, str]:
        return {"year": self.year, "month": self.month, "day": self.day, "hour": self.hour}

    def wuxing_summary(self) -> Dict[str, int]:
        counts = {"木": 0, "火": 0, "土": 0, "金": 0, "水": 0}
        for pillar in [self.year, self.month, self.day, self.hour]:
            if len(pillar) >= 2:
                gan, zhi = pillar[0], pillar[1]
                counts[GAN_TO_WUXING.get(gan, "")] = counts.get(GAN_TO_WUXING.get(gan, ""), 0) + 1 if GAN_TO_WUXING.get(gan) else counts.get("木",0)
                counts[ZHI_TO_WUXING.get(zhi, "")] = counts.get(ZHI_TO_WUXING.get(zhi, ""), 0) + 1 if ZHI_TO_WUXING.get(zhi) else counts.get("木",0)
        return counts


# ======================
# Core: compute Bazi via lunar_python
# ======================

def compute_bazi(year: int, month: int, day: int, hour: int, minute: int = 0) -> Bazi:
    if not LUNAR_AVAILABLE:
        raise RuntimeError(
            "未检测到 `lunar_python`，请先安装依赖：pip install lunar-python\n"
            "或在终端执行：pip install -r requirements.txt"
        )
    # `Solar` expects local time; users should input local birth time.
    solar = Solar.fromYmdHms(year, month, day, hour, minute, 0)
    lunar = solar.getLunar()
    ec = lunar.getEightChar()
    # The library returns strings like '甲子', '乙丑' etc.
    gz_year = ec.getYear()
    gz_month = ec.getMonth()
    gz_day = ec.getDay()
    gz_time = ec.getTime()
    return Bazi(year=gz_year, month=gz_month, day=gz_day, hour=gz_time)


# ======================
# OpenAI client helpers
# ======================

def get_openai_client():
    if OPENAI_SDK == "v1":
        return OpenAI()
    elif OPENAI_SDK == "legacy":
        # create a thin wrapper for legacy style
        class Legacy:
            def __init__(self):
                self.lib = openai
            def chat_complete(self, model: str, messages: List[Dict[str, str]]):
                return self.lib.ChatCompletion.create(model=model, messages=messages)
        return Legacy()
    else:
        raise RuntimeError(
            "未检测到 OpenAI SDK。请先安装：pip install openai"
        )


def call_llm_for_reading(bazi: Bazi, profile: Dict[str, Any]) -> str:
    """Compose the system & user prompts and call the LLM to generate the reading."""
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    system_msg = SYSTEM_TONE

    # Prepare brief Wuxing stats for extra context
    wx = bazi.wuxing_summary()

    user_payload = {
        "input_format": {
            "bazi": bazi.to_dict(),
            "wuxing_counts": wx,
            "user_profile": profile,
        },
        "instructions": {
            "style": "简洁、真诚、接地气、可执行，避免吓唬与绝对化结论",
            "sections": OUTPUT_SECTIONS,
            "format": "使用有层级的小标题与项目符号；重点给出具体行动建议",
        }
    }

    # Build messages
    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": (
                "请基于以下八字信息进行命理分析，并按照 `sections` 输出结构化中文报告。\n"
                + json.dumps(user_payload, ensure_ascii=False, indent=2)
            ),
        },
    ]

    client = get_openai_client()

    if OPENAI_SDK == "v1":
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0.7)
        return resp.choices[0].message.content.strip()
    else:
        # legacy wrapper path
        resp = client.chat_complete(model=model, messages=messages)
        return resp["choices"][0]["message"]["content"].strip()


# ======================
# Streamlit UI
# ======================

st.set_page_config(page_title="八字算命 · MVP", page_icon="🔮", layout="centered")
st.title("🔮 八字算命 · MVP")
st.caption("输入出生信息，自动排盘并生成结构化的运势建议（教育娱乐用途）")

with st.expander("使用说明", expanded=False):
    st.markdown(
        """
        - **准确性**：本工具使用权威农历/干支库 `lunar_python` 进行排盘；请尽量提供**出生地时区对应的本地时间**。
        - **隐私**：数据仅用于生成本次报告；若勾选保存，将写入本地会话状态（示例级）。
        - **用途**：报告仅供参考，不构成医疗、法律或财务建议。
        """
    )

with st.form("bazi_form"):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("姓名（可选）", value="")
        gender = st.selectbox("性别（可选）", ["未填写", "男", "女", "其他"], index=0)
        save_history = st.checkbox("保存到历史记录（本地会话示例）", value=False)
    with col2:
        birth_date = st.date_input("出生日期", help="请选择阳历生日")
        birth_time = st.time_input("出生时间")

    addl = st.text_area("补充背景（可选）", placeholder="例如：目前职业、关注的问题（事业/感情/健康/财务）、目标等…")

    submitted = st.form_submit_button("生成运势报告 ✨")

if submitted:
    try:
        if not LUNAR_AVAILABLE:
            st.error("未检测到 `lunar_python` 依赖，请先在环境中安装：`pip install lunar-python`。")
            st.stop()
        if OPENAI_SDK is None:
            st.error("未检测到 OpenAI SDK，请先安装：`pip install openai` 并设置环境变量 OPENAI_API_KEY。")
            st.stop()
        if not os.getenv("OPENAI_API_KEY"):
            st.error("未设置 OPENAI_API_KEY。请在终端执行：`export OPENAI_API_KEY=your_key_here` 后重试。")
            st.stop()

        y, m, d = birth_date.year, birth_date.month, birth_date.day
        h, minute = birth_time.hour, birth_time.minute

        bz = compute_bazi(y, m, d, h, minute)

        st.subheader("排盘结果（四柱）")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("年柱", bz.year)
        c2.metric("月柱", bz.month)
        c3.metric("日柱", bz.day)
        c4.metric("时柱", bz.hour)

        wx = bz.wuxing_summary()
        st.write("五行计数（粗略）：", wx)

        profile = {
            "name": name,
            "gender": gender,
            "notes": addl.strip(),
        }

        with st.status("正在生成命理解读……", expanded=False):
            reading = call_llm_for_reading(bz, profile)

        st.markdown("---")
        st.subheader("命理解读与建议")
        st.write(reading)

        if save_history:
            if "history" not in st.session_state:
                st.session_state["history"] = []
            st.session_state["history"].append({
                "profile": profile,
                "bazi": bz.to_dict(),
                "wuxing": wx,
                "output": reading,
            })

    except Exception as e:
        st.error("出错了：" + str(e))
        st.exception(e)

# History viewer
if "history" in st.session_state and st.session_state["history"]:
    with st.expander("历史记录", expanded=False):
        for i, item in enumerate(reversed(st.session_state["history"])):
            st.markdown(f"**记录 {len(st.session_state['history']) - i}**")
            cols = st.columns(4)
            cols[0].write("年柱:" + item["bazi"]["year"])
            cols[1].write("月柱:" + item["bazi"]["month"])
            cols[2].write("日柱:" + item["bazi"]["day"])
            cols[3].write("时柱:" + item["bazi"]["hour"])
            st.write(item["output"])
            st.markdown("---")

# ======================
# Inline requirements for convenience (copy to requirements.txt)
# ======================

REQUIREMENTS = """
openai>=1.0.0
streamlit>=1.33.0
lunar-python>=1.6.5
"""

with st.expander("requirements.txt（点击展开以复制）", expanded=False):
    st.code(REQUIREMENTS, language="text")
