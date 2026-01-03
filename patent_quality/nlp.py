import re
from typing import List, Set
import jieba
import os
from .log import get_logger

PATENT_STOPWORDS = [
    "本发明",
    "实用新型",
    "外观设计",
    "权利要求",
    "特征",
    "涉及",
    "公开",
    "提供",
    "实施例",
    "所述",
    "其中",
    "及其",
    "用于",
    "一种",
    "方法",
    "装置",
    "系统",
    "组件",
    "结构",
    "步骤",
    "属于",
    "技术领域",
    "背景技术",
    "优选",
    "位于",
    "连接",
    "附图",
    "示意图",
    "包含",
    "包括",
    "使得",
    "能够",
    "相比",
    "现有技术",
    "其特征在于",
]

_re_cn = re.compile(r"[\u4e00-\u9fff]+")


def init_jieba(user_dict_path: str | None) -> None:
    logger = get_logger(level="INFO")
    if user_dict_path:
        try:
            if os.path.exists(user_dict_path):
                size = os.path.getsize(user_dict_path)
                logger.info(f"加载用户词典: {user_dict_path} 大小={size/1024/1024:.2f}MB")
                jieba.load_userdict(user_dict_path)
                logger.info("用户词典加载完成")
            else:
                logger.warning(f"用户词典路径不存在: {user_dict_path}，跳过加载")
        except Exception as e:
            logger.error(f"加载用户词典失败: {e}")


def load_stopwords(paths: List[str]) -> Set[str]:
    logger = get_logger(level="INFO")
    s: Set[str] = set(PATENT_STOPWORDS)
    for p in paths:
        # 如果是目录，收集所有 .txt 文件
        if os.path.isdir(p):
            txt_files = [os.path.join(root, f)
                         for root, _, files in os.walk(p)
                         for f in files if f.lower().endswith('.txt')]
        else:
            txt_files = [p]

        for txt_path in txt_files:
            # 尝试 utf-8 读取
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    for line in f:
                        w = line.strip()
                        if w:
                            s.add(w)
                logger.info(f"停用词载入: {txt_path} (utf-8)")
            except Exception:
                # 尝试 gb18030 读取
                try:
                    with open(txt_path, "r", encoding="gb18030") as f:
                        for line in f:
                            w = line.strip()
                            if w:
                                s.add(w)
                    logger.info(f"停用词载入: {txt_path} (gb18030)")
                except Exception as e:
                    logger.warning(f"停用词载入失败: {txt_path}")
                    # 输出异常信息
                    logger.error(e)
                    continue
    return s


def tokenize(text: str, stopwords: Set[str]) -> List[str]:
    toks = []
    for w in jieba.cut(text, HMM=True):
        if not w:
            continue
        if w in stopwords:
            continue
        if not _re_cn.fullmatch(w):
            continue
        toks.append(w)
    return toks
