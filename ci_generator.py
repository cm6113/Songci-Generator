"""
宋词生成器 - Song Ci Generator
==============================
基于 LSTM 的多词牌宋词生成模型

支持词牌：
  - 浣溪沙 (双调 42 字，上阕三句下阕三句)
  - 生查子 (双调 40 字，上下阕各四句)
  - 鹧鸪天 (双调 55 字，含两个三字对句)
  - 菩萨蛮 (双调 44 字，句长 7/7/5/5)
  - 蝶恋花 (双调 60 字，结构较复杂)

功能：
  1. 随机生成宋词
  2. 指定词牌生成
  3. 藏头词生成（自定义每句首字）

基于项目: https://github.com/shouxieai/LSTM_poetry_generate
"""

import os
import random
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
#  词牌结构定义
#  每个词牌对应一个列表，表示每句的字数（含上阕 + 下阕）
# ============================================================
CIPAI_STRUCTURE = {
    "浣溪沙": [7, 7, 7, 7, 7, 7],                     # 双调 42 字，上下阕各三句
    "生查子": [5, 5, 5, 5, 5, 5, 5, 5],               # 双调 40 字，上下阕各四句
    "鹧鸪天": [7, 7, 7, 7, 3, 3, 7, 7, 7],            # 双调 55 字，上阕四句 + 下阕五句
    "菩萨蛮": [7, 7, 5, 5, 5, 5, 5, 5],               # 双调 44 字，上下阕各 7,7,5,5
    "蝶恋花": [7, 4, 5, 7, 7, 7, 4, 5, 7, 7],         # 双调 60 字，上下阕各 7,4,5,7,7
}

CIPAI_NAMES = list(CIPAI_STRUCTURE.keys())

# 每个词牌上下阕的分界（上阕句数）
CIPAI_UPPER_STANZA = {
    "浣溪沙": 3,
    "生查子": 4,
    "鹧鸪天": 4,
    "菩萨蛮": 4,
    "蝶恋花": 5,
}


# ============================================================
#  数据预处理 & 词表构建
# ============================================================

def build_vocab(data_file="ci_data.txt", train_num=3000):
    """
    读取宋词数据，构建字符级词表。
    返回: (诗词列表, (词表大小, word2idx, idx2word))
    """
    org_data = open(data_file, "r", encoding="utf-8").read().split("\n")
    org_data = [line.strip() for line in org_data if line.strip()]
    org_data = org_data[:train_num]

    # 收集所有字符
    all_chars = set()
    for line in org_data:
        all_chars.update(line)

    # 构建映射（预留 0 作为 padding/unknown）
    word_2_index = {"<PAD>": 0}
    for i, ch in enumerate(sorted(all_chars), start=1):
        word_2_index[ch] = i

    index_2_word = {v: k for k, v in word_2_index.items()}
    word_size = len(word_2_index)

    print(f"[词表] 共 {word_size} 个字符（含 <PAD>）")
    print(f"[数据] 共 {len(org_data)} 首诗词")

    return org_data, (word_size, word_2_index, index_2_word)


# ============================================================
#  数据集
# ============================================================

class Poetry_Dataset(Dataset):
    """将诗词数据转为 (输入下标序列, 目标下标序列) 的 Dataset"""

    def __init__(self, word_2_index, all_data):
        self.word_2_index = word_2_index
        self.all_data = all_data

    def __getitem__(self, index):
        a_poetry = self.all_data[index]
        a_poetry_index = [self.word_2_index.get(ch, 0) for ch in a_poetry]
        xs = np.array(a_poetry_index[:-1], dtype=np.int64)   # 前 n-1 个字作为输入
        ys = np.array(a_poetry_index[1:], dtype=np.int64)    # 后 n-1 个字作为标签
        return xs, ys

    def __len__(self):
        return len(self.all_data)


def collate_fn(batch):
    """Padding 对齐：同一 batch 内的序列补齐到相同长度"""
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    padded_xs = []
    padded_ys = []
    for x, y in zip(xs, ys):
        pad_len = max_len - len(x)
        padded_xs.append(np.pad(x, (0, pad_len), constant_values=0))
        padded_ys.append(np.pad(y, (0, pad_len), constant_values=0))
    return torch.tensor(np.stack(padded_xs), dtype=torch.long), \
           torch.tensor(np.stack(padded_ys), dtype=torch.long)


# ============================================================
#  LSTM 模型
# ============================================================

class Poetry_Model_lstm(nn.Module):
    def __init__(self, params):
        super().__init__()

        # 加载数据 & 构建词表
        self.all_data, (self.word_size, self.word_2_index, self.index_2_word) = build_vocab(
            data_file=params["data_file"],
            train_num=params["train_num"]
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[设备] 使用: {self.device}")

        self.embedding_num = params["embedding_num"]
        self.hidden_num = params["hidden_num"]
        self.batch_size = params["batch_size"]
        self.epochs = params["epochs"]
        self.lr = params["lr"]
        self.optimizer_type = params["optimizer"]
        self.model_file = params["model_file"]

        # 标点字符集 —— 生成时屏蔽，由代码统一加标点
        self._punct_set = set("，。？！、；：""''（）…—《》〈〉·")
        self._forbidden_ids = None  # 延迟构建

        # 字符嵌入层（替代 Word2Vec，端到端训练）
        self.embedding = nn.Embedding(self.word_size, self.embedding_num, padding_idx=0)

        # 双层 LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_num,
            hidden_size=self.hidden_num,
            batch_first=True,
            num_layers=2,
            bidirectional=False,
            dropout=0.3
        )
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(self.hidden_num, self.word_size)
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 padding

    def forward(self, xs_index, h_0=None, c_0=None):
        """
        前向传播。
        xs_index: (batch, seq_len) 字符下标
        返回: (预测, (h, c))
        """
        batch_size = xs_index.shape[0]

        if h_0 is None or c_0 is None:
            h_0 = torch.zeros(2, batch_size, self.hidden_num, device=self.device)
            c_0 = torch.zeros(2, batch_size, self.hidden_num, device=self.device)

        xs_index = xs_index.to(self.device)
        xs_embedding = self.embedding(xs_index)   # (batch, seq_len, embedding_num)

        hidden, (h_0, c_0) = self.lstm(xs_embedding, (h_0, c_0))
        hidden_drop = self.dropout(hidden)
        pre = self.linear(hidden_drop)            # (batch, seq_len, word_size)

        return pre, (h_0, c_0)

    # --------------------------------------------------------
    #  训练
    # --------------------------------------------------------

    def to_train(self):
        """训练模型（或加载已有模型）"""
        model_file = self.model_file

        if os.path.exists(model_file):
            print(f"[加载] 已有模型权重: {model_file}")
            checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
            self.load_state_dict(checkpoint['model_state_dict'])
            # 恢复非模型属性
            if 'word_2_index' in checkpoint:
                self.word_2_index = checkpoint['word_2_index']
            if 'index_2_word' in checkpoint:
                self.index_2_word = checkpoint['index_2_word']
            if 'word_size' in checkpoint:
                self.word_size = checkpoint['word_size']
            if 'all_data' in checkpoint:
                self.all_data = checkpoint['all_data']
            print(f"[加载] 模型已就绪 (epoch={checkpoint.get('epoch', '?')})")
            return self

        dataset = Poetry_Dataset(self.word_2_index, self.all_data)
        dataloader = DataLoader(dataset, self.batch_size, shuffle=True, collate_fn=collate_fn)

        optimizer = self.optimizer_type(self.parameters(), self.lr)
        self = self.to(self.device)

        print(f"[训练] epochs={self.epochs}  batch={self.batch_size}  "
              f"vocab={self.word_size}  poems={len(self.all_data)}")
        print("-" * 50)

        for e in range(self.epochs):
            total_loss = 0.0
            for batch_index, (batch_x, batch_y) in enumerate(dataloader):
                self.train()
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                pre, _ = self(batch_x)
                # pre: (batch, seq_len, word_size), batch_y: (batch, seq_len)
                loss = self.cross_entropy(pre.reshape(-1, self.word_size), batch_y.reshape(-1))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

                if batch_index % 50 == 0:
                    print(f"  Epoch {e+1:3d}/{self.epochs}  Batch {batch_index:3d}  Loss: {loss:.4f}")

            avg_loss = total_loss / max(len(dataloader), 1)
            print(f"[Epoch {e+1:3d}] 平均 Loss: {avg_loss:.4f}")

            # 每 200 epoch 展示一次生成效果
            if (e + 1) % 200 == 0:
                print("\n  --- 训练中试生成 ---")
                self.generate_ci(random.choice(CIPAI_NAMES))

        # 保存模型
        torch.save({
            'epoch': self.epochs,
            'model_state_dict': self.state_dict(),
            'word_2_index': self.word_2_index,
            'index_2_word': self.index_2_word,
            'word_size': self.word_size,
            'all_data': self.all_data,
        }, model_file)
        print(f"[保存] 模型已保存: {model_file}")
        return self

    # --------------------------------------------------------
    #  辅助：将一段文本逐字喂入 LSTM
    # --------------------------------------------------------

    def _feed_sequence(self, text, h_0, c_0):
        """
        将 text 中的每个字符依次通过 LSTM，返回最终的隐状态 (h, c)。
        用于将前缀（词牌名：）编码为上下文。
        """
        for ch in text:
            wid = self.word_2_index.get(ch, 0)
            idx = torch.tensor([[wid]], dtype=torch.long)
            _, (h_0, c_0) = self(idx, h_0, c_0)
        return h_0, c_0

    # --------------------------------------------------------
    #  宋词生成（核心）
    # --------------------------------------------------------

    def _sample_token(self, logits, temperature=0.8, recent_chars=None, penalty=2.0, top_k=20):
        """
        基于温度 + top-k + 重复惩罚的采样策略。
        禁止生成标点符号（标点由代码在句末统一添加）。

        参数:
            logits: (word_size,) 原始 logits
            temperature: 温度系数，越低越确定，越高越随机
            recent_chars: 最近出现过的字符集合，用于重复惩罚
            penalty: 重复惩罚强度（>1.0 惩罚，1.0 不惩罚）
            top_k: 保留概率最高的前 k 个 token
        返回:
            int: 选中的字符下标
        """
        # 延迟构建禁止 token 列表
        if self._forbidden_ids is None:
            self._forbidden_ids = [
                self.word_2_index.get(ch, -1) for ch in self._punct_set
                if self.word_2_index.get(ch, -1) >= 0
            ]

        # 温度缩放
        logits = logits / max(temperature, 0.01)

        # 屏蔽标点符号
        for fid in self._forbidden_ids:
            logits[fid] = float('-inf')

        # 重复惩罚：降低最近出现过的字符的概率
        if recent_chars and penalty > 1.0:
            for ch in recent_chars:
                wid = self.word_2_index.get(ch)
                if wid is not None and wid not in self._forbidden_ids:
                    logits[wid] /= penalty

        # Top-k 过滤
        if top_k > 0 and top_k < len(logits):
            topk_values, topk_indices = torch.topk(logits, top_k)
            mask = torch.full_like(logits, float('-inf'))
            mask[topk_indices] = topk_values
            logits = mask

        # Softmax + 多项式采样
        probs = torch.softmax(logits, dim=-1)
        wid = int(torch.multinomial(probs, num_samples=1).item())
        return wid

    def generate_ci(self, cipai_name, acrostic=None, verbose=True,
                    temperature=0.75, repeat_penalty=2.5, top_k=30):
        """
        根据词牌结构逐句生成宋词。

        参数:
            cipai_name (str): 词牌名
            acrostic (str, 可选): 藏头词的首字序列（每句第一个字）
            verbose (bool): 是否打印结果
            temperature (float): 采样温度（0.5-1.2，低=保守，高=随机）
            repeat_penalty (float): 重复惩罚（1.0=不惩罚，2.0+=强惩罚）
            top_k (int): top-k 采样保留数

        返回:
            str: 生成的完整宋词（含词牌前缀）
        """
        self.eval()

        structure = CIPAI_STRUCTURE.get(cipai_name)
        if structure is None:
            print(f"[错误] 未知词牌: {cipai_name}")
            print(f"        可选: {', '.join(CIPAI_NAMES)}")
            return None

        # 初始化隐状态
        h_0 = torch.zeros(2, 1, self.hidden_num, device=self.device)
        c_0 = torch.zeros(2, 1, self.hidden_num, device=self.device)

        # 用词牌名作为前缀，建立生成语境
        prefix = f"{cipai_name}："
        h_0, c_0 = self._feed_sequence(prefix, h_0, c_0)
        result = prefix

        # 已生成字符追踪（用于重复惩罚）
        recent_chars = set()

        # 处理藏头字
        acrostic_chars = list(acrostic) if acrostic else []

        # 逐句生成
        for i, sentence_len in enumerate(structure):
            # --- 藏头模式：使用用户指定的首字 ---
            if i < len(acrostic_chars):
                head_char = acrostic_chars[i]
                if head_char in self.word_2_index:
                    result += head_char
                    wid = self.word_2_index[head_char]
                    idx = torch.tensor([[wid]], dtype=torch.long, device=self.device)
                    _, (h_0, c_0) = self(idx, h_0, c_0)
                    recent_chars.add(head_char)
                    chars_to_generate = sentence_len - 1
                else:
                    print(f"[警告] '{head_char}' 不在词表中，该句将随机生成")
                    chars_to_generate = sentence_len
            else:
                chars_to_generate = sentence_len

            # --- 逐字生成 ---
            for j in range(chars_to_generate):
                last_char = result[-1]
                wid = self.word_2_index.get(last_char, 0)
                idx = torch.tensor([[wid]], dtype=torch.long, device=self.device)
                pre, (h_0, c_0) = self(idx, h_0, c_0)

                # 使用改进的采样策略
                wid = self._sample_token(
                    pre[0, 0],
                    temperature=temperature,
                    recent_chars=recent_chars,
                    penalty=repeat_penalty,
                    top_k=top_k
                )
                new_char = self.index_2_word[wid]
                result += new_char
                recent_chars.add(new_char)

                # 保持 recent_chars 窗口大小合理
                if len(recent_chars) > 20:
                    recent_chars = set(list(recent_chars)[-15:])

            # --- 加标点 ---
            if i < len(structure) - 1:
                result += "，"
            else:
                result += "。"

            # 标点也作为上下文喂入 LSTM
            punct = result[-1]
            if punct in self.word_2_index:
                wid = self.word_2_index[punct]
                idx = torch.tensor([[wid]], dtype=torch.long, device=self.device)
                _, (h_0, c_0) = self(idx, h_0, c_0)

        # --- 美化输出 ---
        if verbose:
            self._pretty_print(cipai_name, result, prefix, structure)

        return result

    def _pretty_print(self, cipai_name, result, prefix, structure):
        """分上下阕美化打印"""
        upper_lines = CIPAI_UPPER_STANZA.get(cipai_name, len(structure) // 2)
        poem_body = result[len(prefix):]  # 去掉词牌前缀

        # 按标点拆句
        lines = []
        current = ""
        for ch in poem_body:
            current += ch
            if ch in "，。？！":
                lines.append(current)
                current = ""
        if current:
            lines.append(current)

        print(f"\n{'=' * 50}")
        print(f"  [{cipai_name}]")
        print(f"{'=' * 50}")

        if len(lines) <= upper_lines:
            print("  【上阕】")
            for line in lines:
                print(f"    {line}")
        else:
            print("  【上阕】")
            for line in lines[:upper_lines]:
                print(f"    {line}")
            print("  【下阕】")
            for line in lines[upper_lines:]:
                print(f"    {line}")
        print(f"{'═' * 50}\n")

    # --------------------------------------------------------
    #  随机生成 & 藏头词交互
    # --------------------------------------------------------

    def generate_random_ci(self):
        """随机选择一个词牌并生成"""
        cipai = random.choice(CIPAI_NAMES)
        return self.generate_ci(cipai)

    def generate_acrostic_ci(self, cipai_name=None):
        """交互式藏头词生成"""
        if cipai_name is None:
            print(f"可选词牌: {', '.join(CIPAI_NAMES)}")
            cipai_name = input("请输入词牌名: ").strip()

        structure = CIPAI_STRUCTURE.get(cipai_name)
        if structure is None:
            print(f"[错误] 未知词牌: {cipai_name}")
            return None

        num_sentences = len(structure)
        print(f"\n{cipai_name} 共 {num_sentences} 句，需要 {num_sentences} 个藏头字")
        print(f"句子结构: {' → '.join(f'{s}言' for s in structure)}")
        print(f"每字将作为对应句子的第一个字\n")

        head_text = input(f"请输入 {num_sentences} 个藏头字: ").strip()

        if len(head_text) < num_sentences:
            print(f"[提示] 输入了 {len(head_text)} 字，不足 {num_sentences} 字。")
            print(f"       剩余句子将随机生成首字。")

        return self.generate_ci(cipai_name, acrostic=head_text)


# ============================================================
#  主程序入口
# ============================================================

if __name__ == "__main__":
    print("+" + "=" * 58 + "+")
    print("|" + "  ** 宋词生成器 - Song Ci Generator **".ljust(48) + "|")
    print("|" + "  基于双层 LSTM 的多词牌宋词生成模型".ljust(48) + "|")
    print("+" + "=" * 58 + "+")

    # ======================== 可调参数 ========================
    params = {
        "data_file": "ci_data.txt",
        "model_file": "ci_model.pt",
        "batch_size": 32,
        "epochs": 500,
        "lr": 0.003,
        "hidden_num": 128,
        "embedding_num": 128,
        "train_num": 3000,
        "optimizer": torch.optim.AdamW,
    }

    print(f"\n[配置信息]")
    print(f"   . 词向量维度:  {params['embedding_num']}")
    print(f"   . LSTM 隐层:   {params['hidden_num']}")
    print(f"   . 训练轮数:    {params['epochs']}")
    print(f"   . Batch 大小:  {params['batch_size']}")
    print(f"   . 学习率:      {params['lr']}")
    print(f"   . 优化器:      AdamW")

    # ---- 初始化 & 训练 ----
    model = Poetry_Model_lstm(params)
    model = model.to_train()

    # ---- 交互菜单 ----
    print(f"\n{'=' * 50}")
    print("  [模型就绪!] 请选择功能：")
    print(f"{'=' * 50}")

    while True:
        print()
        print("  1. 随机生成    -- 随机选词牌，生成一首宋词")
        print("  2. 指定词牌    -- 选择词牌后生成")
        print("  3. 藏头词      -- 自定义每句首字")
        print("  4. 词牌列表    -- 查看支持的词牌及结构")
        print("  0. 退出")
        print()

        choice = input(">> 请选择 (0-4): ").strip()

        if choice == "0":
            print("再见!")
            break
        elif choice == "1":
            model.generate_random_ci()
        elif choice == "2":
            print(f"\n可选词牌: {', '.join(CIPAI_NAMES)}")
            cipai = input("词牌名: ").strip()
            if cipai:
                model.generate_ci(cipai)
        elif choice == "3":
            print(f"\n可选词牌: {', '.join(CIPAI_NAMES)}")
            cipai = input("词牌名: ").strip()
            if cipai:
                model.generate_acrostic_ci(cipai)
        elif choice == "4":
            print(f"\n{'─' * 50}")
            print("  📋 支持的词牌及其结构")
            print(f"{'─' * 50}")
            for name, struct in CIPAI_STRUCTURE.items():
                up = CIPAI_UPPER_STANZA[name]
                upper = " → ".join(f"{s}言" for s in struct[:up])
                lower = " → ".join(f"{s}言" for s in struct[up:])
                print(f"  【{name}】")
                print(f"    上阕: {upper}")
                print(f"    下阕: {lower}")
                print(f"    共 {len(struct)} 句 · {sum(struct)} 字")
                print()
        else:
            print("[X] 无效选项，请重新输入")
