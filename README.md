# 宋词生成器 · Song Ci Generator

基于双层 LSTM 的多词牌宋词生成模型，支持 5 种经典词牌，可随机生成、指定词牌生成及藏头词创作。

## 项目简介

使用 PyTorch 实现字符级 LSTM 语言模型，在 704 首历代宋词原文上训练，能按照指定词牌的结构（句数、字数）自动生成符合格律的宋词。

## 支持词牌

| 词牌 | 字数 | 结构 | 说明 |
|------|:---:|------|------|
| 浣溪沙 | 42 | 7/7/7 + 7/7/7 | 双调，上下阕各三句七言 |
| 生查子 | 40 | 5×8 句 | 双调，上下阕各四句五言 |
| 鹧鸪天 | 55 | 7/7/7/7 + 3/3/7/7/7 | 双调，含两个三字对句 |
| 菩萨蛮 | 44 | 7/7/5/5 + 5/5/5/5 | 双调，句长交替 |
| 蝶恋花 | 60 | 7/4/5/7/7 + 7/4/5/7/7 | 双调，结构较复杂 |

## 项目结构

```
SongCi-Generator/
│
├── ci_generator.py      # 核心代码：模型定义、训练、生成、交互菜单
├── ci_data.txt          # 训练数据：704 首宋词
├── ci_model.pt          # 预训练模型权重（可直接生成）
├── README.md            # 项目说明
└── output_final.txt     # 生成示例输出
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch >= 1.10
- NumPy

### 安装

```bash
pip install torch numpy
```

### 运行

```bash
python ci_generator.py
```

交互菜单：

```
1. 随机生成    -- 随机选词牌，生成一首宋词
2. 指定词牌    -- 选择词牌后生成
3. 藏头词      -- 自定义每句首字
4. 词牌列表    -- 查看支持的词牌及结构
0. 退出
```

## 重新训练

删除旧模型后运行程序将自动训练：

```bash
rm ci_model.pt            # 或手动删除
python ci_generator.py    # 自动检测并训练
```

可在 `ci_generator.py` 的 `params` 字典中调整超参数：

```python
params = {
    "batch_size": 32,
    "epochs": 300,
    "lr": 0.003,
    "hidden_num": 128,
    "embedding_num": 128,
}
```

## 技术细节

**模型架构：** nn.Embedding(128维) → 双层 LSTM(128维, Dropout 0.3) → Linear → CrossEntropyLoss

**生成策略：** 温度缩放 + Top-K 过滤 + 重复惩罚 + 多项式采样

**结构约束：** 生成时屏蔽标点符号，按词牌句长强制断句，标点由程序统一添加

## 数据来源

704 首宋词原文，收录自历代古籍：全唐诗、全宋词、花间集、纳兰词等，涵盖唐、五代、北宋、南宋、金元、明、清各代词人作品。

## 参考与致谢

本项目在以下开源项目基础上完成：

### [shouxieai/LSTM_poetry_generate](https://github.com/shouxieai/LSTM_poetry_generate)

- 参考了其 LSTM 诗歌生成的核心思路与模型框架
- 原项目使用 gensim Word2Vec 预训练词向量，本项目改用 PyTorch nn.Embedding 端到端训练，去除了 gensim 依赖
- 新增了温度采样、Top-K 过滤、重复惩罚等生成策略，替代原项目的简单 argmax
- 重新设计了词牌结构约束机制，强制句长匹配格律

### [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)

- 最全中华古诗词数据库，提供 2.1 万首宋词 JSON 数据
- 本项目部分训练数据来源于此，经筛选整理为 5 种词牌共 704 首

## License

MIT
