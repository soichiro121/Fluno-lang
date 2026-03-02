# gen_grammar.py
import json

# 5次元ベクトル: [生物性, 具体性, ポジティブ, エネルギー, 複雑さ]

# 名詞 (Nouns)
nouns = {
    "Love": [0.8, 0.2, 1.0, 0.9, 0.8],
    "Life": [1.0, 0.6, 0.8, 0.9, 1.0],
    "Death": [0.0, 0.5, 0.1, 0.0, 0.9],
    "Void": [0.0, 0.0, 0.4, 0.1, 1.0],
    "Hope": [0.7, 0.0, 1.0, 0.7, 0.5],
    "Fear": [0.9, 0.1, 0.0, 0.8, 0.6],
    "Time": [0.0, 0.0, 0.5, 0.6, 0.9],
    "Silence": [0.1, 0.1, 0.6, 0.0, 0.2],
    "Machine": [0.0, 1.0, 0.5, 0.7, 0.6],
    "Soul": [1.0, 0.0, 0.9, 0.8, 1.0]
}

# 動詞 (Verbs) - 動作の性質をベクトル化
verbs = {
    "seeks": [0.8, 0.5, 0.7, 0.8, 0.5],      # 探求する
    "fears": [0.9, 0.5, 0.1, 0.7, 0.6],      # 恐れる
    "consumes": [0.6, 0.8, 0.2, 0.9, 0.7],   # 飲み込む
    "creates": [0.5, 0.7, 0.9, 0.9, 0.8],    # 創り出す
    "destroys": [0.2, 0.8, 0.1, 1.0, 0.6],   # 破壊する
    "echoes": [0.1, 0.3, 0.5, 0.4, 0.3],     # 反響する
    "becomes": [0.5, 0.5, 0.6, 0.5, 0.9]     # ～になる
}

# 形容詞 (Adjectives) - 状態
adjectives = {
    "eternal": [0.0, 0.0, 0.7, 0.5, 0.8],    # 永遠の
    "empty": [0.0, 0.1, 0.3, 0.1, 0.5],      # 空っぽの
    "beautiful": [0.8, 0.5, 1.0, 0.7, 0.7],  # 美しい
    "broken": [0.4, 0.6, 0.2, 0.4, 0.6],     # 壊れた
    "dark": [0.2, 0.4, 0.1, 0.3, 0.7],       # 暗い
    "cold": [0.1, 0.5, 0.2, 0.2, 0.4]        # 冷たい
}

data_list = []

def add_words(word_dict, pos_tag):
    for k, v in word_dict.items():
        data_list.append({
            "__type": "Word",
            "name": k,
            "pos": pos_tag, # Part of Speech
            "vector": v
        })

add_words(nouns, "noun")
add_words(verbs, "verb")
add_words(adjectives, "adj")

with open("grammar_data.json", "w") as f:
    json.dump({"__type": "KnowledgeBase", "words": data_list}, f)

print(f"Generated grammar dictionary with {len(data_list)} words.")