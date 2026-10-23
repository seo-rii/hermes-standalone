"""Preserved original Python blocks inside the Cython NJD controller."""


def expand_features(mecab_features):
    expanded_features = []
    for mecab_feature in mecab_features:
        columns = mecab_feature.split(",")
        surface = columns[0]
        # 「〇七〇−〇〇二四」が未知語にまとまった場合も、桁と区切りを NJD の数詞処理へ渡す
        if (
            len(columns) == 8
            and len(surface) > 1
            and all(character in "０１２３４５６７８９〇零一二三四五六七八九−－ー‐‑‒–—-" for character in surface)
            and any(character in "０１２３４５６７８９零一二三四五六七八九" for character in surface)
        ):
            for character in surface:
                if character in "−－ー‐‑‒–—-":
                    expanded_features.append(
                        character + ",記号,一般,*,*,*,*," + character + ",、,、,0/0,*"
                    )
                else:
                    digit_index = "０１２３４５６７８９".find(character)
                    if digit_index < 0:
                        digit_index = "〇一二三四五六七八九".find(character) if character != "零" else 0
                    reading = ("ゼロ", "イチ", "ニ", "サン", "ヨン", "ゴ", "ロク", "ナナ", "ハチ", "キュー")[digit_index]
                    # 未知の「二」「五」は1拍で渡し、電話番号の「ニー」「ゴー」への伸長は NJD が決める
                    mora_size = (2, 2, 1, 2, 2, 1, 2, 2, 2, 2)[digit_index]
                    expanded_features.append(
                        character + ",名詞,数,*,*,*,*," + character + "," + reading + "," + reading + ",0/" + str(mora_size) + ",C3"
                    )
        else:
            expanded_features.append(mecab_feature)
    return expanded_features


def bound_features(feature):
    bounded_features = []
    has_number_boundary = False
    for index, current in enumerate(feature):
        # 「65 1032」の2つの数を別々に位取りし、後段では無音の境界ノードを取り除く
        if (
            index > 0
            and current["pos_group1"] == "数"
            and feature[index - 1]["pos_group1"] == "数"
            and current["chain_flag"] == 0
        ):
            boundary = current.copy()
            boundary.update(
                string="", orig="", read="*", pron="空白境界", pos_group1="一般",
                pos_group3="空白境界", acc=0, mora_size=0, chain_rule="*",
            )
            bounded_features.append(boundary)
            has_number_boundary = True
        bounded_features.append(current)
    return bounded_features, has_number_boundary


def unbound_features(nodes):
    feature = []
    after_boundary = False
    for current in nodes:
        if current["pos_group3"] == "空白境界":
            after_boundary = True
            continue
        if after_boundary is True:
            # 「二〇 万円」のように空白を挟んで「十」「百」「千」に「万」以上の位が続く場合は、空白のない「二〇万円」と同じく同じ数として句を続ける
            is_place_after_place = (
                len(feature) > 0
                and feature[-1]["string"] in ("十", "百", "千")
                and current["string"] in ("万", "億", "兆", "京")
            )
            current["chain_flag"] = 1 if is_place_after_place is True else 0
            after_boundary = False
        feature.append(current)
    return feature
