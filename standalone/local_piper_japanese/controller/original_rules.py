from __future__ import annotations
from ._kana_utils import contains_hiragana, is_katakana_word
from ._known_symbols import KNOWN_SYMBOL_FEATURES
_NON_PAUSE_SYMBOLS = frozenset((
    "「", "」", "『", "』", "（", "）", "(", ")",
    "【", "】", "［", "］", "[", "]", "〈", "〉",
    "《", "》", "〔", "〕", "｛", "｝", "{", "}",
    "\"", "'", "”", "“", "’", "‘",
))

# MeCab が「！？！？」のような記号の並びを1つの未知語にまとめたとき、1文字ずつに分けるきっかけにする記号
# 疑問符か感嘆符を含む並びだけを分け、踊り字の「ヾ」などを含む顔文字はまとめたまま NJD へ渡す
_PAUSE_SYMBOLS_REQUIRING_EXPANSION = frozenset(("！", "？"))

# 数詞の直後で英字の記号として解析された1文字の単位を、助数詞として読むための読み・アクセント核・モーラ数・結合規則
# 値は既定辞書の助数詞の行と同じにして、「四千ｇ」と辞書の助数詞が選ばれた場合とで読みとアクセントをそろえる
## 1文字の単位の助数詞の行はコストが高く、「四千ｇだった」「八十ｍ先」では英字の記号に負けて単位が読まれない
## 辞書のコストを下げると、読点の後の「ｔ値」や数式の変数の「ｍ」まで助数詞になるので、数詞の直後に限って読み替える
_UNIT_LETTER_COUNTERS = {
    "ｇ": ("グラム", 1, 3, "C1"),
    "ｍ": ("メートル", 0, 4, "C2"),
    "ｔ": ("トン", 1, 2, "C3"),
    "ｌ": ("リットル", 0, 4, "C2"),
}
# 単位の英字の直後にあると、その英字を数式の変数とみなす演算子 (「2m+1」の「＋」など)
## 「1m60cm」の数字、「五百ｇ／タマネギ」の区切りの斜線、空白を挟んだ「七十ｇ　Ａ」の英字は単位の後にも続くので含めない
_UNIT_LETTER_FORMULA_OPERATORS = frozenset("＋−－＝×÷＊＜＞+-=*<>")


def _expand_symbol_feature(mecab_feature):
    """
    MeCab が1つの未知語にまとめた記号の並びを、1文字ずつの feature に分ける。
    疑問符か感嘆符を含まない feature は分けず、そのまま1件だけ返す。

    Args:
        mecab_feature (str): MeCab が返した surface 付き feature 文字列

    Returns:
        list[str]: 1文字ずつに分けた feature の列 (分けない場合は元の feature 1件)
    """

    columns = mecab_feature.split(",")
    surface = columns[0]
    is_unknown_symbol_chunk = (
        len(columns) == 8
        and len(surface) > 1
        and all(character.isalnum() is False for character in surface)
        and any(character in _PAUSE_SYMBOLS_REQUIRING_EXPANSION for character in surface)
    )
    if is_unknown_symbol_chunk is False:
        return [mecab_feature]

    expanded_features = []
    for character in surface:
        known_symbol = KNOWN_SYMBOL_FEATURES.get(character)
        if known_symbol is None:
            expanded_features.append(character + "," + ",".join(columns[1:]))
        else:
            expanded_features.append(character + "," + known_symbol[3])
    return expanded_features


def _mark_numeral_space_boundaries(features):
    """
    数字間の空白を後続の結合境界へ移し、空白自体は NJD 入力から除く。

    Args:
        features (list[str]): 空白を含む MeCab feature 列

    Returns:
        list[str]: 空白を除き、数字間の空白の直後を独立させた feature 列
    """
    result = []
    previous_was_number = False
    number_space = False
    for feature in features:
        columns = feature.split(",")
        if "記号,空白" in feature:
            # 「EF65 1032号機」の空白は無音のまま、次の数詞の位取りだけを独立させる
            number_space = previous_was_number
            continue
        is_number = len(columns) > 2 and columns[1:3] == ["名詞", "数"]
        if number_space is True and is_number is True:
            if len(columns) == 12:
                columns.append("0")
            elif len(columns) > 12:
                columns[12] = "0"
            feature = ",".join(columns)
        result.append(feature)
        previous_was_number = is_number
        number_space = False
    return result


def _restore_unknown_katakana_features(
    njd_features: list[NJDFeature],
    mecab_features: list[str],
) -> list[NJDFeature]:
    """
    `njd_set_pronunciation()` がフィラーへ変えた未知のカタカナ語を MeCab の品詞へ戻す。
    読みを持たない未知語だけを対象にするため、辞書に登録された本来のフィラーは維持する。

    Args:
        njd_features (list[NJDFeature]): `njd_set_pronunciation()` 適用後の NJD features
        mecab_features (list[str]): NJD 変換前の MeCab feature 文字列

    Returns:
        list[NJDFeature]: 未知のカタカナ語の品詞とアクセントを復元した NJD features
    """
    unknown_pos_by_surface: dict[str, list[tuple[str, str, str, str]]] = {}

    # 既知語は読み以降を含む12列以上になるため、短い未知語 feature の品詞だけを保存する
    for mecab_feature in mecab_features:
        fields = mecab_feature.split(",")
        if len(fields) < 5 or len(fields) >= 12:
            continue
        unknown_pos_by_surface.setdefault(fields[0], []).append(
            (fields[1], fields[2], fields[3], fields[4])
        )

    for feature in njd_features:
        # 英字未知語のフィラー化は英語読み補正が利用するため、全カタカナの表層形へ限定する
        surface = feature["string"]
        if (
            feature["pos"] != "フィラー"
            or is_katakana_word(surface) is False
            or surface not in unknown_pos_by_surface
        ):
            continue

        # 同じ表層形が複数回現れても後方の品詞で上書きせず、MeCab の出現順に対応させる
        pos, pos_group1, pos_group2, pos_group3 = unknown_pos_by_surface[surface].pop(0)
        if len(unknown_pos_by_surface[surface]) == 0:
            del unknown_pos_by_surface[surface]
        feature["pos"] = pos
        feature["pos_group1"] = pos_group1
        feature["pos_group2"] = pos_group2
        feature["pos_group3"] = pos_group3

        # 外来語の核を後ろから3モーラ目へ置き、特殊拍に当たる場合は1つ前へ移す
        moras: list[str] = []
        for char in feature["pron"]:
            if char in "ャュョァィゥェォ" and len(moras) > 0:
                moras[-1] += char
            else:
                moras.append(char)
        if len(moras) <= 3:
            feature["acc"] = 1
        else:
            accent_index = len(moras) - 3
            while accent_index > 0 and moras[accent_index] in ("ー", "ン", "ッ"):
                accent_index -= 1
            feature["acc"] = accent_index + 1

    return njd_features


def _apply_original_rule_before_chaining(
    njd_features: list[NJDFeature],
    modify_numeral_reading: bool = True,
) -> list[NJDFeature]:
    """
    NJD features に chaining 前の独自ルールを適用する。内部用。
    サ変接続・接頭語・動詞連続・連用形・助動詞などのアクセント結合規則を適用する。

    Args:
        njd_features (list[NJDFeature]): NJDNode 用 features 。インプレースで更新される
        modify_numeral_reading (bool): True の場合、分数の分母の「分」を「ブン」、2つ以上続く「〇」を「マル」と読み、数詞の直後の「ｇ」「ｍ」などの単位の英字を助数詞として読む (デフォルト: True)

    Returns:
        list[NJDFeature]: 更新後の njd_features（同一オブジェクト）
    """
    numeral_zero_indices: set[int] = set()
    digit_characters = "０１２３４５６７８９〇零一二三四五六七八九"
    index = 0
    while index < len(njd_features):
        start = index
        while (
            index < len(njd_features)
            and njd_features[index]["string"] in digit_characters
            and len(njd_features[index]["string"]) == 1
        ):
            index += 1
        if index == start:
            index += 1
            continue
        # 「一〇〇一号室」の〇は番号の桁なので、伏字の「〇〇町」と区別して数詞へ渡す
        is_number = any(feature["string"] != "〇" for feature in njd_features[start:index])
        if index < len(njd_features):
            following = njd_features[index]
            is_number = is_number or following["pos_group2"] == "助数詞"
            # 「〇〇〇-一二三四-五六七八」の先頭も、区切りの後に数字が続く番号として扱う
            if following["string"] in "−－ー‐‑‒–—-" and index + 1 < len(njd_features):
                is_number = is_number or njd_features[index + 1]["string"] in digit_characters
        # 「〇一二〇-九九九-〇〇〇」の最後の組と、「一,〇〇〇」の3桁区切りの後の組も、数字の後の区切りに続く数として扱う
        if start >= 2 and njd_features[start - 1]["string"] in "−－ー‐‑‒–—-，":
            is_number = is_number or njd_features[start - 2]["string"] in digit_characters
        # 「〇円」の単独の〇は数値のレーを保ち、2桁以上の番号に含まれる〇だけをゼロへ戻す
        if is_number is True and index - start > 1:
            for zero_index in range(start, index):
                zero = njd_features[zero_index]
                if zero["string"] == "〇":
                    numeral_zero_indices.add(zero_index)
                    zero["pos"] = "名詞"
                    zero["pos_group1"] = "数"
                    zero["read"] = zero["pron"] = "ゼロ"
                    zero["mora_size"] = 2

    for i, njd in enumerate(njd_features[:-1]):
        # 名詞の後ろで新しい語を作る「不足」は連濁した「ブソク」と読む
        # 「情報が不足する」のように単独で用いる場合は、辞書本来の「フソク」という読みを変更しない
        next_njd = njd_features[i + 1]
        # ナイ形に続く「の」「と」は、助動詞に付く結合規則を明示して否定形の核を保つ
        ## 助動詞用の規則がないと結合規則の末尾の F2 が適用され、平板型の「知らないのか」が「シラナイ＼ノカ」になる
        if (
            njd["pos"] == "助動詞"
            and njd["orig"] == "ない"
            and (
                next_njd["pos"] == "助詞"
                or (next_njd["pos"] == "名詞" and next_njd["pos_group1"] == "非自立")
            )
            and next_njd["orig"] in ("の", "と")
        ):
            next_njd["chain_rule"] = "助動詞%F1/" + "/".join(
                rule for rule in next_njd["chain_rule"].split("/")
                if not rule.startswith("助動詞%")
            )
        # 放置を意味する「放っておく」「放っとく」の「放っ」は、読み補正で平板型の「放る」の「ホーッ」に替えるため、結合の前に核を0にする
        ## 結合した後で核だけを書き換えると、「放っておきません」「放っておきなさい」で後続の語が置く核まで消えてしまう
        if njd["string"] == "放っ" and (
            (
                next_njd["string"] == "て"
                and i + 2 < len(njd_features)
                and njd_features[i + 2]["orig"] == "おく"
                and njd_features[i + 2]["pos"] == "動詞"
            )
            or (next_njd["orig"] == "とく" and next_njd["pos"] == "動詞")
        ):
            njd["acc"] = 0
        # 動詞の意向形の「う」は、付属語の「う」が核を決める型で、「守ろう」「盛ろう」を「マモロ＼ー」「モロ＼ー」と「う」の直前の拍の後で下げる
        ## 推量の「だろう」の「う」は直前が助動詞なので変えず、「かかるだろう」を1つの句の「カカルダロ＼ー」にしない
        ## 平板の動詞の意向形に「と」と「する」「思う」が続く「盛ろうとする」は、平板のまま「モロートスル」と読むので変えない
        if (
            njd["pos"] == "動詞"
            and njd["cform"] == "未然ウ接続"
            and next_njd["pos"] == "助動詞"
            and next_njd["orig"] == "う"
        ):
            is_flat_before_to_suru = (
                njd["acc"] == 0
                and i + 3 < len(njd_features)
                and njd_features[i + 2]["string"] == "と"
                and njd_features[i + 3]["orig"] in ("する", "思う")
            )
            if is_flat_before_to_suru is False:
                next_njd["chain_rule"] = "動詞%F4@0/特殊助動詞%F2@0"
        # 平板の動詞に使役の「せ」「させ」と「ない」が続く「聞かせない」「歌わせない」は、平板の「キカセナイ」「ウタワセナイ」と読む
        ## MeCab は「聞か」「せ」「ない」に分けるので、動詞の連続の規則が「せ」の核を採って「キカセ＼ナイ」と読んでいた
        ## 「せ」の核をなくすと、後ろの動詞の連続の規則が平板の C4 で結合し、「ない」は直前の動詞の型を保つ F1 で続く
        ## 起伏の動詞の「書かせない」は、今までどおり「カカセ＼ナイ」と読む
        if (
            njd["pos"] == "動詞"
            and njd["pos_group1"] == "自立"
            and njd["acc"] == 0
            and next_njd["pos"] == "動詞"
            and next_njd["pos_group1"] == "接尾"
            and next_njd["orig"] in ("せる", "させる")
            and i + 2 < len(njd_features)
            and njd_features[i + 2]["pos"] == "助動詞"
            and njd_features[i + 2]["orig"] == "ない"
        ):
            next_njd["acc"] = 0
            njd_features[i + 2]["chain_rule"] = "/".join(
                "動詞%F1" if rule.startswith("動詞%") else rule
                for rule in njd_features[i + 2]["chain_rule"].split("/")
            )
        # 平板の動詞の使役の基本形「聞かせる」「歌わせる」も、平板の「キカセル」「ウタワセル」と読む
        ## 「せる」の核を動詞の連続の規則が採って「キカセ＼ル」と読んでいたので、ナイ形と同じく「せる」の核をなくす
        ## 起伏の動詞の「書かせる」は「カカセ＼ル」のまま読み、「聞かせます」「聞かせれば」は付属語の規則で核を置く
        if (
            njd["pos"] == "動詞"
            and njd["pos_group1"] == "自立"
            and njd["acc"] == 0
            and next_njd["pos"] == "動詞"
            and next_njd["pos_group1"] == "接尾"
            and next_njd["orig"] in ("せる", "させる")
            and next_njd["cform"] == "基本形"
        ):
            next_njd["acc"] = 0
        # 一般名詞の「湖」を接尾辞として読む場合、全体が4モーラ以上なら前部末型で結合する
        ## 「ミズウミ」は後処理で1モーラの「コ」になるので、その長さで判定する
        ## 短い湖名は辞書の結合規則を使い、語全体で登録された湖名の核も辞書に任せる
        if njd["pos"] == "名詞" and next_njd["string"] == "湖" and next_njd["pron"] == "ミズウミ":
            preceding_mora_size = 0
            for preceding_index in range(i, -1, -1):
                preceding_njd = njd_features[preceding_index]
                if preceding_njd["pos"] != "名詞":
                    break
                preceding_mora_size += preceding_njd["mora_size"]
                if preceding_njd["chain_flag"] == 0:
                    break
            if preceding_mora_size + 1 >= 4:
                next_njd["chain_rule"] = "C3"

        if (
            njd["pos"] == "名詞"
            and next_njd["string"] == "不足"
            and next_njd["pron"] == "フソク"
        ):
            next_njd["read"] = "ブソク"
            next_njd["pron"] = "ブソク"

        # 分母を表す「数値 + 分 + の + 数値」だけ、時間量の「フン」・「プン」や割合の「ブ」と区別して「ブン」と読む
        # 漢数字を含む長単位候補と、算用数字の後ろへ付く助数詞のどちらにも同じ文法条件を適用する
        is_fraction_denominator = False
        if i + 2 < len(njd_features) and next_njd["string"] == "の":
            following_njd = njd_features[i + 2]
            # 「3分の何」「何分の何」の分子の「何」は数を問う語なので、数詞の「ナン」と読み、分母を「ブン」にする
            ## 「5分の何もない時間」のように「も」が続く「何」は、数でない「ナニ」の用法なので除く
            ## 「千分の何がし」は MeCab が「何」「が」「し」に分けるが、数でない「ナニガシ」なので除く
            if (
                modify_numeral_reading is True
                and njd["string"].endswith("分")
                and (
                    (
                        len(njd["string"]) > 1
                        and all(character in digit_characters + "十百千万何" for character in njd["string"][:-1])
                    )
                    or (i > 0 and njd_features[i - 1]["pos_group1"] == "数")
                )
                and following_njd["string"] == "何"
                and following_njd["pos_group1"] != "数"
                and (i + 3 >= len(njd_features) or njd_features[i + 3]["string"] != "も")
                and not (
                    i + 4 < len(njd_features)
                    and njd_features[i + 3]["string"] == "が"
                    and njd_features[i + 4]["string"] == "し"
                )
            ):
                following_njd.update(
                    pos="名詞", pos_group1="数", pos_group2="*", pos_group3="*",
                    read="ナン", pron="ナン", acc=1, mora_size=2, chain_rule="C3",
                )
            is_fraction_denominator = following_njd["pos_group1"] == "数"
        if (
            modify_numeral_reading is True
            and is_fraction_denominator is True
            and njd["string"].endswith("分")
        ):
            if njd["pron"].endswith(("フン", "プン")):
                njd["read"] = njd["read"][:-2] + "ブン"
                njd["pron"] = njd["pron"][:-2] + "ブン"
            elif njd["pron"].endswith("ブ"):
                njd["read"] += "ン"
                njd["pron"] += "ン"
            # 読みの長さを変更した場合も、後続のアクセント句が参照するモーラ数を同じ値に保つ
            njd["mora_size"] = sum(
                character not in "ャュョァィゥェォ’" for character in njd["pron"]
            )
            # 分数は「万分の一」(「マンブンノイチ＼」) と同じく、分母を平板にして分子の数詞の核で1つのアクセント句に読む
            ## 数詞に続く助数詞の「分」は、数詞との結合後の句を平板にする F5 にし、「三分」のような1語の分母は核をなくす
            if njd["string"] == "分" and i > 0 and njd_features[i - 1]["pos_group1"] == "数":
                njd["chain_rule"] = "F5"
            else:
                njd["acc"] = 0
            next_njd["chain_flag"] = 1
            njd_features[i + 2]["chain_flag"] = 1
            njd_features[i + 2]["chain_rule"] = "C1"

        # 算用数字が別形態素になった分数では、数詞と「の」に挟まれた助数詞の「分」を「ブン」へ変える
        if (
            modify_numeral_reading is True
            and i > 0
            and i + 2 < len(njd_features)
            and njd["string"] == "分"
            and njd_features[i - 1]["pos_group1"] == "数"
            and next_njd["string"] == "の"
            and njd_features[i + 2]["pos_group1"] == "数"
        ):
            njd["read"] = "ブン"
            njd["pron"] = "ブン"
            njd["mora_size"] = 2

        # 数詞の直後で英字の記号として解析された「ｇ」「ｍ」などの単位を、辞書の助数詞と同じ読みとアクセントに置き換える
        ## 助数詞に変えておくと、後段の NJD の数詞処理が「サンゼングラム」のように数詞と1つのアクセント句にまとめる
        ## 「2m+1」「x＝2ｔ×3」のように英字の直後に演算子が続く場合は、数式の変数として英字のまま読む
        ## 読み保護付きのユーザー辞書で記号として登録した英字は、登録した読みとアクセント核を使う
        following_string = njd_features[i + 2]["string"] if i + 2 < len(njd_features) else ""
        if (
            modify_numeral_reading is True
            and njd["pos_group1"] == "数"
            and next_njd["pos"] == "記号"
            and next_njd["string"] in _UNIT_LETTER_COUNTERS
            and next_njd["pos_group3"] != "読み保護"
            and following_string[:1] not in _UNIT_LETTER_FORMULA_OPERATORS
        ):
            read, acc, mora_size, chain_rule = _UNIT_LETTER_COUNTERS[next_njd["string"]]
            next_njd.update(
                pos="名詞", pos_group1="接尾", pos_group2="助数詞", pos_group3="*",
                read=read, pron=read, acc=acc, mora_size=mora_size, chain_rule=chain_rule,
            )

        # 番号の桁に含まれない「〇〇町」のような伏字だけ、NJD の数字変換へ渡さずマルと読む
        # 単独の「〇円」などは数詞のまま残し、従来の零読みを維持する
        # アクセント句の核は NJD の結合に任せる (「〇〇です」は「マル＼マルデス」、「〇〇町」は「マルマル＼マチ」になる)
        ## Haqumei は NJD の処理が終わったあとで核を1に書き換えているが、それでは「マ＼ルマル」になってしまうので取り入れていない
        if (
            modify_numeral_reading is True
            and njd["string"] == "〇"
            and next_njd["string"] == "〇"
            and i not in numeral_zero_indices
            and i + 1 not in numeral_zero_indices
        ):
            for placeholder_njd in (njd, next_njd):
                # MeCab は後ろに何も続かない「〇〇」を記号として返すので、品詞も名詞にそろえて読ませる
                placeholder_njd["pos"] = "名詞"
                placeholder_njd["pos_group1"] = "一般"
                placeholder_njd["read"] = "マル"
                placeholder_njd["pron"] = "マル"
                # 単独の「マル」は平板
                placeholder_njd["acc"] = 0
                placeholder_njd["mora_size"] = 2

        # 接尾辞「球」は漢語・外来語との生産的な結合をキュウとし、送り仮名を持つ和語だけ連濁させる
        # 漢字とひらがなの混在を条件にすることで、ひらがなだけの生産的な造語までダマへ変えない
        if (
            next_njd["string"] == "球"
            and next_njd["pos"] == "名詞"
            and next_njd["pos_group1"] == "接尾"
            and next_njd["pron"] == "キュー"
            and any("一" <= character <= "鿿" for character in njd["string"])
            and contains_hiragana(njd["string"]) is True
        ):
            next_njd["read"] = "ダマ"
            next_njd["pron"] = "ダマ"
            next_njd["acc"] = 1
            next_njd["mora_size"] = 2
            next_njd["chain_rule"] = "C4"

        # サ変動詞(スル)の前にサ変接続や名詞が来た場合は、一つのアクセント句に纏める
        if (njd["pos_group1"] in ["サ変接続", "格助詞", "接続助詞"] or (njd["pos"] == "名詞" and njd["pos_group1"] == "一般") or njd["pos"] == "副詞" ) and njd_features[i+1]["ctype"] == "サ変・スル":
            njd_features[i+1]["chain_flag"] = 1
        # ご遠慮、ご配慮のような接頭語がつく場合にその後に続く単語の結合則を変更する
        if (njd["string"] in ["お","御","ご"] and njd["chain_rule"] == "P1"):
            if njd_features[i+1]["acc"] == 0 or njd_features[i+1]["acc"] == njd_features[i+1]["mora_size"]:
                njd_features[i+1]["chain_rule"] = "C4"
                njd_features[i+1]["acc"] = 0
            else:
                njd_features[i+1]["chain_rule"] = "C1"
        # 動詞(自立)が連続する場合(ex 推し量る、刺し貫く)、後ろの動詞のアクセント核が採用される
        if njd["pos"] == "動詞"  and njd_features[i+1]["pos"] == "動詞" :
            njd_features[i+1]["chain_rule"] = "C1" if njd_features[i+1]["acc"] != 0 else "C4"
        # 連用形のアクセント核の登録を修正する
        if njd["cform"] in ["連用形","連用タ接続","連用ゴザイ接続","連用テ接続"] and njd["acc"] == njd["mora_size"] > 1 :
            njd["acc"] -= 1
        # 「らる、られる」＋「た」の組み合わせで「た」の助動詞/F2@0を上書きしてアクセントを下げないようにする
        if njd["orig"] in ["れる", "られる","せる", "させる","ちゃう"]  and njd_features[i+1]["string"] in ["た"] :
            njd_features[i+1]["chain_rule"] = "F2@1"

        # 形容詞＋「なる、する」は一つのアクセント句に纏める
        if njd["pos"] == "形容詞" and njd_features[i+1]["orig"] in ["なる", "する"]:
            njd_features[i+1]["chain_flag"] = 1

    # フィラーは前の語の句につなげず、フィラー自身と直後の名詞からそれぞれ新しいアクセント句を始める
    ## 句の切れ目を NJD が核を計算した後で変えると、前の句の核が句の外に残る (「東京タワーえーとスカイツリー」が「タワー」の後で下がらない) ので、核の計算の前に決める
    for i, njd in enumerate(njd_features):
        if njd["pos"] != "フィラー":
            continue
        njd["chain_flag"] = 0
        if i + 1 < len(njd_features) and njd_features[i + 1]["pos"] == "名詞":
            njd_features[i + 1]["chain_flag"] = 0

    if modify_numeral_reading is True:
        # 「当大学病院」「本研究計画」は、接頭辞の後ろで句を分けてから複合語の核を計算する
        ## 計算後に句だけを分けると、接頭辞につながった間に失われた後ろの句の核を復元できない
        from .utils import split_prefix_accent_phrase
        split_prefix_accent_phrase(njd_features)

    return njd_features
