from __future__ import annotations
from contextlib import contextmanager
from .openjtalk import OpenJTalk
from ._context_reading import modify_context_reading
from ._kaomoji import exclude_kaomoji_from_numeral_processing, find_kaomoji_parts, silence_kaomoji
from ._loanword import restore_loanword_kana
from ._mapping import mark_user_dictionary_reading_protection
from ._unknown_kanji import read_unknown_kanji
from .utils import (
    fill_missing_feature_flags, modify_acc_after_chaining, modify_kanji_yomi,
    modify_old_province_yomi, normalize_iu, normalize_roman_numerals, normalize_text,
    normalize_unknown_itaiji, predict_nani_reading, process_odori_features,
    retreat_acc_nuc, revert_pron_to_read, suppress_unnatural_auxiliary_u_long_vowel,
)
from typing import Generic, TypeVar
from threading import Condition, Lock
_T = TypeVar("_T")

class _ReplaceableInstanceManager(Generic[_T]):
    """
    利用中の処理を完了させてから交換できる共有インスタンスを管理する。
    """

    def __init__(self, instance_factory: Callable[[], _T]) -> None:
        """
        インスタンスを遅延生成するマネージャーを初期化する。

        Args:
            instance_factory (Callable[[], _T]): 初回の借り出し時に呼ぶファクトリ
        """

        self._instance: _T | None = None
        self._instance_factory = instance_factory
        self._mutex = Lock()
        self._condition = Condition(self._mutex)
        self._active_leases = 0
        self._is_replacing = False

    @contextmanager
    def __call__(self) -> Generator[_T, None, None]:
        """
        共有インスタンスを借り出す。

        Yields:
            _T: 遅延生成または固定のシングルトン
        """

        with self._condition:
            # 交換開始後の呼び出しは、旧インスタンスを取得せず交換完了まで待機
            while self._is_replacing is True:
                self._condition.wait()
            instance = self._instance
            if instance is None:
                instance = self._instance_factory()
                self._instance = instance
            self._active_leases += 1
        try:
            yield instance
        finally:
            with self._condition:
                self._active_leases -= 1
                # 交換処理が待つのは最後の借り出しが返却される瞬間だけ
                if self._active_leases == 0:
                    self._condition.notify_all()

    def replace(self, instance: _T) -> None:
        """
        進行中の借り出しを待って共有インスタンスを交換する。

        Args:
            instance (_T): 次の借り出しから返すインスタンス

        NOTE:
            非リエントラント。借り出し中 (`with _global_jtalk()` 内) から呼んではいけない。
            `run_frontend()` の後処理中に `update_global_jtalk_with_user_dict()` や
            `unset_user_dict()` を呼ぶと、借り出し完了待ちでデッドロックする
        """

        with self._condition:
            # 複数の交換要求も順番に処理し、交換中は新規借り出しを止める
            while self._is_replacing is True:
                self._condition.wait()
            self._is_replacing = True
            try:
                while self._active_leases > 0:
                    self._condition.wait()
                self._instance = instance
            finally:
                self._is_replacing = False
                self._condition.notify_all()


_global_jtalk = _ReplaceableInstanceManager(OpenJTalk)

MULTI_READ_KANJI_LIST = [
    '風','何','観','方','出','時','上','下','君','手','嫌','表',
    '対','色','人','前','後','角','金','頭','筆','水','間','棚',
    # 以下、Wikipedia「同形異音語」からミスりそうな漢字を抜粋 (ただしこれらは NN 使わない限り完璧な判定は無理な気がする…)
    # Sudachi の方が不正確な '汚','通','臭','辛' は除外した
    # ref: https://ja.wikipedia.org/wiki/%E5%90%8C%E5%BD%A2%E7%95%B0%E9%9F%B3%E8%AA%9E
    '床','入','来','塗','怒','包','被','開','弾','捻','潜','支','抱','行','降','種','訳','糞',
    # 以下、Wikipedia「同形異音語」記事内「読み方が3つ以上ある同形異音語」より
    '空','性','体','等','生','止','堪','捩',
    # 以下、独自に追加
    '家','縁','労','中','高','低','気','要','退','面','色','主','術','直','片','緒','小','大','値',
    # 他にも日付（月・火・水・木・金・土・日）も入るが、当面は入れない (金を除く)
]
_MULTI_READ_KANJI_SET_EXCLUDING_NANI = frozenset(
    kanji for kanji in MULTI_READ_KANJI_LIST if kanji != "何"
)

@contextmanager
def _resolve_jtalk(*, jtalk: OpenJTalk | None) -> Generator[OpenJTalk, None, None]:
    """
    呼び出し元指定またはグローバルの `OpenJTalk` インスタンスを返す。

    Args:
        jtalk (OpenJTalk | None): 明示インスタンス。None なら `_global_jtalk()` を使う

    Yields:
        OpenJTalk: 処理中の借り出しが記録された OpenJTalk インスタンス

    NOTE:
        呼び出し元がインスタンスを渡した場合は、グローバルインスタンスの交換待機へ影響しない
    """

    # 明示インスタンスはグローバルの辞書交換と独立しているため、そのまま使用
    if jtalk is not None:
        yield jtalk
        return
    # グローバル利用中は辞書交換が同じインスタンスを途中で無効化しないよう借り出しを記録
    with _global_jtalk() as instance:
        yield instance


def modify_filler_accent(njd: list[NJDFeature]) -> list[NJDFeature]:
    """
    フィラーのアクセント核がモーラ数を超えている場合に平板へ直す。
    フィラーとその直後の名詞を別のアクセント句にする処理は、核の計算より前に `_apply_original_rule_before_chaining()` が行う。

    Args:
        njd (list[NJDFeature]): NJDNode 用 features

    Returns:
        list[NJDFeature]: フィラーのアクセント核補正を反映した features
    """

    for features in njd:
        if features["pos"] == "フィラー" and features["acc"] > features["mora_size"]:
            features["acc"] = 0

    return njd


def apply_postprocessing(
    text: str,
    njd_features: list[NJDFeature],
    *,
    run_marine: bool = False,
    use_vanilla: bool = False,
    use_sudachi_kanji_yomi: bool = True,
    predict_nani: bool = True,
    normalize_mode: NormalizeMode = "None",
    iu_pronunciation: IuPronunciation | None = None,
    use_read_as_pron: bool = False,
    revert_long_vowels: bool = False,
    revert_yotsugana: bool = False,
    jtalk: OpenJTalk | None = None,
) -> list[NJDFeature]:
    """
    加工されていない生の NJD features に後処理を適用する。
    run_frontend() / run_frontend_detailed() の通常経路・tsqyomi 適用経路の双方で呼び出される。

    Args:
        text (str): Unicode 日本語テキスト
        njd_features (list[NJDFeature]): NJDNode 用 features (pyopenjtalk.run_frontend() の戻り値)
        run_marine (bool): marine を用いたアクセント推定を行うか (デフォルト: False)
            有効にするには `pip install pyopenjtalk-plus[marine]` で marine をインストールする必要がある
        use_vanilla (bool): True の場合、辞書は pyopenjtalk-plus のまま、pyopenjtalk-plus 独自の後処理だけを省略し、OpenJTalk の素の NJDFeature をそのまま後段に流す
            後処理の効果を切り分けて測るための引数で、use_sudachi_kanji_yomi と predict_nani は後処理の一部として無効化される
            明示的に指定した run_marine (モデルの効果だけを測るため)、normalize_mode、iu_pronunciation と発音復元オプション (出力の書き方を揃えるため) は併用できる (デフォルト: False)
        use_sudachi_kanji_yomi (bool): True の場合、Sudachi による同形異音語の読み補正を行う
            use_vanilla が True の場合は後処理の一部として無効化される (デフォルト: True)
        predict_nani (bool): True の場合、ONNX モデルで単独形態素として出現した「何」の読みを推定する
            use_vanilla が True の場合は後処理の一部として無効化される (デフォルト: True)
        normalize_mode (NormalizeMode): 入力テキストに適用する Unicode 正規化方式
            `"NFC"` は結合文字を正規化し、`"NFKC"` は半角カナなどの互換文字も正規化する (デフォルト: `"None"`)
        iu_pronunciation (IuPronunciation | None): 「言う」や「という」などの定型表現に含まれる「イウ」を、どう発音するかの方式
            "Iu"、"Yuu"、"KanjiIu"、"KanjiYuu"、"YuuBase"、"KanjiYuuBase" のいずれかを指定する。None の場合は辞書の発音のままにし、指定した場合は use_vanilla の設定に関係なく適用される (デフォルト: None)
        use_read_as_pron (bool): True の場合、全ての発音を読みで丸ごと上書きする
            助詞「は」も「ハ」になるため、TTS 用途には適さない (デフォルト: False)
            このオプションが True の場合、revert_long_vowels / revert_yotsugana の指定に関係なく
            全ての pron が read で上書きされる
        revert_long_vowels (bool): True の場合、辞書が長音にした母音を、読みの仮名へ文字単位で戻す
            元の表記に「ー」がある語は、書かれた長音なので戻さない
            助詞 (は→ワ, へ→エ)、連濁、無声化の記号などの長音以外の違いは発音のまま残す
            iu_pronunciation も指定した場合は、「言う」の発音にはその方式を優先する
            (例: 「効果」コーカ → コウカ / 「人生」ジンセー → ジンセイ) (デフォルト: False)
        revert_yotsugana (bool): True の場合、辞書が「ズ」「ジ」にした四つ仮名を、読みの「ヅ」「ヂ」へ文字単位で戻す
            四つ仮名以外の違いは発音のまま残す
            (例: 「気づかず」キズカズ → キヅカズ / 「鼻血」ハナジ → ハナヂ) (デフォルト: False)
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う

    NOTE:
        run_marine / iu_pronunciation / 発音復元オプション (use_read_as_pron, revert_long_vowels, revert_yotsugana) は
        use_vanilla の設定に関係なく、明示的に指定された場合にのみ適用される
        後処理の規則 (踊り字の展開を含む) が読み (read) を書き換えた形態素には、`is_reading_rewritten` に True を付ける
        発音だけを書き換えた形態素 (外来語の表記の復元など) には付けない

    Returns:
        list[NJDFeature]: 後処理後の NJDNode 用 features
    """
    text = normalize_text(text, normalize_mode)
    # 公開関数の入口なので、印のない古い形の素性を渡されても動くよう、ここで欠けた印を補ってから後処理へ渡す
    fill_missing_feature_flags(njd_features)
    # 読み保護を指定したユーザー辞書の形態素は、後処理の前の読み・発音・モーラ数・アクセント核を控えておく
    protected_readings = [
        (index, feature["read"], feature["pron"], feature["mora_size"], feature["acc"])
        for index, feature in enumerate(njd_features)
        if feature["is_reading_protected"] is True
    ]
    # 後処理の規則が読みや発音を書き換えた形態素を、関数ごとに印を付けずに見分けられるよう、規則を当てる前の読みと発音を控えておく
    readings_before_rules = [(feature["read"], feature["pron"]) for feature in njd_features]
    if use_vanilla is False:
        # フィラーのアクセントは読み変更より先に補正する既存の処理順序を維持する
        njd_features = modify_filler_accent(njd_features)
        if predict_nani is True:
            njd_features = predict_nani_reading(njd_features)
        if use_sudachi_kanji_yomi is True:
            with _resolve_jtalk(jtalk=jtalk) as resolved_jtalk:
                njd_features = modify_kanji_yomi(
                    text,
                    njd_features,
                    _MULTI_READ_KANJI_SET_EXCLUDING_NANI,
                    jtalk=resolved_jtalk,
                )
        njd_features = suppress_unnatural_auxiliary_u_long_vowel(njd_features)
        njd_features = modify_context_reading(njd_features)
        njd_features = modify_old_province_yomi(njd_features)
        njd_features = restore_loanword_kana(njd_features)
        njd_features = read_unknown_kanji(njd_features, text)

        # 読みを書き換える補正が増えても個別に例外を書かずに済むよう、保護した形態素の読みはここでまとめて辞書の値に戻す
        ## 形態素の位置で戻すので、ここまでの後処理は形態素の数を変えてはならない (変える処理を足すなら文字位置で戻す形に変える)
        for index, read, pronunciation, mora_size, _acc in protected_readings:
            feature = njd_features[index]
            feature["read"] = read
            feature["pron"] = pronunciation
            feature["mora_size"] = mora_size

        # ここまでの後処理は形態素の数を変えないので、位置ごとに規則の前後の読みを比べて、読みを書き換えた形態素に印を付ける
        ## 印は呼び出し元が「辞書から読みが変わった語」を見分けるためのものなので、読みを変えずに発音だけを書き換えた形態素 (外来語の表記の復元など) には付けない
        ## 読み保護で辞書の値へ戻した形態素は、比べると前と同じなので印が付かない
        for feature, (read_before_rules, _) in zip(
            njd_features, readings_before_rules, strict=True
        ):
            if feature["read"] != read_before_rules:
                feature["is_reading_rewritten"] = True

        # NJD は書き換える前の発音で母音の無声化を判定したので、読みか発音を書き換えた形態素とその直前の形態素の末尾は、書き換えた後の発音で判定し直す
        ## 「博士課程」の「ハクシ」は「ク」が無声化し、「東高洲橋」の「ス」は連濁した「バシ」の前では無声化しない
        ## 読み保護した形態素は辞書の発音のまま残す
        rewritten_indices = [
            index
            for index, (feature, reading_before_rules) in enumerate(
                zip(njd_features, readings_before_rules, strict=True)
            )
            if (feature["read"], feature["pron"]) != reading_before_rules
            and feature["is_reading_protected"] is False
        ]
        if len(rewritten_indices) > 0:
            for index in rewritten_indices:
                njd_features[index]["pron"] = njd_features[index]["pron"].replace("’", "")
                previous = njd_features[index - 1] if index > 0 else None
                if previous is not None and previous["is_reading_protected"] is False:
                    previous["pron"] = previous["pron"].removesuffix("’")
            with _resolve_jtalk(jtalk=jtalk) as resolved_jtalk:
                revoiced_features = resolved_jtalk.set_unvoiced_vowel(njd_features)
            for index in rewritten_indices:
                for target_index in (index - 1, index):
                    if (
                        target_index >= 0
                        and njd_features[target_index]["is_reading_protected"] is False
                    ):
                        njd_features[target_index]["pron"] = revoiced_features[target_index]["pron"]

    # marine には読みとモーラ数を確定した形態素を渡し、補正後の発音に合ったアクセントを推定させる
    ## use_vanilla=True でも、明示的に指定された marine は適用する
    if run_marine:
        pred_njd_features = estimate_accent(njd_features)
        njd_features = preserve_noun_accent(njd_features, pred_njd_features)

    if use_vanilla is False:
        njd_features = retreat_acc_nuc(njd_features)
        njd_features = modify_acc_after_chaining(njd_features)

    # ユーザー辞書に登録したアクセント核は、marine とアクセントの補正のあとで登録した値に戻して最優先で守る
    ## アクセント句のつながり (chain_flag) は NJD が前後の文脈から決めるものなので守らず、形態素の核の位置だけを登録した値に保つ
    ## use_vanilla=True でも明示的に指定した marine は動くので、この戻しは use_vanilla に関係なく行う
    ## 形態素の数が変わる踊り字の展開より前に戻し、控えたときの位置との対応を保つ
    ## 後ろの名詞と複合語の1つのアクセント句に結合した語 (「ローンメール」「ローン会社」) の核は、NJD が結合した後の句の核なので戻さず、撥音の上の核を下げる補正などをそのまま当てる
    ## 助詞や助動詞が続くだけの句 (「ローンを」) は、登録した核のまま読む
    for index, _read, _pronunciation, _mora_size, acc in protected_readings:
        is_compounded_with_next_noun = (
            index + 1 < len(njd_features)
            and njd_features[index + 1]["chain_flag"] == 1
            and njd_features[index + 1]["pos"] == "名詞"
        )
        if is_compounded_with_next_noun is False:
            njd_features[index]["acc"] = acc

    if use_vanilla is False:
        # 踊り字の展開は形態素の数を変えるので、展開の前の形態素を参照ごと控え、展開後に新しく作られた形態素と、読みを書き換えられた形態素に印を付ける
        readings_before_odori = {
            id(feature): (feature, feature["read"]) for feature in njd_features
        }
        with _resolve_jtalk(jtalk=jtalk) as resolved_jtalk:
            njd_features = process_odori_features(njd_features, jtalk=resolved_jtalk)
        for feature in njd_features:
            reading_before_odori = readings_before_odori.get(id(feature))
            if reading_before_odori is None or reading_before_odori[1] != feature["read"]:
                feature["is_reading_rewritten"] = True
    # 発音復元は use_vanilla の設定に関係なく、明示的に指定された場合のみ独立して適用する
    if use_read_as_pron is True or revert_long_vowels is True or revert_yotsugana is True:
        with _resolve_jtalk(jtalk=jtalk) as resolved_jtalk:
            njd_features = revert_pron_to_read(
                njd_features,
                use_read_as_pron=use_read_as_pron,
                revert_long_vowels=revert_long_vowels,
                revert_yotsugana=revert_yotsugana,
                jtalk=resolved_jtalk,
            )
    # 「言う」の発音の方式も、明示的に指定された場合だけ適用する (発音の復元より後に置き、どの入口から呼んでも同じ結果を返す)
    if iu_pronunciation is not None:
        njd_features = normalize_iu(njd_features, iu_pronunciation)
    return njd_features


def run_frontend(
    text: str,
    *,
    run_marine: bool = False,
    use_vanilla: bool = False,
    use_tsqyomi: bool = False,
    use_sudachi_kanji_yomi: bool = True,
    predict_nani: bool = True,
    normalize_mode: NormalizeMode = "None",
    iu_pronunciation: IuPronunciation | None = None,
    use_read_as_pron: bool = False,
    revert_long_vowels: bool = False,
    revert_yotsugana: bool = False,
    jtalk: OpenJTalk | None = None,
) -> list[NJDFeature]:
    """
    OpenJTalk のテキスト処理フロントエンドを実行する。
    pyopenjtalk.run_frontend_detailed() のラッパー。NJD features のみを返す。

    Args:
        text (str): Unicode 日本語テキスト
        run_marine (bool): marine を用いたアクセント推定を行うか (デフォルト: False)
            有効にするには `pip install pyopenjtalk-plus[marine]` で marine をインストールする必要がある
        use_vanilla (bool): True の場合、辞書は pyopenjtalk-plus のまま、pyopenjtalk-plus 独自の後処理だけを省略し、
            OpenJTalk の素の NJDFeature をそのまま後段に流す
            後処理の効果を切り分けて測るための引数で、use_sudachi_kanji_yomi と predict_nani は後処理の一部として無効化される
            明示的に指定した use_tsqyomi と run_marine (モデルの効果だけを測るため)、normalize_mode、iu_pronunciation と
            出力の書式を揃えるための発音復元オプションはこのオプションと併用可能 (デフォルト: False)
        use_tsqyomi (bool): True の場合、ロード済みの tsqyomi で文脈に合う読み候補を選ぶ
            Sudachi と「何」モデルによる読み変更を省き、tsqyomi の選択を維持する (デフォルト: False)
        use_sudachi_kanji_yomi (bool): True の場合、Sudachi による同形異音語の読み補正を行う
            use_tsqyomi が True の場合は tsqyomi を優先して無効化し、use_vanilla が True の場合も後処理の一部として無効化される (デフォルト: True)
        predict_nani (bool): True の場合、ONNX モデルで単独形態素として出現した「何」の読みを推定する
            use_tsqyomi が True の場合は tsqyomi を優先して無効化し、use_vanilla が True の場合も後処理の一部として無効化される (デフォルト: True)
        normalize_mode (NormalizeMode): 入力テキストに適用する Unicode 正規化方式
            `"NFC"` は結合文字を正規化し、`"NFKC"` は半角カナなどの互換文字も正規化する (デフォルト: `"None"`)
        iu_pronunciation (IuPronunciation | None): 「言う」や「という」などの定型表現に含まれる「イウ」を、どう発音するかの方式
            "Iu"、"Yuu"、"KanjiIu"、"KanjiYuu"、"YuuBase"、"KanjiYuuBase" のいずれかを指定する
            None の場合は辞書の発音のままにし、指定した場合は use_vanilla の設定に関係なく適用される (デフォルト: None)
        use_read_as_pron (bool): True の場合、全ての発音を読みで丸ごと上書きする
            助詞「は」も「ハ」になるため、TTS 用途には適さない (デフォルト: False)
            このオプションが True の場合、revert_long_vowels / revert_yotsugana の指定に関係なく
            全ての pron が read で上書きされる
        revert_long_vowels (bool): True の場合、辞書が長音にした母音を、読みの仮名へ文字単位で戻す
            元の表記に「ー」がある語は、書かれた長音なので戻さない
            助詞 (は→ワ, へ→エ)、連濁、無声化の記号などの長音以外の違いは発音のまま残す
            iu_pronunciation も指定した場合は、「言う」の発音にはその方式を優先する
            (例: 「効果」コーカ → コウカ / 「人生」ジンセー → ジンセイ) (デフォルト: False)
        revert_yotsugana (bool): True の場合、辞書が「ズ」「ジ」にした四つ仮名を、読みの「ヅ」「ヂ」へ文字単位で戻す
            四つ仮名以外の違いは発音のまま残す
            (例: 「気づかず」キズカズ → キヅカズ / 「鼻血」ハナジ → ハナヂ) (デフォルト: False)
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う

    Returns:
        list[NJDFeature]: NJDNode 用 features
    """
    raw_text = text
    has_kaomoji = use_vanilla is False and bool(find_kaomoji_parts(raw_text))
    text = normalize_text(text, normalize_mode)
    with _resolve_jtalk(jtalk=jtalk) as inference_jtalk:
        if use_vanilla is False:
            # 辞書で読めない異体字だけを通用字へ置き換え、辞書で読める旧字体に固有の読みは残す
            text = normalize_unknown_itaiji(text, inference_jtalk=inference_jtalk)
            # 「Ⅰ章」「Ⅱ日」のローマ数字を算用数字へ置き換え、助数詞の音便と和語の読みを算用数字と同じにする
            # ユーザー辞書の「Ⅲ型」のような語の中のローマ数字は、登録した読みで読むため置き換えない
            text = normalize_roman_numerals(text, inference_jtalk=inference_jtalk)
        # 読み保護と顔文字の判定には辞書の出どころと語境界が必要なので、この場合だけ詳細形態素を作る
        # 普通の本文は詳細形態素を作らず、従来の軽い経路で解析する
        reading_protection = inference_jtalk.userdic_reading_protection
        needs_morphs = any(reading_protection) or has_kaomoji
        if use_tsqyomi is True:
            njd_features, morphs = _run_frontend_with_tsqyomi(
                text,
                include_morphs=needs_morphs,
                restore_unknown_katakana=use_vanilla is False,
                modify_numeral_reading=use_vanilla is False,
                raw_text=raw_text if has_kaomoji else None,
                normalize_mode=normalize_mode,
                jtalk=inference_jtalk,
            )
        elif needs_morphs is True:
            if has_kaomoji is True:
                mecab_features, morphs = inference_jtalk.run_mecab_detailed(text)
                mecab_features = exclude_kaomoji_from_numeral_processing(
                    raw_text, text, mecab_features, morphs, normalize_mode, jtalk=inference_jtalk
                )
                njd_features = inference_jtalk.run_njd_from_mecab(
                    mecab_features,
                    restore_unknown_katakana=True,
                    modify_numeral_reading=True,
                )
            else:
                njd_features, morphs = inference_jtalk.run_frontend_detailed(
                    text,
                    restore_unknown_katakana=use_vanilla is False,
                    modify_numeral_reading=use_vanilla is False,
                )
        else:
            njd_features = inference_jtalk.run_frontend(
                text,
                restore_unknown_katakana=use_vanilla is False,
                modify_numeral_reading=use_vanilla is False,
            )
            morphs = []
        if has_kaomoji:
            njd_features, morphs = silence_kaomoji(
                raw_text, text, njd_features, morphs, normalize_mode, jtalk=inference_jtalk
            )
        mark_user_dictionary_reading_protection(njd_features, morphs, reading_protection)

        # 読みとアクセントの後処理は apply_postprocessing() でまとめて行う
        ## tsqyomi を使うときは、tsqyomi が選んだ読みと競合する Sudachi の読み補正と「何」の読み推定だけを、下の引数で無効化して渡す
        ## 前後の語だけで読みが1つに決まる補正 (modify_context_reading() など) は tsqyomi と競合しないので、apply_postprocessing() の中で tsqyomi を使うときも適用する
        njd_features = apply_postprocessing(
            text,
            njd_features,
            run_marine=run_marine,
            use_vanilla=use_vanilla,
            use_sudachi_kanji_yomi=use_sudachi_kanji_yomi if use_tsqyomi is False else False,
            predict_nani=predict_nani if use_tsqyomi is False else False,
            normalize_mode="None",  # 既に normalize_text() で正規化されているため、再度正規化しない
            iu_pronunciation=iu_pronunciation,
            use_read_as_pron=use_read_as_pron,
            revert_long_vowels=revert_long_vowels,
            revert_yotsugana=revert_yotsugana,
            jtalk=inference_jtalk,
        )
    return njd_features


def run_frontend_detailed(
    text: str,
    *,
    run_marine: bool = False,
    use_vanilla: bool = False,
    use_tsqyomi: bool = False,
    use_sudachi_kanji_yomi: bool = True,
    predict_nani: bool = True,
    normalize_mode: NormalizeMode = "None",
    iu_pronunciation: IuPronunciation | None = None,
    use_read_as_pron: bool = False,
    revert_long_vowels: bool = False,
    revert_yotsugana: bool = False,
    jtalk: OpenJTalk | None = None,
) -> tuple[list[NJDFeature], list[MeCabMorph]]:
    """
    OpenJTalk のテキスト処理フロントエンドを MeCab 形態素詳細付きで実行する。
    MeCab で形態素解析を 1 回だけ実行し、NJD features と MeCab morphs を同時に返す。
    pyopenjtalk.run_frontend() と異なり、MeCab の未知語フラグ・コスト情報付きの morphs も取得できる。

    Args:
        text (str): Unicode 日本語テキスト
        run_marine (bool): marine を用いたアクセント推定を行うか (デフォルト: False)
            有効にするには `pip install pyopenjtalk-plus[marine]` で marine をインストールする必要がある
        use_vanilla (bool): True の場合、辞書は pyopenjtalk-plus のまま、pyopenjtalk-plus 独自の後処理だけを省略し、
            OpenJTalk の素の NJDFeature をそのまま後段に流す
            後処理の効果を切り分けて測るための引数で、use_sudachi_kanji_yomi と predict_nani は後処理の一部として無効化される
            明示的に指定した use_tsqyomi と run_marine (モデルの効果だけを測るため)、normalize_mode、iu_pronunciation と
            出力の書式を揃えるための発音復元オプションはこのオプションと併用可能 (デフォルト: False)
        use_tsqyomi (bool): True の場合、ロード済みの tsqyomi で文脈に合う読み候補を選ぶ
            Sudachi と「何」モデルによる読み変更を省き、tsqyomi の選択を維持する (デフォルト: False)
        use_sudachi_kanji_yomi (bool): True の場合、Sudachi による同形異音語の読み補正を行う
            use_tsqyomi が True の場合は tsqyomi を優先して無効化し、use_vanilla が True の場合も後処理の一部として無効化される (デフォルト: True)
        predict_nani (bool): True の場合、ONNX モデルで単独形態素として出現した「何」の読みを推定する
            use_tsqyomi が True の場合は tsqyomi を優先して無効化し、use_vanilla が True の場合も後処理の一部として無効化される (デフォルト: True)
        normalize_mode (NormalizeMode): 入力テキストに適用する Unicode 正規化方式
            `"NFC"` は結合文字を正規化し、`"NFKC"` は半角カナなどの互換文字も正規化する (デフォルト: `"None"`)
        iu_pronunciation (IuPronunciation | None): 「言う」や「という」などの定型表現に含まれる「イウ」を、どう発音するかの方式
            "Iu"、"Yuu"、"KanjiIu"、"KanjiYuu"、"YuuBase"、"KanjiYuuBase" のいずれかを指定する
            None の場合は辞書の発音のままにし、指定した場合は use_vanilla の設定に関係なく適用される (デフォルト: None)
        use_read_as_pron (bool): True の場合、全ての発音を読みで丸ごと上書きする
            助詞「は」も「ハ」になるため、TTS 用途には適さない (デフォルト: False)
            このオプションが True の場合、revert_long_vowels / revert_yotsugana の指定に関係なく
            全ての pron が read で上書きされる
        revert_long_vowels (bool): True の場合、辞書が長音にした母音を、読みの仮名へ文字単位で戻す
            元の表記に「ー」がある語は、書かれた長音なので戻さない
            助詞 (は→ワ, へ→エ)、連濁、無声化の記号などの長音以外の違いは発音のまま残す
            iu_pronunciation も指定した場合は、「言う」の発音にはその方式を優先する
            (例: 「効果」コーカ → コウカ / 「人生」ジンセー → ジンセイ) (デフォルト: False)
        revert_yotsugana (bool): True の場合、辞書が「ズ」「ジ」にした四つ仮名を、読みの「ヅ」「ヂ」へ文字単位で戻す
            四つ仮名以外の違いは発音のまま残す
            (例: 「気づかず」キズカズ → キヅカズ / 「鼻血」ハナジ → ハナヂ) (デフォルト: False)
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う

    Returns:
        tuple[list[NJDFeature], list[MeCabMorph]]: (NJD features, MeCab morphs)
            - NJD features: pyopenjtalk.run_frontend() と同一の結果が得られる
            - MeCab morphs: pyopenjtalk.run_mecab_detailed()[1] の結果を使い、認識した顔文字だけは読みの空の1要素へまとめる
    """
    raw_text = text
    has_kaomoji = use_vanilla is False and bool(find_kaomoji_parts(raw_text))
    text = normalize_text(text, normalize_mode)
    with _resolve_jtalk(jtalk=jtalk) as inference_jtalk:
        if use_vanilla is False:
            # 辞書で読めない異体字だけを通用字へ置き換え、辞書で読める旧字体に固有の読みは残す
            text = normalize_unknown_itaiji(text, inference_jtalk=inference_jtalk)
            # 「Ⅰ章」「Ⅱ日」のローマ数字を算用数字へ置き換え、助数詞の音便と和語の読みを算用数字と同じにする
            # ユーザー辞書の「Ⅲ型」のような語の中のローマ数字は、登録した読みで読むため置き換えない
            text = normalize_roman_numerals(text, inference_jtalk=inference_jtalk)
        if use_tsqyomi is True:
            njd_features, morphs = _run_frontend_with_tsqyomi(
                text,
                include_morphs=True,
                restore_unknown_katakana=use_vanilla is False,
                modify_numeral_reading=use_vanilla is False,
                raw_text=raw_text if has_kaomoji else None,
                normalize_mode=normalize_mode,
                jtalk=inference_jtalk,
            )
        elif has_kaomoji is True:
            mecab_features, morphs = inference_jtalk.run_mecab_detailed(text)
            mecab_features = exclude_kaomoji_from_numeral_processing(
                raw_text, text, mecab_features, morphs, normalize_mode, jtalk=inference_jtalk
            )
            njd_features = inference_jtalk.run_njd_from_mecab(
                mecab_features,
                restore_unknown_katakana=True,
                modify_numeral_reading=True,
            )
        else:
            njd_features, morphs = inference_jtalk.run_frontend_detailed(
                text,
                restore_unknown_katakana=use_vanilla is False,
                modify_numeral_reading=use_vanilla is False,
            )
        if use_vanilla is False:
            njd_features, morphs = silence_kaomoji(
                raw_text, text, njd_features, morphs, normalize_mode, jtalk=inference_jtalk
            )
        mark_user_dictionary_reading_protection(
            njd_features,
            morphs,
            inference_jtalk.userdic_reading_protection,
        )
        njd_features = apply_postprocessing(
            text,
            njd_features,
            run_marine=run_marine,
            use_vanilla=use_vanilla,
            use_sudachi_kanji_yomi=use_sudachi_kanji_yomi if use_tsqyomi is False else False,
            predict_nani=predict_nani if use_tsqyomi is False else False,
            normalize_mode="None",  # 既に normalize_text() で正規化されているため、再度正規化しない
            iu_pronunciation=iu_pronunciation,
            use_read_as_pron=use_read_as_pron,
            revert_long_vowels=revert_long_vowels,
            revert_yotsugana=revert_yotsugana,
            jtalk=inference_jtalk,
        )
    return njd_features, morphs


def make_label(njd_features: list[NJDFeature], *, jtalk: OpenJTalk | None = None) -> list[str]:
    """
    HTS 音声合成用のフルコンテキストラベルを返す。

    Args:
        njd_features (list[NJDFeature]): NJDNode 用 features (pyopenjtalk.run_frontend() の戻り値)
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う

    Returns:
        list[str]: フルコンテキストラベル文字列のリスト
    """
    with _resolve_jtalk(jtalk=jtalk) as resolved_jtalk:
        return resolved_jtalk.make_label(njd_features)


