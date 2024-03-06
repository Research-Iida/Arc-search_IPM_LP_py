# Usage
---
## Run scripts
### `src/__main__.py`
```
poetry run python -m src -n {求解する問題数} -s {使用するソルバー} -c {config のセクション名} -sn {何番目の問題から解くか}
```

NETLIB のベンチマーク問題を読み取り, ソルバーにかけ, 必要な反復回数や実行可能性を記録
問題数は指定可能, 指定がなければ存在するすべて
ソルバーやconfig のセクションは指定可能だが, 指定しない場合は `src.__main__.py` に記載しているすべての組み合わせで計算実行

### 特定の問題を求解
```
poetry run python -m src.solve_netlib {instance name of Netlib} -s {solver name} -c {section name of config}
```

### Preprocess a instance of Netlib
```
poetry run python -m src.preprocess_netlib {instance name of Netlib} -c {section name of config}
```

---
## Set up
Before setting up,
you need to install `poetry`.
Please check [here](https://python-poetry.org/docs/).

### Prepare Netlib instances
1. Get SIF files from the Netlib site; such as [here](https://www.netlib.org/lp/data/index.html).
2. Place them in directory `data/raw/netlib`.

### Install packages
```
poetry install
```

## Testing programs
```
poetry run pytest
```

### Targets for test
`test_*.py` のファイルにある `test_*`という形式のメソッド.
テストを追加する際は上記の形式

### 処理の遅いテストを実行する場合
```
poetry run pytest -m slow
```

### すべてのテストを実行し, カバレッジを測定する
```
poetry run pytest -v --cov=src/
```

## 文書の build
### set up
```
poetry run sphinx-apidoc -f -o ./docs .
```

以下の条件に適合した場合、上記 set up コマンドをうつ必要がある

- 初めてローカルに `git clone` した
- フォルダの構成を変更した
- 新しく文書化対象のコードを追加した（`index.rst` も変更する必要あり）

### 最新のコードを文書に反映
```
poetry run sphinx-build -b singlehtml ./docs ./docs/_build
```

文書は `docs/_build/index.html` に格納される. webブラウザで開けばHTML形式で確認可能.


## プロファイリング
各スクリプトをプロファイルする方法は `src/profiler/profiler.py` 参照

### プロファイル結果を確認
#### コマンドライン上で手早く確認
```
poetry run python -m src.profile_result {対称のファイル名}
```

#### 細かく確認
```
poetry run cprofilev -f {対称のファイル名}
```

## クラス図の作成
```
poetry run pyreverse -o png src/
```

---

## 仮想環境関連
### パッケージインストール
```
poetry add <module> {--dev}
```

`pyproject.toml` に書き込みが行われるので, commit すること

### 現在の `pyprofect.toml` もとに更新
```
poetry update --no-dev
```

開発環境であれば `--no-dev` を消す

### 削除
```
poetry env remove .venv
```

install で原因不明のエラーが起きたときとかに再セットアップするため
