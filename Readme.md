# Usage
---
## Run scripts
### `src/__main__.py`
```sh
poetry run python -m src -n {求解する問題数} -s {使用するソルバー} -c {config のセクション名} -sn {何番目の問題から解くか} {-k: Kennington の大きいサイズの問題を解く場合つける}
```

NETLIB のベンチマーク問題を読み取り, ソルバーにかけ, 必要な反復回数や実行可能性を記録
問題数は指定可能, 指定がなければ存在するすべて
ソルバーやconfig のセクションは指定可能だが, 指定しない場合は `src.__main__.py` に記載しているすべての組み合わせで計算実行

### 特定の問題を求解
```sh
poetry run python -m src.solve_netlib {instance name of Netlib} -s {solver name} -c {section name of config}
```

### Preprocess a instance of Netlib
```sh
poetry run python -m src.preprocess_netlib {instance name of Netlib} -c {section name of config}
```

---
## Set up
1. Install `poetry`
2. Prepare Netlib instances
3. プロジェクトのディレクトリ配下に仮想環境を作成するようにする
4. Install packages
5. `pre-commit` の設定

### Install `poetry`
Before setting up,
you need to install `poetry`.
Please check [here](https://python-poetry.org/docs/).

### Prepare Netlib instances
1. Get SIF files from the Netlib site; such as [here](https://www.netlib.org/lp/data/index.html).
2. Place them in directory `data/raw/netlib`.

### プロジェクトのディレクトリ配下に仮想環境を作成するようにする
```sh
poetry config virtualenvs.in-project true
```

プロジェクトの root 配下に `.venv` ディレクトリが作成され, 仮想環境にかかわるファイルはそこへ格納されるようになる

### Install packages
```sh
poetry install
```

### `pre-commit` の設定
```sh
poetry run pre-commit install
```

### Advanced
#### Setting slack notification
`config` ディレクトリ配下に `config_slack.ini` というファイルを追加し,
通知先の API URL とメンションする際の ID を設定する.
設定しなくても問題ない（その場合は通知をしない）.

---
## Testing programs
```sh
poetry run pytest
```

### Targets for test
`test` ディレクトリ中の `test_*.py` のファイルにある `test_*`という形式のメソッド.
テストを追加する際は上記の形式

### 処理の遅いテストを実行する場合
```sh
poetry run pytest -m slow
```

### すべてのテストを実行し, カバレッジを測定する
```sh
poetry run pytest -v --cov=src/ -m ' '
```

---
## プロファイリング
各スクリプトをプロファイルする方法は `src/profiler/profiler.py` 参照

### プロファイル結果を確認
#### コマンドライン上で手早く確認
```sh
poetry run python -m src.profile_result {対称のファイル名}
```

#### 細かく確認
```sh
poetry run cprofilev -f {対称のファイル名}
```

---
## Building documents
```sh
poetry run pdoc src --html -o docs --force
```

---
## クラス図の作成
```sh
poetry run pyreverse -o png src/
```

---
## 仮想環境管理
### パッケージ追加
```sh
poetry add <module> {--group dev or local}
```

`pyproject.toml` に書き込みが行われるので, commit すること

### 現在の `pyprofect.toml` もとに更新
```sh
poetry update
```
