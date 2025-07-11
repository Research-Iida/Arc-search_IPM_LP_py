# Setting slack notification
`config/config_slack.ini` というファイルを追加し,
通知先の API URL とメンションする際の ID を設定する.
もしくは `get_slack_api` の引数 `filename` を置きたい場所に修正すればok.
設定しなくても問題ない（その場合は通知をしない）.
