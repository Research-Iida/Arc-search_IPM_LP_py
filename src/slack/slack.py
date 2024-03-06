"""slack への通知を行う Sub module."""
from typing import Union
import traceback

from slack_sdk.webhook import WebhookClient
import ssl
import certifi


class Slack:
    """飯田の slack への通知を行う"""
    def __init__(self):
        """`SLACK_API_URL` を元にAPIを設定"""
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        url = "https://hooks.slack.com/services/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"
        self._slack = WebhookClient(url, ssl=ssl_context)

    def notify(self, text: str) -> None:
        """通知

        Args:
            text : 表示するテキスト
        """
        self._slack.send(text=text)

    def notify_mentioned(self, text: str) -> None:
        """メンションして通知. モバイルとかで通知が行くようにする

        Args:
            text : 表示するテキスト
        """
        user_id = "ideruf96"
        self.notify(text=f"<@{user_id}> {text}")

    def notify_with_runtime(self, text: str, runtime: Union[str, float]):
        """実行秒数を通知する

        Args:
            text : 表示するテキスト
            runtime : 実行時間（秒）
        """
        self.notify(text + ", 実行時間 : " + str(runtime) + " seconds.")

    def notify_error(self) -> None:
        """問題が発生したときに"問題発生"と通知する"""
        self.notify_mentioned(f":no_entry_sign: エラーが発生しました. ```{traceback.format_exc()}```")
