"""slack への通知を行う Sub module."""
import abc
import traceback

from slack_sdk.webhook import WebhookClient
import ssl
import certifi

from ..utils.file_util import exists_file
from ..utils.config_utils import read_config


class SlackInterface(abc.ABC):
    """slack の通知をするための Super class
    メソッドの定義を行う
    """
    _slack: WebhookClient
    _mention_id: str

    @abc.abstractmethod
    def notify(self, text: str) -> None:
        """通知

        Args:
            text : 表示するテキスト
        """
        pass

    @abc.abstractmethod
    def notify_mentioned(self, text: str) -> None:
        """メンションして通知. モバイルとかで通知が行くようにする

        Args:
            text : 表示するテキスト
        """
        pass

    @abc.abstractmethod
    def notify_error(self) -> None:
        """問題が発生したときに"問題発生"と通知する"""
        pass


class SlackImplementation(SlackInterface):
    """Slack Interface への実装"""
    def __init__(self, url: str, mentioned_id: str):
        """`SLACK_API_URL` を元にAPIを設定"""
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        self._slack = WebhookClient(url, ssl=ssl_context)

        self._mention_id = mentioned_id

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
        self.notify(text=f"<@{self._mention_id}> {text}")

    def notify_error(self) -> None:
        """問題が発生したときに"問題発生"と通知する"""
        self.notify_mentioned(f":no_entry_sign: Ocurred error! ```{traceback.format_exc()}```")


class EmptySlack(SlackInterface):
    """設定がされなかった場合に適用される None Class
    メソッド呼び出しが行われても何もしない
    """
    def notify(self, text: str) -> None:
        pass

    def notify_mentioned(self, text: str) -> None:
        pass

    def notify_error(self) -> None:
        pass


def get_slack_api(filename: str = "config/config_slack.ini") -> SlackInterface:
    if not exists_file(filename):
        return EmptySlack()

    config_slack = read_config(filename)
    return SlackImplementation(config_slack.get("SLACK_API_URL"), config_slack.get("MENTIONED_ID"))
