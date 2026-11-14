"""各资讯源 Provider 实现。"""

from .akshare_cn import AkshareCnProvider
from .binance_deriv import BinanceDerivProvider
from .cryptopanic import CryptoPanicProvider
from .fear_greed import FearGreedProvider
from .finnhub import FinnhubProvider
from .fred import FredProvider
from .gdelt import GdeltProvider
from .http_news import (
    GithubReleasesProvider,
    GNewsProvider,
    GuardianProvider,
    MediastackProvider,
    NewsApiProvider,
)
from .reddit import RedditProvider
from .rss import RssProvider

__all__ = [
    "AkshareCnProvider",
    "BinanceDerivProvider",
    "CryptoPanicProvider",
    "FearGreedProvider",
    "FinnhubProvider",
    "FredProvider",
    "GNewsProvider",
    "GdeltProvider",
    "GithubReleasesProvider",
    "GuardianProvider",
    "MediastackProvider",
    "NewsApiProvider",
    "RedditProvider",
    "RssProvider",
]
