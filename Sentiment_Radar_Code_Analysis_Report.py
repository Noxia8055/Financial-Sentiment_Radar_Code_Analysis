import math
import re
import threading
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Tuple

import customtkinter as ctk
import matplotlib
import requests
from bs4 import BeautifulSoup, FeatureNotFound

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


APP_BG = "#131722"
PANEL_BG = "#1c2030"
PANEL_ALT = "#202638"
BORDER = "#2a2e39"
TEXT = "#d1d4dc"
MUTED = "#868993"
GREEN = "#089981"
RED = "#f23645"
GRAY = "#787b86"
YELLOW = "#f5c542"
BLUE = "#2962ff"


@dataclass
class NewsItem:
    title: str
    link: str
    source: str
    published: str
    summary: str
    score: float
    label: str
    entities: List[str]
    policy_mood: str = "Neutral"
    policy_confidence: int = 50
    is_fallback: bool = False


class SentimentEngine:
    def __init__(self) -> None:
        self.backend = "Local finance lexicon"
        self.vader = None
        self.textblob = None

        try:
            from nltk.sentiment import SentimentIntensityAnalyzer

            analyzer = SentimentIntensityAnalyzer()
            analyzer.lexicon.update(
                {
                    "beat": 2.0,
                    "beats": 2.0,
                    "upgrade": 1.9,
                    "upgraded": 1.9,
                    "rally": 2.1,
                    "surge": 2.2,
                    "bullish": 2.4,
                    "profit": 1.7,
                    "profits": 1.7,
                    "growth": 1.8,
                    "easing": 1.4,
                    "dovish": 1.7,
                    "cut": 1.1,
                    "cuts": 1.1,
                    "miss": -1.8,
                    "misses": -1.8,
                    "downgrade": -1.9,
                    "downgraded": -1.9,
                    "selloff": -2.3,
                    "sell-off": -2.3,
                    "slump": -2.1,
                    "plunge": -2.4,
                    "bearish": -2.4,
                    "loss": -1.8,
                    "losses": -1.8,
                    "inflation": -1.3,
                    "hawkish": -1.6,
                    "default": -2.5,
                    "recession": -2.5,
                }
            )
            self.vader = analyzer
            self.backend = "NLTK VADER with finance tuning"
            return
        except Exception:
            pass

        try:
            from textblob import TextBlob

            self.textblob = TextBlob
            self.backend = "TextBlob"
        except Exception:
            self.textblob = None

        self.positive_terms = {
            "advance",
            "beat",
            "beats",
            "bullish",
            "buy",
            "dovish",
            "easing",
            "gain",
            "gains",
            "growth",
            "higher",
            "outperform",
            "profit",
            "profits",
            "rally",
            "rebound",
            "recover",
            "recovery",
            "rise",
            "rises",
            "strong",
            "surge",
            "upgrade",
            "upgraded",
            "upside",
        }
        self.negative_terms = {
            "bearish",
            "crash",
            "decline",
            "default",
            "downgrade",
            "downgraded",
            "drop",
            "fall",
            "falls",
            "fear",
            "hawkish",
            "inflation",
            "loss",
            "losses",
            "lower",
            "miss",
            "misses",
            "plunge",
            "pressure",
            "recession",
            "risk",
            "selloff",
            "slump",
            "weak",
        }

    def score(self, text: str) -> float:
        clean = " ".join(text.split())
        if not clean:
            return 0.0

        if self.vader is not None:
            return round(self.vader.polarity_scores(clean)["compound"] * 100, 1)

        if self.textblob is not None:
            try:
                return round(float(self.textblob(clean).sentiment.polarity) * 100, 1)
            except Exception:
                pass

        tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]+", clean.lower())
        if not tokens:
            return 0.0
        positive = sum(1 for token in tokens if token in self.positive_terms)
        negative = sum(1 for token in tokens if token in self.negative_terms)
        raw = positive - negative
        if raw == 0:
            return 0.0
        return round(max(-100.0, min(100.0, raw / max(4, len(tokens) / 5) * 100)), 1)

    @staticmethod
    def label_for(score: float) -> str:
        if score >= 15:
            return "Bullish"
        if score <= -15:
            return "Bearish"
        return "Neutral"

    @staticmethod
    def color_for(score: float) -> str:
        if score >= 15:
            return GREEN
        if score <= -15:
            return RED
        return GRAY


class PolicyMoodAnalyzer:
    hawkish_terms = {
        "inflation",
        "tightening",
        "rate hike",
        "hike",
        "liquidity absorption",
        "withdrawal",
        "restrictive",
        "price stability",
        "macroprudential",
        "higher rates",
        "rupee pressure",
    }
    dovish_terms = {
        "growth support",
        "accommodative",
        "easing",
        "rate cut",
        "cut",
        "liquidity infusion",
        "stimulus",
        "lower rates",
        "supportive",
        "relief",
        "credit growth",
    }

    def analyze(self, text: str, score: float = 0.0) -> Tuple[str, int, Dict[str, int]]:
        lowered = text.lower()
        hawkish = sum(1 for term in self.hawkish_terms if term in lowered)
        dovish = sum(1 for term in self.dovish_terms if term in lowered)

        if hawkish > dovish:
            mood = "Hawkish"
        elif dovish > hawkish:
            mood = "Dovish"
        else:
            mood = "Neutral"

        spread = abs(hawkish - dovish)
        confidence = int(min(95, max(50, 50 + spread * 12 + min(25, abs(score) * 0.25))))
        return mood, confidence, {"hawkish": hawkish, "dovish": dovish}


class EntityExtractor:
    known_entities = {
        "RBI",
        "SEBI",
        "FED",
        "FOMC",
        "ECB",
        "BOE",
        "BOJ",
        "NIFTY",
        "SENSEX",
        "NASDAQ",
        "DOW",
        "S&P",
        "SPX",
        "NSE",
        "BSE",
        "USD",
        "INR",
        "CRUDE",
        "OIL",
        "GOLD",
        "BITCOIN",
        "BTC",
        "TESLA",
        "TSLA",
        "NVIDIA",
        "NVDA",
        "APPLE",
        "AAPL",
        "MICROSOFT",
        "MSFT",
        "RELIANCE",
        "TCS",
        "INFOSYS",
        "HDFC",
        "ICICI",
        "SBI",
        "ADANI",
    }

    def extract(self, text: str) -> List[str]:
        found: List[str] = []
        upper_text = text.upper()

        for entity in sorted(self.known_entities):
            if entity in upper_text and entity not in found:
                found.append(entity)

        acronyms = re.findall(r"\b[A-Z]{2,6}\b", text)
        title_words = re.findall(r"\b[A-Z][a-z]{2,}(?:\s[A-Z][a-z]{2,})?\b", text)

        for token in acronyms + title_words:
            cleaned = token.strip()
            if cleaned.upper() not in {"THE", "AND", "FOR", "WITH", "FROM"} and cleaned not in found:
                found.append(cleaned)

        return found[:12]


class NewsFetcher:
    sources = {
        "Global Markets": [
            ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex/"),
        ],
        "Indian Markets": [
            ("Business Standard Markets", "https://www.business-standard.com/rss/markets-106.rss"),
            ("Economic Times Markets", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
        ],
        "RBI/SEBI Releases": [
            ("Reserve Bank of India", "https://www.rbi.org.in/Scripts/RssPublications.aspx?Id=0"),
            ("SEBI Press Releases", "https://www.sebi.gov.in/sebiweb/ajax/home/getnewslistinfo.jsp?txtPageNo=1&txtPageSize=20&txtSearch=&txtCategory=Press%20Releases"),
        ],
    }

    fallback_samples = {
        "Global Markets": [
            (
                "Global equities steady as investors weigh central bank policy and technology earnings",
                "Major indexes traded in a tight range as traders balanced resilient earnings against rate uncertainty.",
            ),
            (
                "Oil slips while dollar firms before key inflation data",
                "Energy traders turned cautious ahead of macro data that may shape expectations for monetary policy.",
            ),
            (
                "Chip stocks rally after upbeat AI infrastructure forecasts",
                "Semiconductor names gained as demand projections for data-center spending remained strong.",
            ),
        ],
        "Indian Markets": [
            (
                "Nifty closes higher led by banks and information technology shares",
                "Domestic benchmarks gained as foreign institutional flows improved and large-cap lenders advanced.",
            ),
            (
                "Rupee weakens against dollar as crude prices rise",
                "Currency traders cited stronger import demand and firmer energy prices as pressure points.",
            ),
            (
                "Reliance and HDFC Bank lift Sensex; broader market breadth remains mixed",
                "Blue-chip buying supported the index while small-cap counters saw selective profit booking.",
            ),
        ],
        "RBI/SEBI Releases": [
            (
                "RBI reiterates price stability priority while keeping liquidity conditions balanced",
                "The release emphasized inflation vigilance, orderly market conditions, and durable growth.",
            ),
            (
                "SEBI strengthens disclosure norms for listed entities",
                "The regulator announced additional investor-protection measures and clearer reporting timelines.",
            ),
            (
                "RBI announces liquidity infusion through variable rate repo operation",
                "The operation is intended to support short-term banking system liquidity and credit conditions.",
            ),
        ],
    }

    def __init__(self, sentiment: SentimentEngine, policy: PolicyMoodAnalyzer, extractor: EntityExtractor) -> None:
        self.sentiment = sentiment
        self.policy = policy
        self.extractor = extractor
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0 Safari/537.36"
                ),
                "Accept": "application/rss+xml, application/xml, text/xml, text/html",
                "Timeout": "10",
            }
        )

    def fetch(self, source_name: str, limit: int = 18) -> Tuple[List[NewsItem], str]:
        errors: List[str] = []
        all_items: List[NewsItem] = []

        for source_label, url in self.sources[source_name]:
            try:
                response = self.session.get(url, timeout=12)
                response.raise_for_status()
                parsed = self._parse_response(response.text, source_label, source_name)
                all_items.extend(parsed)
                if len(all_items) >= limit:
                    break
            except Exception as exc:
                errors.append(f"{source_label}: {exc}")

        if all_items:
            return all_items[:limit], "Live headlines loaded."

        fallback = self._fallback_items(source_name)
        message = "Live fetch failed; showing offline sample data."
        if errors:
            message += " " + " | ".join(errors[:2])
        return fallback, message

    def _parse_response(self, text: str, source_label: str, selected_source: str) -> List[NewsItem]:
        soup = self._soup(text)
        items = soup.find_all("item")
        if not items:
            items = soup.find_all("entry")

        parsed: List[NewsItem] = []
        for item in items[:25]:
            title = self._tag_text(item, ["title"])
            link = self._extract_link(item)
            published = self._tag_text(item, ["pubDate", "published", "updated", "dc:date"])
            summary = self._tag_text(item, ["description", "summary", "content:encoded"])
            if not title:
                continue
            parsed.append(self._make_item(title, summary, link, source_label, published, selected_source))

        if parsed:
            return parsed

        if selected_source == "RBI/SEBI Releases":
            return self._parse_regulator_html(soup, source_label, selected_source)

        return []

    def _soup(self, text: str) -> BeautifulSoup:
        try:
            return BeautifulSoup(text, "xml")
        except FeatureNotFound:
            return BeautifulSoup(text, "html.parser")

    @staticmethod
    def _tag_text(item: BeautifulSoup, names: List[str]) -> str:
        for name in names:
            tag = item.find(name)
            if tag and tag.get_text(strip=True):
                return BeautifulSoup(tag.get_text(" ", strip=True), "html.parser").get_text(" ", strip=True)
        return ""

    @staticmethod
    def _extract_link(item: BeautifulSoup) -> str:
        link_tag = item.find("link")
        if link_tag:
            href = link_tag.get("href")
            if href:
                return href
            text = link_tag.get_text(strip=True)
            if text:
                return text
        guid = item.find("guid")
        return guid.get_text(strip=True) if guid else ""

    def _parse_regulator_html(self, soup: BeautifulSoup, source_label: str, selected_source: str) -> List[NewsItem]:
        parsed: List[NewsItem] = []
        for anchor in soup.find_all("a", href=True)[:30]:
            title = anchor.get_text(" ", strip=True)
            if len(title) < 18:
                continue
            href = anchor["href"]
            if href.startswith("/"):
                href = "https://www.rbi.org.in" + href
            parsed.append(self._make_item(title, "", href, source_label, "", selected_source))
            if len(parsed) >= 15:
                break
        return parsed

    def _make_item(
        self,
        title: str,
        summary: str,
        link: str,
        source: str,
        published: str,
        selected_source: str,
        is_fallback: bool = False,
    ) -> NewsItem:
        text = f"{title}. {summary}"
        score = self.sentiment.score(text)
        label = self.sentiment.label_for(score)
        mood, confidence, _ = self.policy.analyze(text, score)
        if selected_source != "RBI/SEBI Releases" and mood == "Neutral":
            mood = "Risk-On" if score >= 15 else "Risk-Off" if score <= -15 else "Neutral"
        return NewsItem(
            title=title,
            link=link,
            source=source,
            published=self._format_date(published),
            summary=summary or "No expanded description was published in the feed for this item.",
            score=score,
            label=label,
            entities=self.extractor.extract(text),
            policy_mood=mood,
            policy_confidence=confidence,
            is_fallback=is_fallback,
        )

    def _fallback_items(self, source_name: str) -> List[NewsItem]:
        items: List[NewsItem] = []
        for title, summary in self.fallback_samples[source_name]:
            items.append(
                self._make_item(
                    title=title,
                    summary=summary,
                    link="",
                    source="Offline Sample",
                    published=datetime.now().strftime("%d %b %Y, %H:%M"),
                    selected_source=source_name,
                    is_fallback=True,
                )
            )
        return items

    @staticmethod
    def _format_date(raw: str) -> str:
        if not raw:
            return datetime.now().strftime("%d %b %Y, %H:%M")
        try:
            parsed = parsedate_to_datetime(raw)
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone()
            return parsed.strftime("%d %b %Y, %H:%M")
        except Exception:
            return raw[:40]


class SentimentRadarApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("Sentiment Radar - Financial Intelligence Platform")
        self.geometry("1380x850")
        self.minsize(1120, 720)
        self.configure(fg_color=APP_BG)

        self.sentiment = SentimentEngine()
        self.policy = PolicyMoodAnalyzer()
        self.extractor = EntityExtractor()
        self.fetcher = NewsFetcher(self.sentiment, self.policy, self.extractor)

        self.current_items: List[NewsItem] = []
        self.history: List[Tuple[str, float]] = []
        self.fetching = False
        self.auto_refresh_job: Optional[str] = None

        self._build_layout()
        self._show_empty_analytics()
        self.after(300, self.fetch_news)

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.header = ctk.CTkFrame(self, fg_color=APP_BG, height=74, corner_radius=0)
        self.header.grid(row=0, column=0, sticky="ew")
        self.header.grid_columnconfigure(1, weight=1)

        title_box = ctk.CTkFrame(self.header, fg_color="transparent")
        title_box.grid(row=0, column=0, padx=(22, 12), pady=12, sticky="w")

        ctk.CTkLabel(
            title_box,
            text="SENTIMENT RADAR",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=TEXT,
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            title_box,
            text="Financial Intelligence Platform",
            font=ctk.CTkFont(size=12),
            text_color=MUTED,
        ).grid(row=1, column=0, sticky="w", pady=(1, 0))

        controls = ctk.CTkFrame(self.header, fg_color="transparent")
        controls.grid(row=0, column=2, padx=22, pady=12, sticky="e")

        self.source_var = ctk.StringVar(value="Global Markets")
        self.source_menu = ctk.CTkOptionMenu(
            controls,
            values=list(NewsFetcher.sources.keys()),
            variable=self.source_var,
            width=205,
            height=38,
            fg_color=PANEL_BG,
            button_color=PANEL_ALT,
            button_hover_color=BORDER,
            dropdown_fg_color=PANEL_BG,
            dropdown_hover_color=PANEL_ALT,
            text_color=TEXT,
            command=lambda _: self._source_changed(),
        )
        self.source_menu.grid(row=0, column=0, padx=(0, 10))

        self.auto_refresh = ctk.BooleanVar(value=False)
        self.auto_switch = ctk.CTkSwitch(
            controls,
            text="Auto",
            variable=self.auto_refresh,
            command=self._toggle_auto_refresh,
            progress_color=GREEN,
            button_color=TEXT,
            fg_color=BORDER,
            text_color=TEXT,
        )
        self.auto_switch.grid(row=0, column=1, padx=(0, 10))

        self.fetch_button = ctk.CTkButton(
            controls,
            text="FETCH",
            width=110,
            height=38,
            fg_color=GREEN,
            hover_color="#0aa88e",
            text_color="#ffffff",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.fetch_news,
        )
        self.fetch_button.grid(row=0, column=2)

        self.body = ctk.CTkFrame(self, fg_color=APP_BG, corner_radius=0)
        self.body.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 12))
        self.body.grid_columnconfigure(0, weight=4, uniform="main")
        self.body.grid_columnconfigure(1, weight=6, uniform="main")
        self.body.grid_rowconfigure(0, weight=1)

        self.left_panel = ctk.CTkFrame(self.body, fg_color=PANEL_BG, border_color=BORDER, border_width=1, corner_radius=8)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.left_panel.grid_columnconfigure(0, weight=1)
        self.left_panel.grid_rowconfigure(1, weight=1)

        left_head = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        left_head.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 10))
        left_head.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            left_head,
            text="NEWS FEED & SENTIMENT STREAM",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT,
        ).grid(row=0, column=0, sticky="w")
        self.item_count_label = ctk.CTkLabel(left_head, text="0 items", font=ctk.CTkFont(size=12), text_color=MUTED)
        self.item_count_label.grid(row=0, column=1, sticky="e")

        self.news_scroll = ctk.CTkScrollableFrame(
            self.left_panel,
            fg_color="transparent",
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=GRAY,
        )
        self.news_scroll.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.news_scroll.grid_columnconfigure(0, weight=1)

        self.right_panel = ctk.CTkFrame(self.body, fg_color=PANEL_BG, border_color=BORDER, border_width=1, corner_radius=8)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(0, weight=1)

        self.footer = ctk.CTkFrame(self, fg_color=APP_BG, height=42, corner_radius=0)
        self.footer.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 12))
        self.footer.grid_columnconfigure(0, weight=1)
        self.status_label = ctk.CTkLabel(
            self.footer,
            text=f"Ready. Sentiment backend: {self.sentiment.backend}",
            text_color=MUTED,
            anchor="w",
            font=ctk.CTkFont(size=12),
        )
        self.status_label.grid(row=0, column=0, sticky="ew")
        self.policy_label = ctk.CTkLabel(
            self.footer,
            text="RBI/SEBI Policy Stance: Neutral | Confidence: 50%",
            text_color=TEXT,
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.policy_label.grid(row=0, column=1, sticky="e")

    def _source_changed(self) -> None:
        self.fetch_news()

    def _toggle_auto_refresh(self) -> None:
        if self.auto_refresh.get():
            self._schedule_auto_refresh()
        elif self.auto_refresh_job:
            self.after_cancel(self.auto_refresh_job)
            self.auto_refresh_job = None

    def _schedule_auto_refresh(self) -> None:
        if self.auto_refresh_job:
            self.after_cancel(self.auto_refresh_job)
        self.auto_refresh_job = self.after(180000, self._auto_fetch)

    def _auto_fetch(self) -> None:
        self.fetch_news()
        if self.auto_refresh.get():
            self._schedule_auto_refresh()

    def fetch_news(self) -> None:
        if self.fetching:
            return

        self.fetching = True
        self.fetch_button.configure(text="LOADING", state="disabled", fg_color=GRAY)
        self._set_status("Fetching live financial headlines...")

        source = self.source_var.get()
        worker = threading.Thread(target=self._fetch_worker, args=(source,), daemon=True)
        worker.start()

    def _fetch_worker(self, source: str) -> None:
        try:
            items, message = self.fetcher.fetch(source)
        except Exception as exc:
            items = self.fetcher._fallback_items(source)
            message = f"Unexpected fetch error; showing offline sample data. {exc}"

        self.after(0, lambda: self._finish_fetch(items, message))

    def _finish_fetch(self, items: List[NewsItem], message: str) -> None:
        self.fetching = False
        self.current_items = items

        average = self._average_score(items)
        self.history.append((datetime.now().strftime("%H:%M"), average))
        self.history = self.history[-18:]

        self.fetch_button.configure(text="FETCH", state="normal", fg_color=GREEN)
        self._populate_news(items)
        self._show_analytics()
        self._update_policy_footer(items)
        self._set_status(message)

    def _populate_news(self, items: List[NewsItem]) -> None:
        for child in self.news_scroll.winfo_children():
            child.destroy()

        self.item_count_label.configure(text=f"{len(items)} items")

        if not items:
            ctk.CTkLabel(
                self.news_scroll,
                text="No headlines available.",
                text_color=MUTED,
                font=ctk.CTkFont(size=14),
            ).grid(row=0, column=0, padx=18, pady=28)
            return

        for index, item in enumerate(items):
            self._create_news_card(index, item)

    def _create_news_card(self, row: int, item: NewsItem) -> None:
        color = SentimentEngine.color_for(item.score)
        card = ctk.CTkFrame(
            self.news_scroll,
            fg_color=APP_BG,
            border_color=BORDER,
            border_width=1,
            corner_radius=8,
        )
        card.grid(row=row, column=0, sticky="ew", padx=2, pady=6)
        card.grid_columnconfigure(1, weight=1)

        badge = ctk.CTkFrame(card, width=68, height=64, fg_color=color, corner_radius=6)
        badge.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="ns")
        badge.grid_propagate(False)
        ctk.CTkLabel(
            badge,
            text=f"{item.score:+.0f}",
            text_color="#ffffff",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).place(relx=0.5, rely=0.5, anchor="center")

        title = ctk.CTkLabel(
            card,
            text=item.title,
            text_color=TEXT,
            font=ctk.CTkFont(size=14, weight="bold"),
            justify="left",
            anchor="w",
            wraplength=390,
        )
        title.grid(row=0, column=1, sticky="ew", padx=(0, 12), pady=(11, 2))

        meta = f"{item.source} | {item.published} | {item.label}"
        if item.is_fallback:
            meta += " | Offline"
        ctk.CTkLabel(
            card,
            text=meta,
            text_color=MUTED,
            font=ctk.CTkFont(size=11),
            anchor="w",
            justify="left",
        ).grid(row=1, column=1, sticky="ew", padx=(0, 12), pady=(0, 11))

        for widget in (card, badge, title):
            widget.bind("<Button-1>", lambda _event, selected=item: self._show_detail(selected))
            widget.bind("<Enter>", lambda _event, frame=card: frame.configure(border_color=GRAY))
            widget.bind("<Leave>", lambda _event, frame=card: frame.configure(border_color=BORDER))

    def _show_empty_analytics(self) -> None:
        self._clear_right_panel()
        frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        frame.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            frame,
            text="DYNAMIC INTERACTIVE ANALYTICS WORKSPACE",
            text_color=TEXT,
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            frame,
            text="Loading market intelligence...",
            text_color=MUTED,
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=1, column=0)

    def _show_analytics(self) -> None:
        self._clear_right_panel()
        items = self.current_items

        container = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        container.grid(row=0, column=0, sticky="nsew", padx=18, pady=16)
        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)
        container.grid_rowconfigure(1, weight=3)
        container.grid_rowconfigure(2, weight=2)

        header = ctk.CTkFrame(container, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        header.grid_columnconfigure(0, weight=1)

        avg = self._average_score(items)
        mood = self._market_mood(avg)
        mood_color = SentimentEngine.color_for(avg)
        ctk.CTkLabel(
            header,
            text="REAL-TIME ANALYTICS SUITE",
            text_color=TEXT,
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header,
            text=f"Market Mood: {mood}  |  Avg Score: {avg:+.1f}",
            text_color=mood_color,
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=1, sticky="e")

        donut_frame = self._chart_frame(container, 1, 0, "Market Mood Distribution")
        trend_frame = self._chart_frame(container, 1, 1, "Historical Sentiment Trend")
        gauge_frame = self._chart_frame(container, 2, 0, "Bullish/Bearish Pressure")
        table_frame = ctk.CTkFrame(container, fg_color=APP_BG, border_color=BORDER, border_width=1, corner_radius=8)
        table_frame.grid(row=2, column=1, sticky="nsew", padx=(9, 0), pady=(9, 0))
        table_frame.grid_columnconfigure(0, weight=1)

        self._draw_donut(donut_frame, items)
        self._draw_trend(trend_frame, items)
        self._draw_pressure(gauge_frame, items)
        self._draw_summary_table(table_frame, items)

    def _chart_frame(self, parent: ctk.CTkFrame, row: int, col: int, title: str) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(parent, fg_color=APP_BG, border_color=BORDER, border_width=1, corner_radius=8)
        frame.grid(row=row, column=col, sticky="nsew", padx=(0, 9) if col == 0 else (9, 0), pady=(0, 9) if row == 1 else (9, 0))
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(
            frame,
            text=title,
            text_color=TEXT,
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 0))
        return frame

    def _draw_donut(self, parent: ctk.CTkFrame, items: List[NewsItem]) -> None:
        counts = [
            sum(1 for item in items if item.score >= 15),
            sum(1 for item in items if -15 < item.score < 15),
            sum(1 for item in items if item.score <= -15),
        ]
        if not any(counts):
            counts = [0, 1, 0]

        fig = Figure(figsize=(4.4, 3.2), dpi=110, facecolor=APP_BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(APP_BG)
        ax.pie(
            counts,
            colors=[GREEN, GRAY, RED],
            startangle=90,
            counterclock=False,
            wedgeprops={"width": 0.38, "edgecolor": APP_BG, "linewidth": 3},
        )
        ax.text(0, 0.08, self._market_mood(self._average_score(items)), ha="center", va="center", color=TEXT, fontsize=14, weight="bold")
        ax.text(0, -0.16, f"{self._average_score(items):+.1f}", ha="center", va="center", color=MUTED, fontsize=11)
        ax.axis("equal")
        self._embed_figure(parent, fig)

    def _draw_trend(self, parent: ctk.CTkFrame, items: List[NewsItem]) -> None:
        fig = Figure(figsize=(4.5, 3.2), dpi=110, facecolor=APP_BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(APP_BG)

        if len(self.history) >= 2:
            x_labels = [stamp for stamp, _score in self.history]
            y = [score for _stamp, score in self.history]
        else:
            x_labels = [str(i + 1) for i in range(max(1, len(items)))]
            y = [item.score for item in items] or [0]

        x = list(range(len(y)))
        ax.plot(x, y, color=BLUE, linewidth=2.2, marker="o", markersize=4)
        ax.fill_between(x, y, [0] * len(y), color=BLUE, alpha=0.17)
        ax.axhline(0, color=BORDER, linewidth=1)
        ax.set_ylim(-100, 100)
        ax.set_xticks(x[:: max(1, len(x) // 5)])
        ax.set_xticklabels(x_labels[:: max(1, len(x) // 5)], rotation=0, color=MUTED, fontsize=8)
        ax.tick_params(axis="y", colors=MUTED, labelsize=8)
        ax.grid(True, color=BORDER, linewidth=0.7, alpha=0.6)
        for spine in ax.spines.values():
            spine.set_color(APP_BG)
        self._embed_figure(parent, fig)

    def _draw_pressure(self, parent: ctk.CTkFrame, items: List[NewsItem]) -> None:
        positive = sum(max(0, item.score) for item in items)
        negative = abs(sum(min(0, item.score) for item in items))
        neutral = sum(1 for item in items if -15 < item.score < 15) * 12
        values = [positive, neutral, negative]
        labels = ["Bullish", "Neutral", "Bearish"]
        colors = [GREEN, GRAY, RED]

        fig = Figure(figsize=(4.4, 2.4), dpi=110, facecolor=APP_BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(APP_BG)
        bars = ax.barh(labels, values, color=colors, height=0.48)
        ax.tick_params(axis="x", colors=MUTED, labelsize=8)
        ax.tick_params(axis="y", colors=TEXT, labelsize=9)
        ax.grid(True, axis="x", color=BORDER, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color(APP_BG)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f"{width:.0f}", va="center", color=MUTED, fontsize=8)
        self._embed_figure(parent, fig)

    def _draw_summary_table(self, parent: ctk.CTkFrame, items: List[NewsItem]) -> None:
        ctk.CTkLabel(
            parent,
            text="Signal Stack",
            text_color=TEXT,
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 8))

        stats = [
            ("Bullish", sum(1 for item in items if item.score >= 15), GREEN),
            ("Neutral", sum(1 for item in items if -15 < item.score < 15), GRAY),
            ("Bearish", sum(1 for item in items if item.score <= -15), RED),
            ("Backend", self.sentiment.backend, BLUE),
        ]

        for index, (label, value, color) in enumerate(stats, start=1):
            row = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=6)
            row.grid(row=index, column=0, sticky="ew", padx=14, pady=4)
            row.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(row, text=label, text_color=MUTED, font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w", padx=10, pady=8)
            ctk.CTkLabel(row, text=str(value), text_color=color, font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=1, sticky="e", padx=10, pady=8)

    def _show_detail(self, item: NewsItem) -> None:
        self._clear_right_panel()

        container = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        container.grid(row=0, column=0, sticky="nsew", padx=18, pady=16)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(3, weight=1)

        top = ctk.CTkFrame(container, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        top.grid_columnconfigure(1, weight=1)

        color = SentimentEngine.color_for(item.score)
        badge = ctk.CTkFrame(top, width=86, height=66, fg_color=color, corner_radius=8)
        badge.grid(row=0, column=0, padx=(0, 14), sticky="n")
        badge.grid_propagate(False)
        ctk.CTkLabel(
            badge,
            text=f"{item.score:+.0f}",
            text_color="#ffffff",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).place(relx=0.5, rely=0.44, anchor="center")
        ctk.CTkLabel(
            badge,
            text=item.label,
            text_color="#ffffff",
            font=ctk.CTkFont(size=10, weight="bold"),
        ).place(relx=0.5, rely=0.73, anchor="center")

        title_box = ctk.CTkFrame(top, fg_color="transparent")
        title_box.grid(row=0, column=1, sticky="ew")
        ctk.CTkLabel(
            title_box,
            text=item.title,
            text_color=TEXT,
            font=ctk.CTkFont(size=19, weight="bold"),
            wraplength=760,
            justify="left",
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            title_box,
            text=f"{item.source} | {item.published} | Mood: {item.policy_mood} ({item.policy_confidence}%)",
            text_color=MUTED,
            font=ctk.CTkFont(size=12),
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", pady=(5, 0))

        action_row = ctk.CTkFrame(container, fg_color="transparent")
        action_row.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        action_row.grid_columnconfigure(0, weight=1)
        ctk.CTkButton(
            action_row,
            text="BACK TO CHARTS",
            width=142,
            height=34,
            fg_color=PANEL_ALT,
            hover_color=BORDER,
            text_color=TEXT,
            command=self._show_analytics,
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(
            action_row,
            text="OPEN SOURCE",
            width=128,
            height=34,
            fg_color=GREEN if item.link else GRAY,
            hover_color="#0aa88e",
            text_color="#ffffff",
            state="normal" if item.link else "disabled",
            command=lambda: webbrowser.open(item.link) if item.link else None,
        ).grid(row=0, column=1, sticky="e")

        entity_panel = ctk.CTkFrame(container, fg_color=APP_BG, border_color=BORDER, border_width=1, corner_radius=8)
        entity_panel.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        entity_panel.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            entity_panel,
            text="AUTOMATED IMPACT RADIAL ENTITIES",
            text_color=TEXT,
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 8))

        chips = ctk.CTkFrame(entity_panel, fg_color="transparent")
        chips.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 14))
        entities = item.entities or ["Market", "Macro", "Liquidity"]
        for index, entity in enumerate(entities):
            chip = ctk.CTkLabel(
                chips,
                text=entity,
                text_color=TEXT,
                fg_color=PANEL_ALT,
                corner_radius=14,
                padx=12,
                pady=5,
                font=ctk.CTkFont(size=11, weight="bold"),
            )
            chip.grid(row=index // 5, column=index % 5, padx=(0, 8), pady=(0, 8), sticky="w")

        lower = ctk.CTkFrame(container, fg_color="transparent")
        lower.grid(row=3, column=0, sticky="nsew")
        lower.grid_columnconfigure(0, weight=3)
        lower.grid_columnconfigure(1, weight=2)
        lower.grid_rowconfigure(0, weight=1)

        text_panel = ctk.CTkFrame(lower, fg_color=APP_BG, border_color=BORDER, border_width=1, corner_radius=8)
        text_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 9))
        text_panel.grid_columnconfigure(0, weight=1)
        text_panel.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(
            text_panel,
            text="EXPANDED TEXT PAYLOAD BODY",
            text_color=TEXT,
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 8))

        text_box = ctk.CTkTextbox(
            text_panel,
            fg_color=PANEL_BG,
            border_color=BORDER,
            border_width=1,
            text_color=TEXT,
            wrap="word",
            font=ctk.CTkFont(size=14),
            corner_radius=6,
        )
        text_box.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        body_text = (
            f"{item.summary}\n\n"
            f"Sentiment score: {item.score:+.1f} / 100\n"
            f"Signal classification: {item.label}\n"
            f"Policy or market mood: {item.policy_mood}\n"
            f"Confidence: {item.policy_confidence}%"
        )
        text_box.insert("1.0", body_text)
        text_box.configure(state="disabled")

        chart_panel = ctk.CTkFrame(lower, fg_color=APP_BG, border_color=BORDER, border_width=1, corner_radius=8)
        chart_panel.grid(row=0, column=1, sticky="nsew", padx=(9, 0))
        chart_panel.grid_columnconfigure(0, weight=1)
        chart_panel.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(
            chart_panel,
            text="Impact Profile",
            text_color=TEXT,
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 0))
        self._draw_detail_chart(chart_panel, item)

    def _draw_detail_chart(self, parent: ctk.CTkFrame, item: NewsItem) -> None:
        score = max(-100, min(100, item.score))
        policy_strength = item.policy_confidence if item.policy_mood != "Neutral" else 50
        entity_strength = min(100, 20 + len(item.entities) * 8)

        fig = Figure(figsize=(3.8, 3.2), dpi=110, facecolor=APP_BG)
        ax = fig.add_subplot(111, polar=True)
        ax.set_facecolor(APP_BG)

        labels = ["Sentiment", "Policy", "Entities"]
        values = [(score + 100) / 2, policy_strength, entity_strength]
        angles = [n / float(len(labels)) * 2 * math.pi for n in range(len(labels))]
        values += values[:1]
        angles += angles[:1]

        ax.plot(angles, values, color=GREEN if score >= 0 else RED, linewidth=2)
        ax.fill(angles, values, color=GREEN if score >= 0 else RED, alpha=0.2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, color=TEXT, fontsize=9)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["25", "50", "75", "100"], color=MUTED, fontsize=7)
        ax.grid(color=BORDER, alpha=0.7)
        ax.spines["polar"].set_color(BORDER)
        self._embed_figure(parent, fig)

    def _embed_figure(self, parent: ctk.CTkFrame, fig: Figure) -> None:
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.configure(bg=APP_BG, highlightthickness=0)
        widget.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

    def _clear_right_panel(self) -> None:
        for child in self.right_panel.winfo_children():
            child.destroy()

    def _average_score(self, items: List[NewsItem]) -> float:
        if not items:
            return 0.0
        return round(sum(item.score for item in items) / len(items), 1)

    @staticmethod
    def _market_mood(score: float) -> str:
        if score >= 35:
            return "Strongly Bullish"
        if score >= 15:
            return "Bullish"
        if score <= -35:
            return "Strongly Bearish"
        if score <= -15:
            return "Bearish"
        return "Neutral"

    def _update_policy_footer(self, items: List[NewsItem]) -> None:
        if not items:
            self.policy_label.configure(text="RBI/SEBI Policy Stance: Neutral | Confidence: 50%")
            return

        if self.source_var.get() == "RBI/SEBI Releases":
            moods = [item.policy_mood for item in items]
            mood = max(set(moods), key=moods.count)
            confidence = int(sum(item.policy_confidence for item in items) / len(items))
            hawkish_count = moods.count("Hawkish")
            dovish_count = moods.count("Dovish")
            self.policy_label.configure(
                text=f"RBI/SEBI Policy Stance: {mood} | Confidence: {confidence}% | Hawkish: {hawkish_count} | Dovish: {dovish_count}"
            )
        else:
            avg = self._average_score(items)
            self.policy_label.configure(text=f"Market Mood: {self._market_mood(avg)} | Average Sentiment: {avg:+.1f}")

    def _set_status(self, text: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.status_label.configure(text=f"{timestamp} | {text}")


def main() -> None:
    app = SentimentRadarApp()
    app.mainloop()


if __name__ == "__main__":
    main()
