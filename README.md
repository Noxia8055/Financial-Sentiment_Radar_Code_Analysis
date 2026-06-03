# Financial-Sentiment_Radar_Code_Analysis
Sentiment Radar is an advanced desktop-based financial news aggregator and natural language processing (NLP) analyzer. It serves as a centralized local intelligence dashboard, delivering real-time insights into market posture, media sentiment, and central bank shifts.
Here is a comprehensive, production-ready `README.md` file tailored specifically for your project. This file includes all essential elements of a professional GitHub documentation repository—such as an installation index, architecture map, operational workflow guides, and configuration blueprints.

---

# Sentiment Radar - Financial Intelligence Platform

**Sentiment Radar** is an automated financial news aggregator, natural language processing (NLP) analyzer, and real-time interactive charting dashboard. Built with a modernized dark-themed graphical interface using **CustomTkinter** and **Matplotlib**, this desktop application serves as a localized command center for evaluating market velocity, media sentiment, and regulatory disclosures at a single glance.

---

## 🛠️ Prerequisites & Required Libraries

Before launching the application, you must install the runtime dependencies. The core environment relies on the following libraries:

* **`customtkinter`**: Handles the modernized, high-DPI dark-mode window framing, layouts, and input widgets.
* **`matplotlib`**: Manages native canvas drawing environments to render financial analytics charts.
* **`requests`**: Manages HTTP session pulling to extract data packets from financial endpoints.
* **`beautifulsoup4`**: Parses incoming raw XML RSS feeds and structured HTML nodes.
* **`nltk`** *(Optional/Highly Recommended)*: Empowers the primary sentiment tier using its specialized VADER framework.


* **`textblob`** *(Optional)*: Serves as the Tier-2 lexical engine layer if VADER components are absent.



### Quick Installation Command

Open your terminal or command prompt inside the project directory and run the following command to install all assets at once:

```bash
pip install customtkinter matplotlib requests beautifulsoup4 nltk textblob

```

---

## 🏗️ Core Architecture & Component Map

The platform is designed around 6 distinct Object-Oriented classes working seamlessly together:

```
  [ LIVE MARKET ENTIRES ] ──> NewsFetcher (Asynchronous RSS Multi-Streaming)[cite: 2, 3]
                                     │
                                     ▼
                     [ AUTOMATED TEXT-ANALYSIS PIPELINES ]
       ┌─────────────────────────────┼─────────────────────────────┐
       ▼                             ▼                             ▼
SentimentEngine               PolicyMoodAnalyzer            EntityExtractor
(Multi-Tier NLP Polarity)     (Hawkish/Dovish Vectors)      (RegEx Ticker Indexer)
[cite: 2, 3]                 [cite: 2, 3]                 [cite: 2, 3]
       │                             │                             │
       └─────────────────────────────┼─────────────────────────────┘
                                     │ (Data Packets Saved into NewsItem Data Class)[cite: 2, 3]
                                     ▼
                     SentimentRadarApp (CustomTkinter GUI Engine)[cite: 2, 3]

```

1. **`NewsItem`**: The structured data class blueprint that standardizes title strings, source properties, raw descriptions, and mathematical confidence values for each headline.


2. **`SentimentEngine`**: Dynamically calculates text polarity scaled from `-100.0` (Bearish) to `+100.0` (Bullish). Features an intelligent fallback cascade: locks onto NLTK VADER first, switches to TextBlob second, and falls back to a custom local keyword calculation loop if zero outside dependencies exist.


3. **`PolicyMoodAnalyzer`**: Scans macro indicators to deduce financial positioning. It scales and prints Hawkish outcomes (*inflation, tightening, rate hikes*) against Dovish outcomes (*growth support, accommodation, rate cuts*).


4. **`EntityExtractor`**: Employs deterministic regex patterns to track up to 12 corporate indices, stock market tickers, fiat currencies, and economic assets (e.g., *NVIDIA, NIFTY, RBI, FOMC*).


5. **`NewsFetcher`**: Coordinates asynchronous tracking worker threads to handle remote queries without freezing the GUI window.


6. **`SentimentRadarApp`**: Configures the main application layout, maps structural alignments, and draws live chart vectors.



---

## 📊 Workspace Interfaces & Displays

* **News Stream Grid (Left Panel):** Features scrolling data card objects. Each card updates with individual publishing information and brightens custom color badges based on its content (Green for positive sentiment, Red for negative, and Gray for neutral territory).


* **Interactive Analytics Engine (Right Panel):** Synthesizes live aggregated data through four embedded Matplotlib sub-charts:


* *Market Mood Distribution:* A precise donut chart tracking macro bullish/bearish ratios.


* *Historical Sentiment Trend:* An area line graph displaying rolling trend trajectories over your session fetching logs.


* *Bullish/Bearish Pressure:* Horizontal volumetric bar graphs highlighting baseline word weight counts.


* *Signal Stack:* Displays modular engine backend states and logging operations.




* **Deep-Dive Metric Transition:** Clicking a single headline card collapses the primary dashboard to show the full article body summary, localized entity chip-tags, direct browser hyperlink triggers, and a multi-axis **Polar Radar Chart** profiling individual sentiment weight scores.



---

## 💻 Running the Application

1. Ensure all your required packages are verified and installed.


2. Execute the main application file from your interface path terminal:


```bash

```



python "finance analyst  final.py"

```
3. Use the top dropdown option menu to filter your live target market data streams (**Global Markets**, **Indian Markets**, or **RBI/SEBI Releases**)[cite: 1, 3]. Turn on the **"Auto"** switch to automate refresh calls every 3 minutes[cite: 1].

---

## 📄 Repository Blueprint Filenames

For a comprehensive breakdown of the core code structures and architecture models, please refer to the files included in this repository:
* **`finance analyst  final.py`**: The complete production codebase for the application[cite: 1].
* **`Sentiment_Radar_Code_Analysis_Report.docx`**: The official analytical project documentation manual[cite: 2, 3].

```
