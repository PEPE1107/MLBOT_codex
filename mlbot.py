import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

API_KEY = "5a2ac6bc211648ee96f894307c4dc6af"

BASE_HEADERS = {
    "accept": "application/json",
    "CG-API-KEY": API_KEY,
}

ENDPOINTS = {
    "futures_price": "https://open-api-v4.coinglass.com/api/futures/price/history?exchange=Binance&symbol=BTCUSDT&interval=2h&limit=4500",
    "funding_rate": "https://open-api-v4.coinglass.com/api/futures/funding-rate/history?exchange=Binance&symbol=BTCUSDT&interval=2h&limit=4500",
    "funding_oi_weight": "https://open-api-v4.coinglass.com/api/futures/funding-rate/oi-weight-history?symbol=BTC&interval=2h&limit=4500",
    "open_interest": "https://open-api-v4.coinglass.com/api/futures/open-interest/aggregated-history?symbol=BTC&interval=2h&limit=4500",
    "pair_liquidation": "https://open-api-v4.coinglass.com/api/futures/liquidation/history?exchange=Binance&symbol=BTCUSDT&interval=2h&limit=4500",
    "coin_liquidation": "https://open-api-v4.coinglass.com/api/futures/liquidation/aggregated-history?exchange_list=Binance&symbol=BTC&interval=2h&limit=4500",
    "global_account_ratio": "https://open-api-v4.coinglass.com/api/futures/global-long-short-account-ratio/history?exchange=Binance&symbol=BTCUSDT&interval=2h&limit=4500",
    "top_account_ratio": "https://open-api-v4.coinglass.com/api/futures/top-long-short-account-ratio/history?exchange=binance&symbol=BTCUSDT&interval=2h&limit=4500",
    "spot_price": "https://open-api-v4.coinglass.com/api/spot/price/history?exchange=Binance&symbol=BTCUSDT&interval=2h&limit=4500",
    "orderbook_bid_ask": "https://open-api-v4.coinglass.com/api/spot/orderbook/ask-bids-history?exchange=Binance&symbol=BTCUSDT&interval=2h&limit=4500",
    "orderbook_heatmap": "https://open-api-v4.coinglass.com/api/spot/orderbook/history?exchange=Binance&symbol=BTCUSDT&interval=2h&limit=4500",
    "taker_buy_sell": "https://open-api-v4.coinglass.com/api/spot/taker-buy-sell-volume/history?exchange=Binance&symbol=BTCUSDT&interval=2h&limit=4500",
    "coinbase_premium": "https://open-api-v4.coinglass.com/api/coinbase-premium-index?interval=2h&limit=4500",
}


def fetch_json(url: str) -> dict:
    resp = requests.get(url, headers=BASE_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_df(name: str) -> pd.DataFrame:
    data = fetch_json(ENDPOINTS[name])
    if data.get("code") != "0":
        raise RuntimeError(f"failed to fetch {name}: {data}")
    df = pd.DataFrame(data["data"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.sort_values("time")
    df = df.add_prefix(f"{name}_")
    df = df.rename(columns={f"{name}_time": "time"})
    return df


def fetch_orderbook_heatmap_df() -> pd.DataFrame:
    data = fetch_json(ENDPOINTS["orderbook_heatmap"])
    if data.get("code") != "0":
        raise RuntimeError(f"failed to fetch orderbook_heatmap: {data}")
    rows = []
    for entry in data["data"]:
        timestamp = pd.to_datetime(entry[0], unit="s")
        bids = entry[1]
        asks = entry[2]
        rows.append({
            "time": timestamp,
            "orderbook_heatmap_bid_price": bids[0][0] if bids else None,
            "orderbook_heatmap_bid_quantity": bids[0][1] if bids else None,
            "orderbook_heatmap_ask_price": asks[0][0] if asks else None,
            "orderbook_heatmap_ask_quantity": asks[0][1] if asks else None,
        })
    df = pd.DataFrame(rows).sort_values("time")
    return df


def merge_data(dfs: list) -> pd.DataFrame:
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="time", how="outer")
    merged = merged.sort_values("time").reset_index(drop=True)
    return merged


def build_dataset() -> pd.DataFrame:
    dfs = [
        fetch_df("futures_price"),
        fetch_df("funding_rate"),
        fetch_df("funding_oi_weight"),
        fetch_df("open_interest"),
        fetch_df("pair_liquidation"),
        fetch_df("coin_liquidation"),
        fetch_df("global_account_ratio"),
        fetch_df("top_account_ratio"),
        fetch_df("spot_price"),
        fetch_df("orderbook_bid_ask"),
        fetch_orderbook_heatmap_df(),
        fetch_df("taker_buy_sell"),
        fetch_df("coinbase_premium"),
    ]
    return merge_data(dfs)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("time").copy()
    df["target"] = (df["futures_price_close"].shift(-1) > df["futures_price_close"]).astype(int)
    df = df.dropna()
    features = df.drop(columns=["time", "target"])
    return features, df["target"], df["time"]


def train_backtest(features: pd.DataFrame, target: pd.Series, times: pd.Series):
    X_train, X_test, y_train, y_test, time_train, time_test = train_test_split(
        features, target, times, test_size=0.2, shuffle=False
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    df = build_dataset()
    features, target, times = prepare_features(df)
    train_backtest(features, target, times)
