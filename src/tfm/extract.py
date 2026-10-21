"""Descarga de ofertas desde la API de Idealista (v3.5).

Réplica del notebook `extract-data-from-idealista-api.ipynb`, pero las
credenciales se leen de variables de entorno (IDEALISTA_API_KEY e
IDEALISTA_SECRET, o un fichero .env) en lugar de estar escritas en el código.

Uso:
    python -m tfm.extract --pages 50 --out data/raw/idealista_06.csv

Ojo: la API tiene un límite de 100 peticiones al mes.
"""
from __future__ import annotations

import argparse
import base64
import os
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.idealista.com/3.5/"
SEARCH_PARAMS = {
    "operation": "sale",
    "maxItems": "50",
    "order": "priceDown",
    "center": "40.4167,-3.70325",   # Puerta del Sol
    "distance": "15000",
    "propertyType": "homes",
    "sort": "desc",
    "maxPrice": "850000",
    "language": "es",
}


def _load_dotenv(path: Path = Path(".env")) -> None:
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def get_oauth_token() -> str:
    _load_dotenv()
    api_key, secret = os.getenv("IDEALISTA_API_KEY"), os.getenv("IDEALISTA_SECRET")
    if not api_key or not secret:
        raise RuntimeError("Define IDEALISTA_API_KEY e IDEALISTA_SECRET (ver .env.example)")
    auth = base64.b64encode(f"{api_key}:{secret}".encode()).decode()
    r = requests.post(f"{BASE_URL.replace('3.5/', '')}oauth/token",
                      headers={"Authorization": f"Basic {auth}",
                               "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"},
                      params={"grant_type": "client_credentials", "scope": "read"}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def search_page(token: str, page: int, country: str = "es") -> dict:
    params = {**SEARCH_PARAMS, "numPage": page}
    r = requests.post(f"{BASE_URL}{country}/search", params=params,
                      headers={"Authorization": f"Bearer {token}"}, timeout=30)
    r.raise_for_status()
    return r.json()


def download(pages: int, start_page: int = 1, sleep: float = 1.0) -> pd.DataFrame:
    token = get_oauth_token()
    first = search_page(token, start_page)
    total = first.get("totalPages", 1)
    frames = [pd.DataFrame(first["elementList"])]
    for page in range(start_page + 1, min(start_page + pages, total + 1)):
        time.sleep(sleep)
        frames.append(pd.DataFrame(search_page(token, page)["elementList"]))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", type=int, default=10, help="número de páginas (50 ofertas c/u)")
    ap.add_argument("--start-page", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    df = download(args.pages, args.start_page)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(args.out, index=False)
    print(f"{len(df)} ofertas guardadas en {args.out}")


if __name__ == "__main__":
    main()
