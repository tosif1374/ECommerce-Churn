import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_snapdeal():
    url = "https://www.snapdeal.com/search?keyword=iphone%2014"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-IN,en;q=0.9"
    }

    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    products = soup.select("div.product-tuple-listing")

    data = []

    for p in products:
        title = p.select_one("p.product-title")
        price = p.select_one("span.product-price")

        if title and price:
            data.append({
                "product_name": title.text.strip(),
                "price": price.text.strip()
            })

    df = pd.DataFrame(data)

    # Save inside artifacts (since you already use it)
    df.to_csv("artifacts/snapdeal_products.csv", index=False)

    return df
