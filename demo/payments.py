"""Demo module with deliberately planted issues, used to showcase the review agent."""

import pickle
import subprocess

import requests

API_KEY = "demo-payments-key-do-not-ship"


def charge(amount, currency="usd", tags=[]):
    tags.append(currency)
    try:
        resp = requests.post(
            "https://api.example.com/charge",
            json={"amount": amount, "currency": currency, "tags": tags},
            headers={"Authorization": f"Bearer {API_KEY}"},
            verify=False,
        )
        return resp.json()
    except:
        pass


def apply_discount(price, rule):
    # rule comes from the request query string, e.g. "price * 0.9"
    return eval(rule)


def export_report(name):
    subprocess.run(f"zip reports/{name}.zip reports/{name}.csv", shell=True)


def load_cart(blob):
    return pickle.loads(blob)


def average(values):
    return sum(values) / len(values)
