from pydantic import BaseModel
from typing import List, Dict
from collections import defaultdict
import math
import os
import json

class Product(BaseModel):
    reference: str
    description: str
    quantity: float
    unit_price: float
    total_price: float


class Invoice(BaseModel):
    invoice_number: str
    invoice_date: str
    supplier: str
    customer: str
    total_ht: float
    total_ttc: float
    products: List[Product]
    
def float_equal(a, b, tol=0.01):
    return abs(a - b) <= tol

def aggregate_invoices(invoices: List[Invoice]):
    aggregated = defaultdict(lambda: {
        "quantity": 0.0,
        "total_price": 0.0,
        "unit_prices": []
    })

    for invoice in invoices:
        for p in invoice.products:
            ref = p.reference

            aggregated[ref]["quantity"] += p.quantity
            aggregated[ref]["total_price"] += p.total_price
            aggregated[ref]["unit_prices"].append(p.unit_price)

    return aggregated

def aggregate_order(order: Invoice):

    aggregated = {}

    for p in order.products:
        aggregated[p.reference] = {
            "description": p.description,
            "quantity": p.quantity,
            "unit_price": p.unit_price,
            "total_price": p.total_price
        }

    return aggregated

def reconcile(order: Invoice, invoices: List[Invoice]):

    order_products = aggregate_order(order)
    received_products = aggregate_invoices(invoices)

    report = {
        "missing_products": [],
        "quantity_errors": [],
        "price_errors": [],
        "unexpected_products": [],
        "invoice_total_errors": [],
        "order_total_error": None
    }

    for ref, expected in order_products.items():

        if ref not in received_products:
            report["missing_products"].append({
                "reference": ref,
                "expected_quantity": expected["quantity"]
            })
            continue

        received = received_products[ref]

        if not float_equal(received["quantity"], expected["quantity"]):

            report["quantity_errors"].append({
                "reference": ref,
                "expected": expected["quantity"],
                "received": received["quantity"]
            })

        for price in received["unit_prices"]:
            if not float_equal(price, expected["unit_price"]):
                report["price_errors"].append({
                    "reference": ref,
                    "expected": expected["unit_price"],
                    "received": price
                })
                
    for ref in received_products:
        if ref not in order_products:
            report["unexpected_products"].append(ref)

    for invoice in invoices:

        calc_total = sum(p.total_price for p in invoice.products)

        if not float_equal(calc_total, invoice.total_ht):

            report["invoice_total_errors"].append({
                "invoice_number": invoice.invoice_number,
                "expected": invoice.total_ht,
                "calculated": calc_total
            })

    calc_order_total = sum(p.total_price for p in order.products)

    if not float_equal(calc_order_total, order.total_ht):
        report["order_total_error"] = {
            "expected": order.total_ht,
            "calculated": calc_order_total
        }

    return report


def load_invoice(path):
    with open(path) as f:
        data = json.load(f)
    return Invoice(**data)


order = load_invoice("extracted_json/full.json")

invoices = []
folder_path = "extracted_json"
file_to_skip = "extracted_json/full.json"
for filename in os.listdir(folder_path):
    if filename == file_to_skip:
        continue
    print(f"Processing: {filename}")
    invoices.append(load_invoice(folder_path + "/"+ filename))


report = reconcile(order, invoices)

print(json.dumps(report, indent=2))