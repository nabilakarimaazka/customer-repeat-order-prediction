import pandas as pd

PRODUCT_PATTERN = r"^\d{5}[A-Za-z]*$"

#read adn rename date
def load_raw(path="data/raw/online_retail_II.csv"):
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    return df.rename(columns={
        "Customer ID": "customer_id", "Invoice": "invoice",
        "InvoiceDate": "invoice_date", "StockCode": "stock_code",
        "Quantity": "quantity", "Price": "price", "Country": "country"
    })
   
#implemented decision from data profilling
def load_clean(path="data/raw/online_retail_II.csv")  :
    df = load_raw(path)
    df = df.dropna(subset=["customer_id"]).drop_duplicates()
    df["customer_id"] = df["customer_id"].astype(int)
    df["invoice"] = df["invoice"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str)
    df["is_cancel"] = df["invoice"].str.startswith("C")
    df["order_date"] = df["invoice_date"].dt.normalize()
    
    df = df[df["stock_code"].str.match(PRODUCT_PATTERN)]
    
    returns = df[df["is_cancel"]].copy()
    sales = df[~df["is_cancel"] & (df["quantity"] > 0) & (df["price"] > 0)].copy()
    sales["revenue"] = sales["quantity"] * sales["price"]
    
    orders = (sales.groupby(["customer_id", "order_date"])
              .agg(revenue=("revenue", "sum"),
                   n_items=("quantity", "sum"),
                   n_products=("stock_code", "nunique"),
                   n_invoices=("invoice", "nunique"),
                   country=("country", "first"))
              .reset_index())
    return orders, returns

if __name__ == "__main__":
    orders, returns = load_clean()
    print(orders.shape, returns.shape)
    print(orders["order_date"].min(), orders["order_date"].max())
    print(orders["customer_id"].nunique(), "customers") 