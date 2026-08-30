import pandas as pd


def load_data():
    complaints = pd.read_csv("data/customer_complaints.csv")
    kpi_data = pd.read_csv("data/kpi_data.csv")

    return complaints, kpi_data