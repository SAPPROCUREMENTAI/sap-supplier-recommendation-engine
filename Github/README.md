# SAP AI Supplier Recommendation Engine

> Built on SAP BTP AI Core · RandomForest ML · Flask REST API · S/4HANA Data Model

A production-ready proof of concept demonstrating how SAP BTP AI Core can power intelligent supplier recommendation for procurement teams. Built by [Nikhil Bhosale](https://procurearc.in) as part of the [@SAPPROCUREMENTAI](https://www.youtube.com/@SAPPROCUREMENTAI) YouTube series on SAP Procurement AI use cases.

---

## What This Does

Traditional supplier selection in SAP S/4HANA relies on manual evaluation, price-only comparisons, and buyer intuition. This PoC replaces that with a machine learning model trained on 11 vendor KPIs drawn directly from S/4HANA master data tables -- scoring every vendor across material groups and returning ranked recommendations via a REST API.

A buyer submits a material group and quantity. The engine returns a ranked list of vendors with composite scores, individual KPI breakdowns, and a confidence indicator -- in milliseconds.

---

## Architecture Overview

```
S/4HANA Tables          BTP AI Core              Fiori / App
--------------          -----------              -----------
LFA1  (Vendor Master)                            
EKKO  (PO Headers)   →  RandomForest Model   →   REST API   →   UI / Integration
EKPO  (PO Line Items)    11-Feature Scoring       Flask /v1
Vendor Scorecard         Per Material Group       recommend
```

The model is containerised via Docker and deployable directly to SAP AI Core. The Flask API exposes three endpoints. The demo UI runs as a dark-themed browser application with vendor cards, score bars, KPI metrics, and colour-coded status indicators.

---

## S/4HANA Data Model

| Table | Fields Used | Purpose |
|---|---|---|
| LFA1 | LIFNR, NAME1, LAND1, KTOKK | Vendor master -- identity and classification |
| EKKO | EBELN, LIFNR, BEDAT, EKORG | Purchase order headers -- procurement organisation |
| EKPO | EBELN, EBELP, MATNR, MENGE, NETPR, WERKS | PO line items -- material, quantity, price |
| Vendor Scorecard | 11 KPI columns | Composite scoring input |

---

## KPIs Used in Scoring

The RandomForest model is trained on 11 features extracted from historical PO data and vendor master records:

- On-time delivery rate
- Quality rejection rate
- Price competitiveness index
- Lead time consistency
- Invoice accuracy rate
- Contract compliance score
- Sustainability rating
- Financial stability score
- Geographic risk index
- Relationship tenure
- Capacity utilisation

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | RandomForest (scikit-learn) |
| API | Flask + flask-cors |
| Containerisation | Docker (SAP AI Core compatible) |
| Demo UI | HTML / CSS / JavaScript |
| Mock Data | CSV (LFA1, EKKO, EKPO, Vendor Scorecard) |
| Target Platform | SAP BTP AI Core |

---

## API Endpoints

```
GET  /health                    Health check
POST /v1/recommend              Submit material group, get ranked vendor list
GET  /v1/vendor/{id}            Fetch individual vendor scorecard
```

### Sample Request

```json
POST /v1/recommend
{
  "material_group": "MG-ELECT",
  "quantity": 500,
  "plant": "1000"
}
```

### Sample Response

```json
{
  "recommendations": [
    {
      "vendor_id": "V-007",
      "vendor_name": "Volterra Power Systems",
      "composite_score": 87.4,
      "rank": 1,
      "kpis": {
        "on_time_delivery": 94.2,
        "quality_rejection_rate": 1.1,
        "price_competitiveness": 82.0
      },
      "confidence": "HIGH"
    }
  ]
}
```

---

## Repo Structure

```
sap-supplier-recommendation-engine/
├── README.md
├── api/
│   └── app.py                  Flask REST API
├── model/
│   └── train_model.py          RandomForest training script
├── data/
│   ├── lfa1_vendors.csv        Mock vendor master (LFA1)
│   ├── ekko_po_headers.csv     Mock PO headers (EKKO)
│   ├── ekpo_po_items.csv       Mock PO line items (EKPO)
│   └── vendor_scorecard.csv   11-KPI scorecard (20 vendors)
├── ui/
│   └── index.html              Dark-themed demo UI
└── docker/
    └── Dockerfile              SAP AI Core deployment ready
```

---

## Running Locally

### Prerequisites

- Python 3.9 or above
- pip

### Setup

```bash
git clone https://github.com/SAPPROCUREMENTAI/sap-supplier-recommendation-engine.git
cd sap-supplier-recommendation-engine

pip install flask flask-cors scikit-learn pandas numpy

# Train the model
python model/train_model.py

# Start the API
python api/app.py
```

API runs on `http://localhost:8080`

### Demo UI

```bash
cd ui
python -m http.server 3000
```

Open `http://localhost:3000` in your browser.

---

## SAP AI Core Deployment

The included Dockerfile packages the Flask API and trained model for deployment on SAP BTP AI Core.

```bash
docker build -t supplier-recommendation-engine .
docker run -p 8080:8080 supplier-recommendation-engine
```

For full SAP AI Core deployment steps including Artifact registration, Docker registry push, and Serving configuration, refer to the SAP AI Core documentation or watch the deployment walkthrough on the [@SAPPROCUREMENTAI](https://www.youtube.com/@SAPPROCUREMENTAI) YouTube channel.

---

## Mock Data

All vendor names in the dataset are fictional -- Meridian Industrial Supplies, Volterra Power Systems, Precisia Instrumentation, and others. No real company data is used. The data structure mirrors production S/4HANA table schemas and is safe for public demonstration.

---

## Related Use Cases

This repo is part of a broader SAP BTP AI use case series:

- **Supplier Recommendation Engine** (this repo)
- Price Anomaly Detection -- Isolation Forest + Z-score on EKPO price history
- RAG Contract Intelligence -- Document grounding on SAP AI Core
- PR to PO Automation -- AI-driven purchase requisition processing
- Joule in Procurement -- Conversational AI for procurement workflows

All use cases are covered in depth on the [@SAPPROCUREMENTAI YouTube channel](https://www.youtube.com/@SAPPROCUREMENTAI).

---

## Architecture Deep Dive

The full architecture document covering the BTP AI stack, S/4HANA integration points, model governance approach, and 4-week implementation roadmap is available as part of the SAP MM AI Bundle at [procurearc.in](https://procurearc.in).

---

## About

Built by **Nikhil Bhosale** -- SAP BTP AI Architect with 21 years of enterprise IT experience and 16 years specialising in SAP Procurement transformation. Holds 5 SAP certifications including BTP Solution Architect and Generative AI Developer.

- YouTube: [@SAPPROCUREMENTAI](https://www.youtube.com/@SAPPROCUREMENTAI)
- Website: [procurearc.in](https://procurearc.in)
- LinkedIn: [Nikhil Bhosale](https://www.linkedin.com/in/nikhilbhosale)

---

## License

MIT License. Free to use, fork, and build on. Attribution appreciated.
