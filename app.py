from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("FORM DATA RECEIVED:", request.form)

        
        data = CustomData(
            CityTier=int(request.form.get("CityTier", 0)),
            WarehouseToHome=float(request.form.get("WarehouseToHome", 0)),
            HourSpendOnApp=float(request.form.get("HourSpendOnApp", 0)),
            NumberOfDeviceRegistered=int(request.form.get("NumberOfDeviceRegistered", 0)),
            SatisfactionScore=int(request.form.get("SatisfactionScore", 0)),
            NumberOfAddress=int(request.form.get("NumberOfAddress", 0)),
            Complain=int(request.form.get("Complain", 0)),
            OrderAmountHikeFromlastYear=float(request.form.get("OrderAmountHikeFromlastYear", 0)),
            CouponUsed=int(request.form.get("CouponUsed", 0)),
            OrderCount=int(request.form.get("OrderCount", 0)),
            DaySinceLastOrder=int(request.form.get("DaySinceLastOrder", 0)),
            CashbackAmount=float(request.form.get("CashbackAmount", 0)),
            Tenure=int(request.form.get("Tenure", 0)),
            Gender=request.form.get("Gender", "Male"),
            PreferredLoginDevice=request.form.get("PreferredLoginDevice", "Mobile"),
            PreferredPaymentMode=request.form.get("PreferredPaymentMode", "UPI"),
            PreferedOrderCat=request.form.get("PreferedOrderCat", "Mobile"),
            MaritalStatus=request.form.get("MaritalStatus", "Single"),
            high_price=int(request.form.get("high_price", 0))
        )

        # prediction 
        pipeline = PredictPipeline()
        df = data.get_data_as_data_frame()

        # Probability of churn (label = 1)
        churn_proba = pipeline.predict_proba(df)[0][1]

        THRESHOLD = 0.35  # tuned for churn problems

        if churn_proba >= THRESHOLD:
            prediction = f"Customer WILL Churn ({churn_proba*100:.1f}% risk)"
            churn_label = 1
        else:
            prediction = f"Customer will NOT Churn  ({churn_proba*100:.1f}% risk)"
            churn_label = 0

        return render_template(
            "result.html",
            prediction=prediction,
            churn=churn_label,
            probability=round(churn_proba * 100, 2)
        )

    except Exception as e:
        return render_template(
            "result.html",
            prediction=f"Error: {str(e)}",
            churn=-1,
            probability=0
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)