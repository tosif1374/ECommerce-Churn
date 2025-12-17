from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            data = CustomData(
                CityTier=int(request.form["CityTier"]),
                WarehouseToHome=float(request.form["WarehouseToHome"]),
                HourSpendOnApp=float(request.form["HourSpendOnApp"]),
                NumberOfDeviceRegistered=int(request.form["NumberOfDeviceRegistered"]),
                SatisfactionScore=int(request.form["SatisfactionScore"]),
                NumberOfAddress=int(request.form["NumberOfAddress"]),
                Complain=int(request.form["Complain"]),
                OrderAmountHikeFromlastYear=float(request.form["OrderAmountHikeFromlastYear"]),
                CouponUsed=int(request.form["CouponUsed"]),
                OrderCount=int(request.form["OrderCount"]),
                DaySinceLastOrder=int(request.form["DaySinceLastOrder"]),
                CashbackAmount=float(request.form["CashbackAmount"]),
                Tenure=int(request.form["Tenure"]),
                Gender=request.form["Gender"],
                PreferredLoginDevice=request.form["PreferredLoginDevice"],
                PreferredPaymentMode=request.form["PreferredPaymentMode"],
                PreferedOrderCat=request.form["PreferedOrderCat"],
                MaritalStatus=request.form["MaritalStatus"]
            )

            df = data.get_data_as_data_frame()
            pipeline = PredictPipeline()
            result = pipeline.predict(df)[0]

            prediction = "Customer WILL Churn ❌" if result == 1 else "Customer will NOT Churn ✅"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
